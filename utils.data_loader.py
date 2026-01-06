import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataLoader:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path=None, df=None):
        """Load data from CSV file or DataFrame"""
        if file_path:
            self.data = pd.read_csv(file_path)
        elif df is not None:
            self.data = df
        else:
            raise ValueError("Either file_path or df must be provided")
        
        self._preprocess_data()
        return self.data
    
    def _preprocess_data(self):
        """Preprocess the data"""
        if self.data is not None:
            # Ensure all required columns exist
            required_cols = ['platform', 'post_type', 'post_length', 'views', 
                           'likes', 'comments', 'shares', 'engagement_rate']
            
            for col in required_cols:
                if col not in self.data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert numeric columns
            numeric_cols = ['post_length', 'views', 'likes', 'comments', 
                          'shares', 'engagement_rate']
            for col in numeric_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Add derived columns
            self.data['total_engagement'] = self.data['likes'] + self.data['comments'] + self.data['shares']
            self.data['engagement_per_view'] = self.data['total_engagement'] / self.data['views']
            self.data['engagement_per_view'] = self.data['engagement_per_view'].replace([np.inf, -np.inf], np.nan)
            
            # Add date for time series (simulated)
            self.data['date'] = pd.date_range(
                start='2023-01-01', 
                periods=len(self.data), 
                freq='H'
            )
    
    def get_platform_data(self, platform=None):
        """Get data for specific platform or all platforms"""
        if platform and platform != 'All':
            return self.data[self.data['platform'] == platform]
        return self.data
    
    def get_time_series_data(self, metric='engagement_rate', platform='All'):
        """Prepare time series data"""
        data = self.get_platform_data(platform)
        ts_data = data.set_index('date')[metric].resample('D').mean()
        return ts_data.fillna(method='ffill')
    
    def get_summary_stats(self):
        """Get summary statistics"""
        return {
            'total_posts': len(self.data),
            'avg_engagement_rate': self.data['engagement_rate'].mean(),
            'avg_views': self.data['views'].mean(),
            'total_engagement': self.data['total_engagement'].sum(),
            'platform_distribution': self.data['platform'].value_counts().to_dict(),
            'post_type_distribution': self.data['post_type'].value_counts().to_dict()
        }
