import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class ChartGenerator:
    def __init__(self, data):
        self.data = data
    
    def create_pareto_chart(self, category='platform', metric='total_engagement'):
        """Create Pareto chart for given category and metric"""
        grouped = self.data.groupby(category)[metric].sum().sort_values(ascending=False)
        cumulative = grouped.cumsum() / grouped.sum() * 100
        
        fig = go.Figure()
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=grouped.index,
            y=grouped.values,
            name=metric,
            marker_color='skyblue'
        ))
        
        # Line chart (cumulative percentage)
        fig.add_trace(go.Scatter(
            x=grouped.index,
            y=cumulative.values,
            name='Cumulative %',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'Pareto Chart: {metric} by {category}',
            xaxis_title=category,
            yaxis_title=metric,
            yaxis2=dict(
                title='Cumulative Percentage',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            legend=dict(x=1.1, y=1),
            template='plotly_white'
        )
        
        return fig
    
    def create_scatter_plot(self, x_col='views', y_col='engagement_rate', 
                          color_col='platform', size_col='total_engagement'):
        """Create interactive scatter plot"""
        fig = px.scatter(
            self.data,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            hover_data=['post_type', 'post_length', 'likes', 'comments', 'shares'],
            title=f'{y_col} vs {x_col} by {color_col}',
            template='plotly_white'
        )
        
        # Add trend line
        fig.update_traces(
            marker=dict(opacity=0.7),
            selector=dict(mode='markers')
        )
        
        return fig
    
    def create_dual_axis_chart(self, platform='All'):
        """Create dual axis chart for engagement metrics"""
        if platform != 'All':
            data = self.data[self.data['platform'] == platform]
        else:
            data = self.data
        
        grouped = data.groupby('post_type').agg({
            'views': 'mean',
            'engagement_rate': 'mean',
            'total_engagement': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for views
        fig.add_trace(
            go.Bar(
                x=grouped['post_type'],
                y=grouped['views'],
                name='Avg Views',
                marker_color='lightblue'
            ),
            secondary_y=False
        )
        
        # Line chart for engagement rate
        fig.add_trace(
            go.Scatter(
                x=grouped['post_type'],
                y=grouped['engagement_rate'],
                name='Avg Engagement Rate',
                mode='lines+markers',
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=f'Dual Axis Chart: Views vs Engagement Rate ({platform})',
            xaxis_title='Post Type',
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Avg Views", secondary_y=False)
        fig.update_yaxes(title_text="Avg Engagement Rate", secondary_y=True)
        
        return fig
    
    def create_waterfall_chart(self):
        """Create waterfall chart for engagement decomposition"""
        platforms = self.data['platform'].unique()
        base_value = 0
        measures = ['relative'] * len(platforms)
        measures[0] = 'total'
        measures[-1] = 'total'
        
        values = []
        for platform in platforms:
            platform_data = self.data[self.data['platform'] == platform]
            values.append(platform_data['total_engagement'].sum())
        
        fig = go.Figure(go.Waterfall(
            name="Engagement",
            orientation="v",
            measure=measures,
            x=platforms,
            textposition="outside",
            text=[f"{v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Waterfall Chart: Total Engagement by Platform",
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_donut_chart(self, metric='platform'):
        """Create donut chart for distribution"""
        if metric == 'platform':
            counts = self.data['platform'].value_counts()
            title = 'Platform Distribution'
        else:
            counts = self.data['post_type'].value_counts()
            title = 'Post Type Distribution'
        
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=.5,
            hoverinfo='label+percent+value',
            textinfo='percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            annotations=[dict(text=metric, x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def create_comparison_matrix(self):
        """Create heatmap comparison matrix"""
        pivot_data = self.data.pivot_table(
            values='engagement_rate',
            index='platform',
            columns='post_type',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            pivot_data,
            text_auto='.3f',
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title='Comparison Matrix: Avg Engagement Rate by Platform & Post Type',
            labels=dict(x="Post Type", y="Platform", color="Engagement Rate")
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis_nticks=len(pivot_data.columns),
            yaxis_nticks=len(pivot_data.index)
        )
        
        return fig
    
    def create_growth_trend_chart(self, metric='engagement_rate'):
        """Create growth trend chart over time"""
        self.data['date'] = pd.to_datetime(self.data['date'])
        daily_data = self.data.set_index('date')[metric].resample('D').mean()
        
        fig = go.Figure()
        
        # Line chart
        fig.add_trace(go.Scatter(
            x=daily_data.index,
            y=daily_data.values,
            mode='lines',
            name=f'Daily {metric}',
            line=dict(color='blue', width=2)
        ))
        
        # Moving average
        ma_7 = daily_data.rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=ma_7.index,
            y=ma_7.values,
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Growth Trend: {metric} Over Time',
            xaxis_title='Date',
            yaxis_title=metric,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_cumulative_performance_chart(self, metric='total_engagement'):
        """Create cumulative performance tracker"""
        self.data = self.data.sort_values('date')
        cumulative = self.data[metric].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=cumulative,
            mode='lines',
            name=f'Cumulative {metric}',
            fill='tozeroy',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title=f'Cumulative Performance: {metric}',
            xaxis_title='Date',
            yaxis_title=f'Cumulative {metric}',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_seasonality_heatmap(self):
        """Create heatmap for seasonality/what-if analysis"""
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['hour'] = self.data['date'].dt.hour
        self.data['day_of_week'] = self.data['date'].dt.day_name()
        
        heatmap_data = self.data.pivot_table(
            values='engagement_rate',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        ).fillna(0)
        
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(days_order)
        
        fig = px.imshow(
            heatmap_data,
            text_auto='.3f',
            aspect="auto",
            color_continuous_scale='Viridis',
            title='Seasonality Heatmap: Engagement Rate by Day & Hour',
            labels=dict(x="Hour of Day", y="Day of Week", color="Engagement Rate")
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis_nticks=24,
            yaxis_nticks=7
        )
        
        return fig