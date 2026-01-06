import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def detect_outliers(self, features=['views', 'likes', 'comments', 'shares']):
        """Detect outliers using Isolation Forest"""
        # Prepare data
        X = self.data[features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(X_scaled)
        
        # Add outlier labels to data
        self.data['is_outlier'] = outliers
        self.data['outlier_score'] = iso_forest.decision_function(X_scaled)
        
        return self.data
    
    def get_outlier_plot_data(self):
        """Prepare data for outlier visualization"""
        outlier_data = self.data[self.data['is_outlier'] == -1]
        normal_data = self.data[self.data['is_outlier'] == 1]
        
        return outlier_data, normal_data
    
    def calculate_margin_profitability(self):
        """Calculate margin/profitability metrics"""
        # Simulating cost and revenue for demonstration
        self.data['cost_per_post'] = np.random.uniform(10, 100, len(self.data))
        self.data['revenue_per_view'] = np.random.uniform(0.001, 0.01, len(self.data))
        
        # Calculate metrics
        self.data['total_revenue'] = self.data['views'] * self.data['revenue_per_view']
        self.data['profit'] = self.data['total_revenue'] - self.data['cost_per_post']
        self.data['profit_margin'] = self.data['profit'] / self.data['total_revenue'] * 100
        self.data['roi'] = (self.data['profit'] / self.data['cost_per_post']) * 100
        
        # Replace infinite values
        self.data['profit_margin'] = self.data['profit_margin'].replace([np.inf, -np.inf], np.nan)
        self.data['roi'] = self.data['roi'].replace([np.inf, -np.inf], np.nan)
        
        return self.data
    
    def apply_apriori_algorithm(self, min_support=0.1):
        """Apply Apriori algorithm for association rule mining"""
        # Create binary features for association rules
        data_encoded = pd.get_dummies(self.data[['platform', 'post_type']])
        
        # Convert to boolean
        data_encoded = data_encoded.astype(bool)
        
        # Apply Apriori
        frequent_itemsets = apriori(data_encoded, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules.sort_values('confidence', ascending=False)
            return rules.head(10)
        
        return pd.DataFrame()
    
    def identify_hvc(self, features=['views', 'likes', 'comments', 'shares', 'engagement_rate']):
        """Identify High Value Customers/Groups using clustering"""
        # Prepare features
        X = self.data[features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster centroids
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Identify HVC cluster (highest average engagement)
        cluster_engagement = []
        for i in range(3):
            cluster_mask = (clusters == i)
            if cluster_mask.sum() > 0:
                avg_engagement = self.data.loc[cluster_mask, 'engagement_rate'].mean()
                cluster_engagement.append(avg_engagement)
            else:
                cluster_engagement.append(0)
        
        hvc_cluster = np.argmax(cluster_engagement)
        
        # Add cluster labels
        self.data['cluster'] = clusters
        self.data['is_hvc'] = (clusters == hvc_cluster).astype(int)
        
        # Prepare ROC curve data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.data['is_hvc'], test_size=0.3, random_state=42
        )
        
        # Train classifier
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        
        # Get probabilities for ROC
        y_probs = clf.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'thresholds': thresholds,
            'hvc_count': (self.data['is_hvc'] == 1).sum(),
            'cluster_centroids': centroids,
            'cluster_labels': ['Low', 'Medium', 'High']
        }
    
    def create_performance_matrix(self):
        """Create performance matrix for platforms"""
        performance_data = self.data.groupby('platform').agg({
            'views': 'mean',
            'engagement_rate': 'mean',
            'total_engagement': 'sum',
            'likes': 'mean',
            'comments': 'mean',
            'shares': 'mean'
        }).reset_index()
        
        # Calculate performance scores
        for metric in ['views', 'engagement_rate', 'total_engagement']:
            performance_data[f'{metric}_score'] = (
                performance_data[metric] / performance_data[metric].max() * 100
            )
        
        # Overall performance score
        performance_data['overall_score'] = (
            performance_data['views_score'] * 0.4 +
            performance_data['engagement_rate_score'] * 0.4 +
            performance_data['total_engagement_score'] * 0.2
        )
        
        return performance_data
    
    def what_if_analysis(self, scenario_type='seasonal', params=None):
        """Perform what-if analysis"""
        if scenario_type == 'seasonal':
            # Seasonal adjustment scenario
            seasonal_multiplier = params.get('multiplier', 1.2)
            
            # Simulate seasonal effect
            simulated = self.data.copy()
            simulated['simulated_views'] = simulated['views'] * seasonal_multiplier
            simulated['simulated_engagement'] = simulated['total_engagement'] * seasonal_multiplier
            
            return {
                'current_avg_views': self.data['views'].mean(),
                'simulated_avg_views': simulated['simulated_views'].mean(),
                'current_total_engagement': self.data['total_engagement'].sum(),
                'simulated_total_engagement': simulated['simulated_engagement'].sum(),
                'percent_change': (seasonal_multiplier - 1) * 100
            }
        
        elif scenario_type == 'content':
            # Content type optimization scenario
            best_post_type = self.data.groupby('post_type')['engagement_rate'].mean().idxmax()
            
            optimized = self.data.copy()
            optimized_mask = optimized['post_type'] == best_post_type
            
            # Simulate shifting all posts to best type
            optimized.loc[~optimized_mask, 'simulated_engagement_rate'] = (
                optimized.loc[~optimized_mask, 'engagement_rate'] * 1.5
            )
            optimized.loc[optimized_mask, 'simulated_engagement_rate'] = (
                optimized.loc[optimized_mask, 'engagement_rate']
            )
            
            return {
                'best_post_type': best_post_type,
                'current_avg_engagement': self.data['engagement_rate'].mean(),
                'optimized_avg_engagement': optimized['simulated_engagement_rate'].mean(),
                'improvement_percentage': (
                    (optimized['simulated_engagement_rate'].mean() / 
                     self.data['engagement_rate'].mean() - 1) * 100
                )
            }
        
        return {}