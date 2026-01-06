# app.py - Complete Social Media Dashboard in Single File
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Data Loader Class
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
                    st.error(f"Missing required column: {col}")
                    return
            
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

# Chart Generator Class
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

# Data Analyzer Class
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
        if params is None:
            params = {}
            
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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'charts' not in st.session_state:
    st.session_state.charts = None

# Sidebar
with st.sidebar:
    st.title("📊 Dashboard Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)", 
        type=['csv'],
        help="Upload a CSV file with social media engagement data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            loader = DataLoader()
            data = loader.load_data(df=pd.read_csv(uploaded_file))
            
            # Initialize analyzer and charts
            analyzer = DataAnalyzer(data)
            charts = ChartGenerator(data)
            
            # Store in session state
            st.session_state.data = data
            st.session_state.analyzer = analyzer
            st.session_state.charts = charts
            st.session_state.data_loaded = True
            
            st.success("✅ Data loaded successfully!")
            
            # Show summary
            with st.expander("📋 Data Summary"):
                st.write(f"**Total Posts:** {len(data):,}")
                st.write(f"**Platforms:** {', '.join(data['platform'].unique())}")
                st.write(f"**Post Types:** {', '.join(data['post_type'].unique())}")
                st.write(f"**Date Range:** {data['date'].min().date()} to {data['date'].max().date()}")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    if st.session_state.data_loaded:
        st.divider()
        
        # Platform filter
        data = st.session_state.data
        platforms = ['All'] + list(data['platform'].unique())
        selected_platform = st.selectbox(
            "Select Platform",
            platforms,
            help="Filter data by platform"
        )
        
        # Metric selection
        metrics = ['engagement_rate', 'views', 'likes', 'comments', 'shares', 'total_engagement']
        selected_metric = st.selectbox(
            "Primary Metric",
            metrics,
            help="Select primary metric for analysis"
        )
        
        # Analysis parameters
        st.divider()
        st.subheader("Analysis Parameters")
        
        min_support = st.slider(
            "Apriori Min Support",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Minimum support for association rules"
        )
        
        contamination = st.slider(
            "Outlier Contamination",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Expected proportion of outliers"
        )
        
        st.divider()
        st.caption("Made with ❤️ using Streamlit")

# Main content
if st.session_state.data_loaded:
    data = st.session_state.data
    analyzer = st.session_state.analyzer
    charts = st.session_state.charts
    
    # Header
    st.markdown('<h1 class="main-header">📈 Social Media Engagement Dashboard</h1>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_posts = len(data)
        platform_posts = len(data[data['platform'] == selected_platform]) if selected_platform != 'All' else total_posts
        delta_posts = f"{platform_posts}" if selected_platform != 'All' else None
        st.metric("Total Posts", f"{total_posts:,}", delta=delta_posts)
    
    with col2:
        avg_engagement = data['engagement_rate'].mean()
        platform_engagement = data[data['platform'] == selected_platform]['engagement_rate'].mean() if selected_platform != 'All' else avg_engagement
        delta_engagement = f"{platform_engagement:.2%}" if selected_platform != 'All' else None
        st.metric("Avg Engagement Rate", f"{avg_engagement:.2%}", delta=delta_engagement)
    
    with col3:
        total_views = data['views'].sum()
        platform_views = data[data['platform'] == selected_platform]['views'].sum() if selected_platform != 'All' else total_views
        delta_views = f"{platform_views:,.0f}" if selected_platform != 'All' else None
        st.metric("Total Views", f"{total_views:,.0f}", delta=delta_views)
    
    with col4:
        total_engagement = data['total_engagement'].sum()
        platform_engagement_total = data[data['platform'] == selected_platform]['total_engagement'].sum() if selected_platform != 'All' else total_engagement
        delta_engagement_total = f"{platform_engagement_total:,.0f}" if selected_platform != 'All' else None
        st.metric("Total Engagement", f"{total_engagement:,.0f}", delta=delta_engagement_total)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview Charts",
        "🔍 Advanced Analysis",
        "📈 Performance Tracking",
        "🤖 Machine Learning",
        "🎯 What-If Analysis"
    ])
    
    with tab1:
        # Overview Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">📈 Pareto Analysis</h3>', unsafe_allow_html=True)
            pareto_fig = charts.create_pareto_chart('platform', 'total_engagement')
            st.plotly_chart(pareto_fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="sub-header">🎯 Donut Chart Distribution</h3>', unsafe_allow_html=True)
            donut_fig = charts.create_donut_chart('platform')
            st.plotly_chart(donut_fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('<h3 class="sub-header">📊 Comparison Matrix</h3>', unsafe_allow_html=True)
            matrix_fig = charts.create_comparison_matrix()
            st.plotly_chart(matrix_fig, use_container_width=True)
        
        with col4:
            st.markdown('<h3 class="sub-header">🌊 Waterfall Chart</h3>', unsafe_allow_html=True)
            waterfall_fig = charts.create_waterfall_chart()
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        st.markdown('<h3 class="sub-header">📈 Dual Axis Chart</h3>', unsafe_allow_html=True)
        dual_fig = charts.create_dual_axis_chart(selected_platform)
        st.plotly_chart(dual_fig, use_container_width=True)
        
        st.markdown('<h3 class="sub-header">🎯 Scatter Plot Analysis</h3>', unsafe_allow_html=True)
        scatter_fig = charts.create_scatter_plot('views', 'engagement_rate', 'platform', 'total_engagement')
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with tab2:
        # Advanced Analysis
        st.markdown('<h3 class="sub-header">🔍 Outlier Detection</h3>', unsafe_allow_html=True)
        
        # Detect outliers
        data_with_outliers = analyzer.detect_outliers()
        outlier_data, normal_data = analyzer.get_outlier_plot_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Outliers", len(outlier_data))
            st.metric("Outlier Percentage", f"{(len(outlier_data)/len(data)*100):.1f}%")
        
        with col2:
            # Outlier visualization
            fig = go.Figure()
            
            # Normal points
            fig.add_trace(go.Scatter(
                x=normal_data['views'],
                y=normal_data['engagement_rate'],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # Outlier points
            fig.add_trace(go.Scatter(
                x=outlier_data['views'],
                y=outlier_data['engagement_rate'],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=12, opacity=0.8)
            ))
            
            fig.update_layout(
                title='Outlier Detection: Engagement Rate vs Views',
                xaxis_title='Views',
                yaxis_title='Engagement Rate',
                template='plotly_white',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h3 class="sub-header">💰 Margin Profitability Analysis</h3>', unsafe_allow_html=True)
        
        # Calculate profitability
        profitability_data = analyzer.calculate_margin_profitability()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_profit = profitability_data['profit'].mean()
            st.metric("Avg Profit per Post", f"${avg_profit:,.2f}")
        
        with col2:
            avg_roi = profitability_data['roi'].mean()
            st.metric("Avg ROI", f"{avg_roi:.1f}%")
        
        with col3:
            avg_margin = profitability_data['profit_margin'].mean()
            st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
        
        # Profitability by platform
        profit_by_platform = profitability_data.groupby('platform').agg({
            'profit': 'mean',
            'roi': 'mean',
            'profit_margin': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=profit_by_platform['platform'],
                y=profit_by_platform['profit'],
                name='Avg Profit',
                marker_color='lightgreen'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=profit_by_platform['platform'],
                y=profit_by_platform['roi'],
                name='Avg ROI',
                mode='lines+markers',
                line=dict(color='orange', width=2)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Profitability Analysis by Platform',
            xaxis_title='Platform',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Avg Profit ($)", secondary_y=False)
        fig.update_yaxes(title_text="Avg ROI (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Performance Tracking
        st.markdown('<h3 class="sub-header">📈 Growth Trends</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            trend_fig = charts.create_growth_trend_chart('engagement_rate')
            st.plotly_chart(trend_fig, use_container_width=True)
        
        with col2:
            trend_fig2 = charts.create_growth_trend_chart('views')
            st.plotly_chart(trend_fig2, use_container_width=True)
        
        st.markdown('<h3 class="sub-header">📊 Cumulative Performance</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            cumulative_fig = charts.create_cumulative_performance_chart('total_engagement')
            st.plotly_chart(cumulative_fig, use_container_width=True)
        
        with col2:
            cumulative_fig2 = charts.create_cumulative_performance_chart('views')
            st.plotly_chart(cumulative_fig2, use_container_width=True)
        
        st.markdown('<h3 class="sub-header">🏆 Performance Matrix</h3>', unsafe_allow_html=True)
        
        performance_matrix = analyzer.create_performance_matrix()
        
        fig = go.Figure(data=[
            go.Bar(name='Views Score', x=performance_matrix['platform'], y=performance_matrix['views_score']),
            go.Bar(name='Engagement Score', x=performance_matrix['platform'], y=performance_matrix['engagement_rate_score']),
            go.Bar(name='Total Engagement Score', x=performance_matrix['platform'], y=performance_matrix['total_engagement_score']),
            go.Scatter(name='Overall Score', x=performance_matrix['platform'], y=performance_matrix['overall_score'],
                      mode='lines+markers', line=dict(color='black', width=3))
        ])
        
        fig.update_layout(
            title='Performance Matrix by Platform',
            xaxis_title='Platform',
            yaxis_title='Score (0-100)',
            barmode='group',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.dataframe(
            performance_matrix.style.background_gradient(subset=['overall_score'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab4:
        # Machine Learning
        st.markdown('<h3 class="sub-header">🤖 Apriori Algorithm - Association Rules</h3>', unsafe_allow_html=True)
        
        # Apply Apriori
        rules = analyzer.apply_apriori_algorithm(min_support=min_support)
        
        if not rules.empty:
            st.write("**Top Association Rules:**")
            
            # Display rules in a nice format
            for idx, rule in rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
                with col1:
                    st.write(f"**If:** {antecedents}")
                with col2:
                    st.write(f"**Then:** {consequents}")
                with col3:
                    st.write(f"Confidence: {rule['confidence']:.2%}")
                with col4:
                    st.write(f"Lift: {rule['lift']:.2f}")
            
            # Visualization of rules
            fig = go.Figure(data=[
                go.Scatter(
                    x=rules['support'],
                    y=rules['confidence'],
                    mode='markers',
                    marker=dict(
                        size=rules['lift'] * 10,
                        color=rules['lift'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Lift')
                    ),
                    text=[f"Rule {i+1}" for i in range(len(rules))],
                    hoverinfo='text+x+y'
                )
            ])
            
            fig.update_layout(
                title='Association Rules: Support vs Confidence',
                xaxis_title='Support',
                yaxis_title='Confidence',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant association rules found. Try lowering the minimum support.")
        
        st.markdown('<h3 class="sub-header">🎯 High Value Customer (HVC) Identification</h3>', unsafe_allow_html=True)
        
        # Identify HVC
        hvc_results = analyzer.identify_hvc()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("HVC Posts Identified", hvc_results['hvc_count'])
            st.metric("HVC Percentage", f"{(hvc_results['hvc_count']/len(data)*100):.1f}%")
            
            # Display HVC characteristics
            st.write("**HVC Cluster Characteristics:**")
            hvc_data = data[data['is_hvc'] == 1]
            avg_metrics = hvc_data[['views', 'engagement_rate', 'total_engagement']].mean()
            
            for metric, value in avg_metrics.items():
                st.write(f"- Avg {metric.replace('_', ' ').title()}: {value:.2f}")
        
        with col2:
            # ROC Curve
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hvc_results['fpr'],
                y=hvc_results['tpr'],
                mode='lines',
                name=f'ROC curve (AUC = {hvc_results["roc_auc"]:.3f})',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random classifier',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve - HVC Identification',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster visualization
        st.markdown('<h3 class="sub-header">👥 Customer Segmentation</h3>', unsafe_allow_html=True)
        
        fig = px.scatter(
            data,
            x='views',
            y='engagement_rate',
            color='cluster',
            size='total_engagement',
            hover_data=['platform', 'post_type'],
            title='Customer Segmentation by Engagement',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        # What-If Analysis
        st.markdown('<h3 class="sub-header">🎯 Scenario Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_type = st.selectbox(
                "Select Scenario",
                ['seasonal', 'content'],
                format_func=lambda x: 'Seasonal Effect' if x == 'seasonal' else 'Content Optimization'
            )
        
        with col2:
            if scenario_type == 'seasonal':
                multiplier = st.slider(
                    "Seasonal Multiplier",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.2,
                    step=0.1,
                    help="Adjustment factor for seasonal effects"
                )
                params = {'multiplier': multiplier}
            else:
                params = {}
        
        # Run what-if analysis
        results = analyzer.what_if_analysis(scenario_type, params)
        
        if results:
            col1, col2, col3 = st.columns(3)
            
            if scenario_type == 'seasonal':
                with col1:
                    st.metric(
                        "Current Avg Views",
                        f"{results['current_avg_views']:,.0f}"
                    )
                with col2:
                    st.metric(
                        "Simulated Avg Views",
                        f"{results['simulated_avg_views']:,.0f}",
                        delta=f"{results['percent_change']:.1f}%"
                    )
                with col3:
                    st.metric(
                        "Total Engagement Impact",
                        f"{results['simulated_total_engagement']:,.0f}",
                        delta=f"{(results['simulated_total_engagement']/results['current_total_engagement']-1)*100:.1f}%"
                    )
            
            elif scenario_type == 'content':
                with col1:
                    st.metric(
                        "Best Post Type",
                        results['best_post_type']
                    )
                with col2:
                    st.metric(
                        "Current Engagement Rate",
                        f"{results['current_avg_engagement']:.2%}"
                    )
                with col3:
                    st.metric(
                        "Optimized Engagement Rate",
                        f"{results['optimized_avg_engagement']:.2%}",
                        delta=f"{results['improvement_percentage']:.1f}%"
                    )
        
        st.markdown('<h3 class="sub-header">🌡️ Seasonality Heatmap</h3>', unsafe_allow_html=True)
        
        heatmap_fig = charts.create_seasonality_heatmap()
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # What-if simulation interface
        st.markdown('<h3 class="sub-header">🔮 Custom What-If Simulation</h3>', unsafe_allow_html=True)
        
        with st.expander("Run Custom Simulation"):
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            
            with sim_col1:
                views_change = st.slider("Views Change (%)", -50, 100, 10)
            
            with sim_col2:
                engagement_change = st.slider("Engagement Change (%)", -50, 100, 15)
            
            with sim_col3:
                platform_filter = st.multiselect(
                    "Apply to Platforms",
                    data['platform'].unique(),
                    default=data['platform'].unique()
                )
            
            if st.button("Run Simulation"):
                # Simulate changes
                sim_data = data.copy()
                platform_mask = sim_data['platform'].isin(platform_filter)
                
                sim_data.loc[platform_mask, 'simulated_views'] = (
                    sim_data.loc[platform_mask, 'views'] * (1 + views_change/100)
                )
                sim_data.loc[platform_mask, 'simulated_engagement'] = (
                    sim_data.loc[platform_mask, 'total_engagement'] * (1 + engagement_change/100)
                )
                
                # Calculate results
                current_total = data.loc[platform_mask, 'views'].sum()
                simulated_total = sim_data.loc[platform_mask, 'simulated_views'].sum()
                
                st.success(f"""
                **Simulation Results:**
                - Current Total Views: {current_total:,.0f}
                - Simulated Total Views: {simulated_total:,.0f}
                - Change: {((simulated_total/current_total)-1)*100:.1f}%
                """)
        
        # Export results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Analysis Report"):
                # Create summary report
                report_data = {
                    'Total Posts': len(data),
                    'Avg Engagement Rate': f"{data['engagement_rate'].mean():.2%}",
                    'Total Views': f"{data['views'].sum():,.0f}",
                    'Platform Distribution': str(data['platform'].value_counts().to_dict()),
                    'Top Performing Platform': data.groupby('platform')['engagement_rate'].mean().idxmax(),
                    'Best Post Type': data.groupby('post_type')['engagement_rate'].mean().idxmax()
                }
                
                report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
                
                # Convert to CSV
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name="social_media_analysis_report.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔄 Reset Analysis"):
                st.session_state.data_loaded = False
                st.session_state.data = None
                st.rerun()

else:
    # Welcome screen
    st.markdown('<h1 class="main-header">📊 Social Media Engagement Dashboard</h1>', unsafe_allow_html=True)
    
    # Sample data for demo
    sample_csv = """platform,post_type,post_length,views,likes,comments,shares,engagement_rate
Facebook,Text,62,91660,2968,276,346,0.039166484835260744
Instagram,Video,104,113115,4164,632,406,0.04598859567696592
Facebook,Video,46,36043,3125,188,100,0.09469245068390533
Twitter,Image,75,133108,3203,452,384,0.030343780989872886
Instagram,Image,83,247591,19425,1114,2863,0.09451878299291977
Twitter,Text,110,12273,1200,222,111,0.1249083353703251
Facebook,Image,39,124886,5970,948,578,0.06002274073955447"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3193/3193000.png", width=200)
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2>Welcome to the Social Media Analytics Dashboard</h2>
        <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
            Upload your social media engagement data to unlock powerful insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    features = [
        {"icon": "📈", "title": "Pareto Analysis", "desc": "Identify key contributors to engagement"},
        {"icon": "🎯", "title": "Scatter Plots", "desc": "Visualize correlations between metrics"},
        {"icon": "📊", "title": "Comparison Matrix", "desc": "Compare performance across dimensions"},
        {"icon": "🔍", "title": "Outlier Detection", "desc": "Find anomalies in your data"},
        {"icon": "📈", "title": "Growth Trends", "desc": "Track performance over time"},
        {"icon": "🤖", "title": "ML Algorithms", "desc": "Apriori and clustering analysis"},
        {"icon": "🎯", "title": "HVC Identification", "desc": "Find high-value customer groups"},
        {"icon": "🔮", "title": "What-If Analysis", "desc": "Simulate different scenarios"}
    ]
    
    cols = st.columns(4)
    for idx, feature in enumerate(features):
        with cols[idx % 4]:
            st.markdown(f"""
            <div style='padding: 1rem; border-radius: 10px; background-color: #F8FAFC; margin-bottom: 1rem;'>
                <div style='font-size: 2rem; text-align: center;'>{feature['icon']}</div>
                <h4 style='text-align: center;'>{feature['title']}</h4>
                <p style='text-align: center; color: #666; font-size: 0.9rem;'>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📋 How to Use This Dashboard"):
        st.markdown("""
        1. **Upload your CSV file** using the file uploader in the sidebar
        2. **Ensure your data has these columns**:
           - `platform` (e.g., Facebook, Instagram, Twitter)
           - `post_type` (e.g., Text, Video, Image)
           - `post_length` (numeric)
           - `views` (numeric)
           - `likes` (numeric)
           - `comments` (numeric)
           - `shares` (numeric)
           - `engagement_rate` (numeric)
        
        3. **Navigate through the tabs** to explore different analyses
        4. **Use filters** in the sidebar to focus on specific platforms
        5. **Download reports** for further analysis
        
        **Sample data format:**
        ```csv
        platform,post_type,post_length,views,likes,comments,shares,engagement_rate
        Facebook,Text,62,91660,2968,276,346,0.039166
        Instagram,Video,104,113115,4164,632,406,0.045988
        ```
        """)
    
    # Demo button
    if st.button("🚀 Try with Sample Data", type="primary"):
        # Create sample data
        sample_df = pd.read_csv(pd.compat.StringIO(sample_csv))
        
        # Load sample data
        loader = DataLoader()
        data = loader.load_data(df=sample_df)
        
        analyzer = DataAnalyzer(data)
        charts = ChartGenerator(data)
        
        st.session_state.data = data
        st.session_state.analyzer = analyzer
        st.session_state.charts = charts
        st.session_state.data_loaded = True
        
        st.rerun()
    
    # Quick upload section
    st.divider()
    st.markdown("### 📤 Quick Start")
    
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        st.info("Upload your CSV file in the sidebar to get started!")
    
    with upload_col2:
        # Download sample CSV
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="sample_social_media_data.csv",
            mime="text/csv"
        )