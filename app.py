import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import DataLoader
from utils.charts import ChartGenerator
from utils.analysis import DataAnalyzer

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
        platforms = ['All'] + list(st.session_state.data['platform'].unique())
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
        st.metric(
            "Total Posts",
            f"{len(data):,}",
            delta=f"{len(data[data['platform'] == selected_platform]):,}" if selected_platform != 'All' else None
        )
    
    with col2:
        avg_engagement = data['engagement_rate'].mean()
        st.metric(
            "Avg Engagement Rate",
            f"{avg_engagement:.2%}",
            delta=f"{data[data['platform'] == selected_platform]['engagement_rate'].mean():.2%}" if selected_platform != 'All' else None
        )
    
    with col3:
        total_views = data['views'].sum()
        st.metric(
            "Total Views",
            f"{total_views:,.0f}",
            delta=f"{data[data['platform'] == selected_platform]['views'].sum():,.0f}" if selected_platform != 'All' else None
        )
    
    with col4:
        total_engagement = data['total_engagement'].sum()
        st.metric(
            "Total Engagement",
            f"{total_engagement:,.0f}",
            delta=f"{data[data['platform'] == selected_platform]['total_engagement'].sum():,.0f}" if selected_platform != 'All' else None
        )
    
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
                        f"${results['simulated_total_engagement']:,.0f}",
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
                - Estimated Impact: ${(simulated_total - current_total) * 0.005:,.2f} (at $0.005 per view)
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
                    'Platform Distribution': data['platform'].value_counts().to_dict(),
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
        # Load sample data
        sample_data = pd.read_csv("social_media_engagement_dataset.csv")
        
        loader = DataLoader()
        data = loader.load_data(df=sample_data)
        
        analyzer = DataAnalyzer(data)
        charts = ChartGenerator(data)
        
        st.session_state.data = data
        st.session_state.analyzer = analyzer
        st.session_state.charts = charts
        st.session_state.data_loaded = True
        
        st.rerun()