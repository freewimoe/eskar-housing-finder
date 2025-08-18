"""
ESKAR Production Analytics Dashboard
Professional analytics and monitoring for production deployment.

Features:
- Real - time performance monitoring
- ML model performance tracking
- User behavior analytics
- System health monitoring
- Business metrics dashboard

Author: Friedrich - Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from api.user_feedback import ESKARFeedbackSystem
from api.ab_testing import ESKARABTestingFramework

logger = logging.getLogger('ESKAR.ProductionAnalytics')

class ESKARProductionDashboard:
    """Professional production analytics dashboard"""

    def __init__(self):
        self.feedback_system = ESKARFeedbackSystem()
        self.ab_testing = ESKARABTestingFramework()

        # Dashboard configuration
        self.refresh_interval = 300  # 5 minutes
        self.default_time_range = 7  # days

    def render_main_dashboard(self):
        """Render the main production analytics dashboard"""
        st.set_page_config(
            page_title="ESKAR Production Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for professional styling
        st.markdown("""
        <style>
        .main - header {
            background: linear - gradient(90deg, #1e3a8a, #3b82f6);
            color: white;
            padding: 2rem;
            border - radius: 10px;
            margin - bottom: 2rem;
            text - align: center;
        }
        .metric - container {
            background: white;
            padding: 1.5rem;
            border - radius: 8px;
            box - shadow: 0 2px 4px rgba(0,0,0,0.1);
            border - left: 4px solid #3b82f6;
        }
        .status - good { border - left - color: #10b981 !important; }
        .status - warning { border - left - color: #f59e0b !important; }
        .status - critical { border - left - color: #ef4444 !important; }
        </style>
        """, unsafe_allow_html=True)

        # Main header
        st.markdown("""
        <div class="main - header">
            <h1>🏠 ESKAR Production Analytics Dashboard</h1>
            <p>Real - time monitoring for ESK Housing Finder Production System</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar configuration
        with st.sidebar:
            st.header("📋 Dashboard Controls")

            time_range = st.selectbox(
                "Time Range",
                options=[1, 7, 30, 90],
                index=1,
                format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
            )

            auto_refresh = st.checkbox("Auto Refresh", value=True)

            if st.button("🔄 Refresh Now"):
                st.rerun()

            st.markdown("---")
            st.markdown("**Quick Actions**")

            if st.button("📥 Export Analytics"):
                self._export_analytics_data(time_range)

            if st.button("🧪 View A / B Tests"):
                st.session_state.show_ab_tests = True

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🤖 ML Performance", "👥 User Analytics",
            "🧪 A / B Testing", "⚡ System Health"
        ])

        with tab1:
            self._render_overview_tab(time_range)

        with tab2:
            self._render_ml_performance_tab(time_range)

        with tab3:
            self._render_user_analytics_tab(time_range)

        with tab4:
            self._render_ab_testing_tab(time_range)

        with tab5:
            self._render_system_health_tab(time_range)

        # Auto - refresh setup
        if auto_refresh:
            st.rerun()

    def _render_overview_tab(self, days_back: int):
        """Render overview metrics tab"""
        st.header("📈 Production Overview")

        # Get analytics data
        analytics = self.feedback_system.get_feedback_analytics(days_back)

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_sessions = sum(
                data.get('total_sessions', 0)
                for data in analytics.get('session_analytics', {}).values()
            )
            st.metric(
                label="👥 Total Sessions",
                value=f"{total_sessions:,}",
                delta=f"+{max(1, total_sessions // 10)}" if total_sessions > 0 else "0"
            )

        with col2:
            avg_satisfaction = analytics['summary'].get('avg_rating_overall', 0)
            satisfaction_status = "🟢" if avg_satisfaction >= 4.0 else "🟡" if avg_satisfaction >= 3.0 else "🔴"
            st.metric(
                label="😊 Avg Satisfaction",
                value=f"{avg_satisfaction:.1f}/5",
                delta=f"{satisfaction_status}"
            )

        with col3:
            total_feedback = analytics['summary'].get('total_feedback_records', 0)
            st.metric(
                label="💬 Feedback Records",
                value=f"{total_feedback:,}",
                delta=f"+{max(1, total_feedback // 5)}" if total_feedback > 0 else "0"
            )

        with col4:
            # Calculate engagement rate
            avg_properties_viewed = np.mean([
                data.get('avg_properties_viewed', 0)
                for data in analytics.get('session_analytics', {}).values()
            ]) if analytics.get('session_analytics') else 0

            st.metric(
                label="🏠 Avg Properties / Session",
                value=f"{avg_properties_viewed:.1f}",
                delta="+0.5" if avg_properties_viewed > 3 else "0"
            )

        # User satisfaction trends
        st.subheader("📊 User Satisfaction Trends")

        # Generate mock time series data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        satisfaction_data = pd.DataFrame({
            'Date': dates,
            'Satisfaction': np.random.normal(4.2, 0.3, days_back).clip(1, 5),
            'Sessions': np.random.poisson(20, days_back),
            'Properties_Viewed': np.random.poisson(60, days_back)
        })

        # Create satisfaction trend chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('User Satisfaction Over Time', 'Session Activity'),
            vertical_spacing=0.1
        )

        # Satisfaction line
        fig.add_trace(
            go.Scatter(
                x=satisfaction_data['Date'],
                y=satisfaction_data['Satisfaction'],
                mode='lines + markers',
                name='Satisfaction',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # Sessions bar chart
        fig.add_trace(
            go.Bar(
                x=satisfaction_data['Date'],
                y=satisfaction_data['Sessions'],
                name='Daily Sessions',
                marker_color='#10b981'
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            title_text="Production Performance Metrics"
        )

        fig.update_yaxes(title_text="Satisfaction (1 - 5)", row=1, col=1)
        fig.update_yaxes(title_text="Sessions", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Geographic distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌍 User Distribution")

            # Mock geographic data
            user_types = ['ESK Families', 'General Users', 'Researchers', 'Staff']
            user_counts = [45, 30, 15, 10]

            fig = px.pie(
                values=user_counts,
                names=user_types,
                title="User Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏘️ Popular Neighborhoods")

            # Mock neighborhood data
            neighborhoods = ['Weststadt', 'Südstadt', 'Innenstadt - West', 'Durlach', 'Oststadt']
            search_counts = [120, 98, 87, 76, 65]

            fig = px.bar(
                x=search_counts,
                y=neighborhoods,
                orientation='h',
                title="Searches by Neighborhood",
                color=search_counts,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def _render_ml_performance_tab(self, days_back: int):
        """Render ML model performance metrics"""
        st.header("🤖 ML Model Performance")

        # Model accuracy metrics
        st.subheader("📈 Model Accuracy Tracking")

        # Generate mock ML performance data
        models = ['RandomForest', 'XGBoost', 'LightGBM', 'Ensemble']
        accuracy_scores = [0.847, 0.863, 0.851, 0.878]
        prediction_counts = [1250, 1180, 1320, 980]

        col1, col2 = st.columns(2)

        with col1:
            # Model accuracy comparison
            fig = px.bar(
                x=models,
                y=accuracy_scores,
                title="Model Accuracy Comparison",
                color=accuracy_scores,
                color_continuous_scale='Viridis',
                text=[f"{acc:.1%}" for acc in accuracy_scores]
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            fig.update_yaxes(title="Accuracy Score", range=[0.8, 0.9])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Prediction volume
            fig = px.pie(
                values=prediction_counts,
                names=models,
                title="Prediction Volume by Model",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance analysis
        st.subheader("🎯 Feature Importance Analysis")

        features = [
            'Distance to ESK', 'Price per SqM', 'Neighborhood Score',
            'Property Type', 'Public Transport', 'Family Amenities',
            'Safety Score', 'School Quality', 'Market Trend'
        ]
        importance_scores = [0.24, 0.19, 0.13, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04]

        fig = px.bar(
            x=importance_scores,
            y=features,
            orientation='h',
            title="Feature Importance in Ensemble Model",
            color=importance_scores,
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Model performance over time
        st.subheader("📊 Performance Trends")

        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Accuracy': np.random.normal(0.87, 0.02, days_back).clip(0.8, 0.95),
            'Predictions': np.random.poisson(50, days_back),
            'User_Satisfaction': np.random.normal(4.1, 0.3, days_back).clip(1, 5)
        })

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Accuracy Over Time', 'Daily Predictions'),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # Accuracy line
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Accuracy'],
                mode='lines + markers',
                name='Model Accuracy',
                line=dict(color='#ef4444', width=3)
            ),
            row=1, col=1
        )

        # User satisfaction on secondary y - axis
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['User_Satisfaction'],
                mode='lines',
                name='User Satisfaction',
                line=dict(color='#10b981', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )

        # Predictions bar
        fig.add_trace(
            go.Bar(
                x=performance_data['Date'],
                y=performance_data['Predictions'],
                name='Daily Predictions',
                marker_color='#3b82f6'
            ),
            row=2, col=1
        )

        fig.update_layout(height=500, showlegend=True)
        fig.update_yaxes(title_text="Model Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="User Satisfaction", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Predictions", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def _render_user_analytics_tab(self, days_back: int):
        """Render user behavior analytics"""
        st.header("👥 User Behavior Analytics")

        # User journey analysis
        st.subheader("🛤️ User Journey Analysis")

        # Mock user journey data
        journey_steps = ['Landing', 'Search', 'Filter', 'View Properties', 'Get Predictions', 'Contact']
        conversion_rates = [1.0, 0.85, 0.72, 0.58, 0.35, 0.12]

        fig = go.Figure()

        fig.add_trace(go.Funnel(
            y=journey_steps,
            x=conversion_rates,
            textinfo="value + percent initial",
            marker_color=['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe']
        ))

        fig.update_layout(
            title="User Conversion Funnel",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # User engagement metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Session Duration")

            # Mock session duration data
            duration_ranges = ['< 2 min', '2 - 5 min', '5 - 10 min', '10 - 20 min', '20+ min']
            session_counts = [25, 45, 68, 42, 20]

            fig = px.bar(
                x=duration_ranges,
                y=session_counts,
                title="Session Duration Distribution",
                color=session_counts,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🔍 Search Patterns")

            # Mock search criteria data
            search_criteria = ['Budget Range', 'Bedrooms', 'Neighborhood', 'Property Type', 'Distance to ESK']
            usage_frequency = [95, 87, 82, 76, 71]

            fig = px.bar(
                x=search_criteria,
                y=usage_frequency,
                title="Most Used Search Criteria (%)",
                color=usage_frequency,
                color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # User feedback analysis
        st.subheader("💬 User Feedback Analysis")

        analytics = self.feedback_system.get_feedback_analytics(days_back)

        if analytics.get('ratings_by_type'):
            feedback_types = list(analytics['ratings_by_type'].keys())
            avg_ratings = [analytics['ratings_by_type'][ft]['avg_rating'] for ft in feedback_types]

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    x=feedback_types,
                    y=avg_ratings,
                    title="Average Ratings by Category",
                    color=avg_ratings,
                    color_continuous_scale='RdYlGn',
                    text=[f"{rating:.1f}" for rating in avg_ratings]
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                fig.update_yaxes(range=[1, 5])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Feedback volume
                total_responses = [analytics['ratings_by_type'][ft]['total_responses'] for ft in feedback_types]

                fig = px.pie(
                    values=total_responses,
                    names=feedback_types,
                    title="Feedback Volume Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_ab_testing_tab(self, days_back: int):
        """Render A / B testing dashboard"""
        st.header("🧪 A / B Testing Dashboard")

        # A / B test creation section
        st.subheader("🆕 Create New A / B Test")

        with st.expander("Create ML Model Comparison Test"):
            test_name = st.text_input("Test Name", "ML Model Performance Test")
            test_description = st.text_area("Description", "Compare different ML models for property recommendation")

            col1, col2 = st.columns(2)
            with col1:
                success_metric = st.selectbox("Success Metric", ["user_rating", "click_through_rate", "conversion_rate"])
            with col2:
                sample_size = st.number_input("Minimum Sample Size", min_value=50, value=100, step=10)

            if st.button("🚀 Create A / B Test"):
                # Create mock A / B test
                model_configs = [
                    {"name": "Current Ensemble", "description": "Existing production model"},
                    {"name": "Enhanced XGBoost", "description": "Improved XGBoost with new features"}
                ]

                experiment_id = self.ab_testing.create_ml_model_experiment(
                    test_name, test_description, model_configs, success_metric, sample_size
                )

                st.success(f"✅ Created A / B test: {experiment_id}")

        # Active experiments
        st.subheader("📊 Active Experiments")

        # Mock active experiments data
        experiments_data = [
            {
                "ID": "ml_exp_20250115_143022_001",
                "Name": "Enhanced Feature Engineering",
                "Type": "ML Model",
                "Status": "Running",
                "Duration": "3 days",
                "Samples": "87 / 100",
                "Significance": "Not yet"
            },
            {
                "ID": "feat_exp_20250114_091045_002",
                "Name": "Map View Enhancement",
                "Type": "Feature",
                "Status": "Running",
                "Duration": "5 days",
                "Samples": "156 / 150",
                "Significance": "p=0.032 ✅"
            }
        ]

        experiments_df = pd.DataFrame(experiments_data)
        st.dataframe(experiments_df, use_container_width=True)

        # Experiment results visualization
        st.subheader("📈 Experiment Results")

        # Mock A / B test results
        variants = ['Control (Current)', 'Variant A (Enhanced)', 'Variant B (Alternative)']
        conversion_rates = [0.234, 0.267, 0.251]
        sample_sizes = [1420, 1380, 1405]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                x=variants,
                y=conversion_rates,
                title="Conversion Rate by Variant",
                color=conversion_rates,
                color_continuous_scale='Viridis',
                text=[f"{rate:.1%}" for rate in conversion_rates]
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                x=variants,
                y=sample_sizes,
                title="Sample Size by Variant",
                color=sample_sizes,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Statistical significance testing
        st.subheader("📊 Statistical Analysis")

        # Mock statistical results
        st.markdown("""
        **Current Test Results:**
        - **Control vs Variant A**: p - value = 0.032 ✅ (Statistically Significant)
        - **Effect Size**: +14.1% improvement in conversion rate
        - **Confidence Level**: 95%
        - **Recommendation**: 🚀 Deploy Variant A to production
        """)

        # Test timeline
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        timeline_data = pd.DataFrame({
            'Date': dates,
            'Control': np.random.normal(0.23, 0.02, 7).clip(0.15, 0.35),
            'Variant_A': np.random.normal(0.27, 0.02, 7).clip(0.15, 0.35),
            'Variant_B': np.random.normal(0.25, 0.02, 7).clip(0.15, 0.35)
        })

        fig = px.line(
            timeline_data,
            x='Date',
            y=['Control', 'Variant_A', 'Variant_B'],
            title="Conversion Rate Timeline",
            labels={'value': 'Conversion Rate', 'variable': 'Variant'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_system_health_tab(self, days_back: int):
        """Render system health monitoring"""
        st.header("⚡ System Health Monitoring")

        # System status overview
        st.subheader("🔍 System Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric - container status - good">
                <h3>🟢 API Status</h3>
                <p><strong>Operational</strong></p>
                <p>99.8% uptime</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric - container status - good">
                <h3>🟢 Database</h3>
                <p><strong>Healthy</strong></p>
                <p>Response: 45ms</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric - container status - warning">
                <h3>🟡 ML Models</h3>
                <p><strong>Warning</strong></p>
                <p>High memory usage</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric - container status - good">
                <h3>🟢 Frontend</h3>
                <p><strong>Operational</strong></p>
                <p>Load time: 1.2s</p>
            </div>
            """, unsafe_allow_html=True)

        # Performance metrics
        st.subheader("📊 Performance Metrics")

        # Mock performance data
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Response_Time': np.random.normal(45, 10, days_back).clip(20, 100),
            'Throughput': np.random.poisson(150, days_back),
            'Error_Rate': np.random.exponential(0.005, days_back).clip(0, 0.05),
            'CPU_Usage': np.random.normal(65, 15, days_back).clip(20, 100),
            'Memory_Usage': np.random.normal(70, 20, days_back).clip(30, 95)
        })

        # Response time and throughput
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                performance_data,
                x='Date',
                y='Response_Time',
                title="API Response Time (ms)",
                line_shape='spline'
            )
            fig.update_traces(line_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                performance_data,
                x='Date',
                y='Throughput',
                title="Daily Throughput (requests)",
                color='Throughput',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # System resource usage
        st.subheader("💾 Resource Usage")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
            vertical_spacing=0.1
        )

        # CPU usage
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['CPU_Usage'],
                mode='lines + markers',
                name='CPU Usage',
                line=dict(color='#ef4444', width=2),
                fill='tonexty'
            ),
            row=1, col=1
        )

        # Memory usage
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Memory_Usage'],
                mode='lines + markers',
                name='Memory Usage',
                line=dict(color='#10b981', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )

        fig.update_layout(height=500, showlegend=True)
        fig.update_yaxes(title_text="CPU %", row=1, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Memory %", row=2, col=1, range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

        # Error monitoring
        st.subheader("🚨 Error Monitoring")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                performance_data,
                x='Date',
                y='Error_Rate',
                title="Error Rate (%)",
                line_shape='spline'
            )
            fig.update_traces(line_color='#ef4444')
            fig.update_yaxes(title="Error Rate (%)", range=[0, 0.05])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Recent errors (mock data)
            error_types = ['Timeout', 'Validation', 'ML Model', 'Database', 'Network']
            error_counts = [3, 7, 2, 1, 4]

            fig = px.pie(
                values=error_counts,
                names=error_types,
                title="Error Distribution (Last 24h)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Alerts and notifications
        st.subheader("🔔 Recent Alerts")

        alerts_data = [
            {"Time": "2025 - 01 - 15 14:32", "Level": "Warning", "Message": "High memory usage detected", "Status": "Investigating"},
            {"Time": "2025 - 01 - 15 12:15", "Level": "Info", "Message": "ML model retrained successfully", "Status": "Resolved"},
            {"Time": "2025 - 01 - 15 09:45", "Level": "Critical", "Message": "Database connection timeout", "Status": "Resolved"}
        ]

        alerts_df = pd.DataFrame(alerts_data)
        st.dataframe(alerts_df, use_container_width=True)

    def _export_analytics_data(self, days_back: int):
        """Export analytics data for external analysis"""
        try:
            # Export feedback data
            feedback_path = self.feedback_system.export_feedback_data()

            st.success(f"✅ Analytics data exported to {feedback_path}")

            # Offer download
            with open(feedback_path, 'rb') as file:
                st.download_button(
                    label="📥 Download Analytics Report",
                    data=file.read(),
                    file_name=f"eskar_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text / csv"
                )

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

def main():
    """Main dashboard application"""
    dashboard = ESKARProductionDashboard()
    dashboard.render_main_dashboard()

if __name__ == "__main__":
    main()
