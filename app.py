"""
ESKAR Housing Finder - Production ML Application
European School Karlsruhe Housing Finder with Advanced Machine Learning

Author: Friedrich-Wilhelm Möller
Purpose: Code Institute Portfolio Project 5 (Advanced Full-Stack Development)
Target: ESK families seeking housing in Karlsruhe, Germany

Features:
- Advanced ML ensemble with XGBoost, LightGBM, RandomForest
- 50+ engineered features for ESK-specific recommendations
- Real-time property scoring and market analysis
- User feedback integration and A/B testing capability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import sys
import os
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Add src directory to path for imports
sys.path.append('src')

# Import production modules
try:
    from config import ESKARConfig
    from features.feature_engineering import ESKARFeatureEngineer
    from models.ml_ensemble import ESKARMLEnsemble
    from api.user_feedback import ESKARFeedbackSystem
    from api.hybrid_real_estate_api import ESKARHybridRealEstateAPI
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("💡 Make sure all production modules are available in src/ directory")

# Import data generator (with fallback)
try:
    from data_generator import ESKARDataGenerator
except ImportError:
    try:
        from eskar_data_generator import ESKARDataGenerator
    except ImportError:
        st.error("❌ Data generator not found!")
        st.stop()

# Page Configuration
st.set_page_config(
    page_title="ESKAR Housing Finder",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .esk-highlight {
        background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize production systems
@st.cache_resource
def initialize_production_systems():
    """Initialize all production ML and analytics systems"""
    config = None
    feedback_system = None
    real_estate_api = None
    
    # Try to initialize each system individually
    try:
        config = ESKARConfig()
    except Exception as e:
        st.warning(f"⚠️ Config system not available: {e}")
    
    try:
        feedback_system = ESKARFeedbackSystem()
        # Start user session for analytics
        if 'session_id' not in st.session_state and feedback_system:
            st.session_state.session_id = feedback_system.start_user_session('esk_family')
    except Exception as e:
        st.warning(f"⚠️ Feedback system not available: {e}")
        # Create a simple fallback feedback system
        feedback_system = create_fallback_feedback_system()
    
    try:
        real_estate_api = ESKARHybridRealEstateAPI()
    except Exception as e:
        st.warning(f"⚠️ Hybrid real estate API not available: {e}")
        # Fallback to original API if available
        try:
            from api.real_estate_api import ESKARRealEstateAPI
            real_estate_api = ESKARRealEstateAPI()
        except:
            real_estate_api = None
    
    return config, feedback_system, real_estate_api

def create_fallback_feedback_system():
    """Create a simple fallback feedback system for development"""
    import random
    class FallbackFeedback:
        def submit_feedback(self, user_id, rating, feedback_text, feature_used="general"):
            st.success("✅ Feedback erfasst! (Entwicklungsmodus)")
            st.info(f"Rating: {rating}/5, Text: {feedback_text}")
            return True
            
        def start_user_session(self, user_type):
            return f"dev_session_{random.randint(1000, 9999)}"
    
    return FallbackFeedback()

# Initialize systems
config, feedback_system, real_estate_api = initialize_production_systems()

# ESK Location and Key Employers (PRECISE: Albert-Schweitzer-Str. 1, 76139 Karlsruhe)
ESK_LOCATION = {"lat": 49.04637, "lon": 8.44805, "name": "European School Karlsruhe"}

# Major Employers with precise coordinates (updated from feedback)
MAJOR_EMPLOYERS = {
    'SAP Walldorf': {"lat": 49.2933, "lon": 8.6428, "color": "blue"},
    'SAP Karlsruhe': {"lat": 49.0233, "lon": 8.4103, "color": "blue"},
    'Ionos Karlsruhe': {"lat": 49.0089, "lon": 8.3858, "color": "blue"},
    'KIT Campus South': {"lat": 49.0069, "lon": 8.4037, "color": "blue"},
    'KIT Campus North': {"lat": 49.0943, "lon": 8.4347, "color": "blue"},
    'Research Center': {"lat": 49.0930, "lon": 8.4279, "color": "blue"},
    'EnBW Karlsruhe': {"lat": 49.006450040902145, "lon": 8.437177202431728, "color": "blue"},
    'dm Karlsruhe': {"lat": 49.00299770848193, "lon": 8.456215912548018, "color": "blue"}
}

# Major Reference Points for orientation
MAJOR_POINTS = {
    'Klinikum Karlsruhe': {"lat": 49.018946188108764, "lon": 8.371897980517069, "color": "gray"},
    'Hauptbahnhof Karlsruhe': {"lat": 48.99535399579631, "lon": 8.400132211538523, "color": "gray"},
    'Messe Karlsruhe': {"lat": 48.98051198180659, "lon": 8.32680592186435, "color": "gray"},
    'Bahnhof Ettlingen': {"lat": 48.93958851101495, "lon": 8.40940991628156, "color": "gray"}
}

@st.cache_data
def calculate_esk_suitability_score(df):
    """Calculate ESK suitability score based on distance and features"""
    import numpy as np
    
    # Base score from distance (closer = higher score)
    # Max distance in dataset, score inversely proportional
    max_distance = df['distance_to_esk'].max()
    distance_score = (max_distance - df['distance_to_esk']) / max_distance * 100
    
    # Bonus points for family-friendly features
    feature_bonus = 0
    if 'garden' in df.columns:
        feature_bonus += df['garden'] * 10
    if 'balcony' in df.columns:
        feature_bonus += df['balcony'] * 5
    if 'garage' in df.columns:
        feature_bonus += df['garage'] * 5
    
    # Bonus for optimal bedroom count for families (3-4 bedrooms)
    bedroom_bonus = np.where(
        (df['bedrooms'] >= 3) & (df['bedrooms'] <= 4), 10, 0
    )
    
    # Final score (0-100 scale)
    total_score = distance_score + feature_bonus + bedroom_bonus
    return np.clip(total_score, 0, 100)

@st.cache_data
def add_missing_columns(df):
    """Add missing columns expected by the UI"""
    import numpy as np
    
    # Add safety score based on neighborhood safety
    neighborhood_safety = {
        'Weststadt': 8.5, 'Südstadt': 8.2, 'Innenstadt-West': 7.8,
        'Durlach': 8.7, 'Oststadt': 8.4, 'Mühlburg': 8.1
    }
    df['safety_score'] = df['neighborhood'].map(neighborhood_safety).fillna(8.0)
    
    # Add current ESK families count (simulated)
    np.random.seed(42)  # For consistent results
    df['current_esk_families'] = np.random.randint(0, 8, len(df))
    
    return df

@st.cache_data
def load_housing_data():
    """Load ESKAR housing data with enhanced ML features"""
    try:
        # Use production real estate API if available
        if real_estate_api:
            properties = real_estate_api.search_properties_karlsruhe({'max_results': 200})
            df = real_estate_api.export_properties_to_dataframe(properties)
            
            # Add missing columns that the app expects
            df['garden'] = df['features'].str.contains('garden', na=False)
            df['balcony'] = df['features'].str.contains('balcony', na=False)
            df['garage'] = df['features'].str.contains('garage', na=False)
            
            # Calculate ESK suitability score based on distance and features
            df['esk_suitability_score'] = calculate_esk_suitability_score(df)
            
            # Add other missing columns expected by the UI
            df = add_missing_columns(df)
            
            st.success(f"✅ Loaded {len(df)} properties from production API")
            return df
    except Exception as e:
        st.warning(f"⚠️ Production API unavailable: {e}")
    
    try:
        # Fallback to data generator
        generator = ESKARDataGenerator()
        df = generator.generate_dataset(200)
        st.info(f"📊 Generated {len(df)} synthetic ESK-optimized properties")
        return df
    except Exception as e:
        st.error(f"❌ Data generation failed: {e}")
        # Return minimal demo data
        return pd.DataFrame({
            'neighborhood': ['Weststadt', 'Südstadt'] * 10,
            'property_type': ['apartment', 'house'] * 10,
            'sqft': np.random.randint(60, 200, 20),
            'bedrooms': np.random.randint(2, 5, 20),
            'price': np.random.randint(300000, 800000, 20),
            'lat': [49.004 + np.random.uniform(-0.02, 0.02) for _ in range(20)],
            'lon': [8.385 + np.random.uniform(-0.02, 0.02) for _ in range(20)]
        })

@st.cache_data  
def get_enhanced_ml_predictions(df, target_features):
    """Get ML predictions using production ensemble if available"""
    try:
        if config:
            # Use production ML ensemble
            feature_engineer = ESKARFeatureEngineer(config)
            ml_ensemble = ESKARMLEnsemble(config)
            
            # Engineer features
            df_features = feature_engineer.engineer_features(df)
            
            # Train ensemble on current data
            y = df['price'] if 'price' in df.columns else df.index
            trained_models = ml_ensemble.train_ensemble(df_features, y)
            
            # Make predictions for filtered data
            predictions = ml_ensemble.predict(df_features)
            
            return predictions, trained_models, df_features
    except Exception as e:
        st.warning(f"⚠️ Advanced ML unavailable, using basic model: {e}")
    
    # Fallback to simple model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # Simple feature preparation
    X = df[['sqft', 'bedrooms']].fillna(df[['sqft', 'bedrooms']].mean())
    y = df['price'] if 'price' in df.columns else np.random.randint(300000, 800000, len(df))
    
    # Train simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Return predictions for all data
    predictions = model.predict(X)
    accuracy = r2_score(y_test, model.predict(X_test))
    
    return predictions, {'simple_rf': {'accuracy': accuracy}}, X
    try:
        # Try to load real ESK data
        df = pd.read_csv('data/eskar_housing_data.csv')
        return df
    except FileNotFoundError:
        # Generate fresh ESK data using our generator
        st.info("🏫 Generating fresh ESK housing data...")
        generator = ESKARDataGenerator()
        df = generator.generate_housing_dataset(n_samples=300)
        generator.save_dataset(df)
        return df

def show_welcome_page():
    """Display welcome page with ESK information"""
    st.markdown('<div class="main-header"><h1>🏫 ESKAR Housing Finder</h1><p>AI-powered housing search for European School Karlsruhe families</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Our Mission</h3>
        <p>Help ESK families find their perfect home in Karlsruhe with ML-powered property recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🏫 For ESK Community</h3>
        <p>Optimized for international families working at SAP, KIT, Ionos, and other major Karlsruhe employers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🤖 ML-Powered</h3>
        <p>Advanced algorithms consider school distance, community fit, and family needs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ESK Quick Facts
    st.markdown("""
    <div class="esk-highlight">
    <h3>🏫 European School Karlsruhe Quick Facts</h3>
    <ul>
    <li><strong>Students:</strong> 500+ international families</li>
    <li><strong>Languages:</strong> German, French, English</li>
    <li><strong>Grades:</strong> Kindergarten through European Baccalaureate</li>
    <li><strong>Community:</strong> 45+ nationalities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_search_filters():
    """Display search filters in sidebar"""
    st.header("🎯 Search Filters")
    
    # Load data for filter ranges
    df = load_housing_data()
    
    # Store filters in session state
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {}
    
    # Price range
    st.session_state.search_filters['price_range'] = st.slider(
        "� Price Range (€)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max())),
        step=10000,
        format="%d€"
    )
    
    # ESK distance
    st.session_state.search_filters['max_distance'] = st.slider(
        "🏫 Max Distance to ESK (km)",
        min_value=0.5,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    # Bedrooms
    st.session_state.search_filters['bedrooms'] = st.multiselect(
        "🛏️ Bedrooms",
        options=sorted(df['bedrooms'].unique()),
        default=sorted(df['bedrooms'].unique())
    )
    
    # Property type
    st.session_state.search_filters['property_types'] = st.multiselect(
        "🏠 Property Type",
        options=['house', 'apartment'],
        default=['house', 'apartment']
    )
    
    # Neighborhoods
    st.session_state.search_filters['neighborhoods'] = st.multiselect(
        "🗺️ Neighborhoods",
        options=sorted(df['neighborhood'].unique()),
        default=sorted(df['neighborhood'].unique())
    )
    
    # ESK Score threshold
    st.session_state.search_filters['min_esk_score'] = st.slider(
        "⭐ Minimum ESK Score",
        min_value=1.0,
        max_value=10.0,
        value=6.0,
        step=0.1
    )

def show_map_filters():
    """Display map filters in sidebar"""
    st.subheader("🎯 Map Filters")
    
    # Store filters in session state
    if 'map_filters' not in st.session_state:
        st.session_state.map_filters = {}
    
    st.session_state.map_filters['max_distance'] = st.slider(
        "Max Distance to ESK (km)", 
        0.5, 15.0, 8.0, 0.5
    )
    st.session_state.map_filters['min_score'] = st.slider(
        "Min ESK Suitability Score", 
        20, 100, 60, 5
    )
    st.session_state.map_filters['max_price'] = st.slider(
        "Max Price (€)", 
        200000, 2000000, 800000, 50000
    )

def show_property_search():
    """Display property search with ESK-optimized filters"""
    st.title("🔍 Property Search")
    st.markdown("### Find your perfect home with ESK-optimized filters")
    
    # Load data
    df = load_housing_data()
    
    # Get filters from session state (set by sidebar)
    filters = st.session_state.get('search_filters', {})
    
    # Use default values if filters not set
    price_range = filters.get('price_range', (int(df['price'].min()), int(df['price'].max())))
    max_distance = filters.get('max_distance', 5.0)
    bedrooms = filters.get('bedrooms', sorted(df['bedrooms'].unique()))
    property_types = filters.get('property_types', ['house', 'apartment'])
    neighborhoods = filters.get('neighborhoods', sorted(df['neighborhood'].unique()))
    min_esk_score = filters.get('min_esk_score', 6.0)
    
    # Filter data
    filtered_df = df[
        (df['price'].between(price_range[0], price_range[1])) &
        (df['distance_to_esk'] <= max_distance) &
        (df['bedrooms'].isin(bedrooms)) &
        (df['property_type'].isin(property_types)) &
        (df['neighborhood'].isin(neighborhoods)) &
        (df['esk_suitability_score'] >= min_esk_score)
    ]
    
    # Results
    st.subheader(f"🎯 {len(filtered_df)} Properties Found")
    
    if len(filtered_df) == 0:
        st.warning("No properties match your criteria. Try adjusting the filters.")
        return
    
    # Top recommendations
    top_properties = filtered_df.nlargest(3, 'esk_suitability_score')
    
    st.subheader("🌟 Top ESK Recommendations")
    
    for idx, prop in top_properties.iterrows():
        with st.expander(f"🏠 {prop['neighborhood']} - ESK Score: {prop['esk_suitability_score']}/10"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Price", f"€{prop['price']:,}")
                st.metric("🛏️ Bedrooms", prop['bedrooms'])
                st.metric("📐 Size", f"{prop['sqft']} m²")
                
            with col2:
                st.metric("🏫 Distance to ESK", f"{prop['distance_to_esk']} km")
                st.metric("🏠 Type", prop['property_type'].title())
                st.metric("🌳 Garden", "Yes" if prop['garden'] else "No")
                
            with col3:
                st.metric("⭐ ESK Score", f"{prop['esk_suitability_score']}/10")
                st.metric("🔒 Safety", f"{prop['safety_score']}/10")
                st.metric("👨‍👩‍👧‍👦 ESK Families", prop['current_esk_families'])
    
    # Full results table
    st.subheader("📊 All Results")
    display_columns = [
        'neighborhood', 'property_type', 'price', 'bedrooms', 'sqft',
        'distance_to_esk', 'esk_suitability_score'
    ]
    
    st.dataframe(
        filtered_df[display_columns].sort_values('esk_suitability_score', ascending=False),
        use_container_width=True
    )

def show_ml_predictions():
    """Show ML price prediction interface"""
    st.title("🤖 AI Price Prediction")
    st.markdown("### Get instant property value estimates using machine learning")
    
    # Load data
    df = load_housing_data()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏠 Property Details")
        
        # Input features
        neighborhood = st.selectbox("Neighborhood", df['neighborhood'].unique())
        property_type = st.selectbox("Property Type", ['house', 'apartment'])
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
        sqft = st.number_input("Size (m²)", min_value=30, max_value=300, value=100)
        distance_esk = st.number_input("Distance to ESK (km)", min_value=0.1, max_value=15.0, value=3.0)
        garden = st.checkbox("Garden/Balcony")
        
        predict_button = st.button("🔮 Predict Price", type="primary")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if predict_button:
            # Prepare features
            features = prepare_ml_features(
                neighborhood, property_type, bedrooms, sqft, distance_esk, garden, df
            )
            
            # Train model and predict
            model, accuracy = train_price_model(df)
            predicted_price = model.predict([features])[0]
            
            # Display results
            st.metric("💰 Predicted Price", f"€{predicted_price:,.0f}")
            st.metric("📊 Model Accuracy", f"{accuracy:.1%}")
            
            # Price breakdown
            price_per_sqm = predicted_price / sqft
            st.metric("💲 Price per m²", f"€{price_per_sqm:.0f}")
            
            # Confidence interval
            margin = predicted_price * 0.15
            st.write(f"**Price Range:** €{predicted_price-margin:,.0f} - €{predicted_price+margin:,.0f}")

def prepare_ml_features(neighborhood, property_type, bedrooms, sqft, distance_esk, garden, df):
    """Prepare features for ML model"""
    # Get neighborhood average price as feature
    neighborhood_avg = df[df['neighborhood'] == neighborhood]['price_per_sqm'].mean()
    
    # Convert categorical variables
    property_type_num = 1 if property_type == 'house' else 0
    garden_num = 1 if garden else 0
    
    return [bedrooms, sqft, distance_esk, property_type_num, garden_num, neighborhood_avg]

@st.cache_resource
def train_price_model(df):
    """Train price prediction model"""
    # Prepare features
    features = []
    targets = []
    
    for _, row in df.iterrows():
        neighborhood_avg = df[df['neighborhood'] == row['neighborhood']]['price_per_sqm'].mean()
        property_type_num = 1 if row['property_type'] == 'house' else 0
        
        feature_row = [
            row['bedrooms'],
            row['sqft'],
            row['distance_to_esk'],
            property_type_num,
            row['garden'],
            neighborhood_avg
        ]
        
        features.append(feature_row)
        targets.append(row['price'])
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = r2_score(y_test, model.predict(X_test))
    
    return model, accuracy

def show_market_analytics():
    """Display market analytics and insights"""
    st.title("📊 Market Analytics")
    st.markdown("### Karlsruhe housing market insights for ESK families")
    
    df = load_housing_data()
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = df['price'].mean()
        st.metric("💰 Average Price", f"€{avg_price:,.0f}")
    
    with col2:
        avg_esk_score = df['esk_suitability_score'].mean()
        st.metric("⭐ Avg ESK Score", f"{avg_esk_score:.1f}/10")
    
    with col3:
        properties_near_esk = len(df[df['distance_to_esk'] <= 3])
        st.metric("🏫 Near ESK (<3km)", properties_near_esk)
    
    with col4:
        family_suitable = len(df[df['bedrooms'] >= 3])
        st.metric("👨‍👩‍👧‍👦 Family Suitable", family_suitable)
    
    # Neighborhood comparison
    st.subheader("🗺️ Neighborhood Comparison")
    
    neighborhood_stats = df.groupby('neighborhood').agg({
        'price': 'mean',
        'esk_suitability_score': 'mean',
        'distance_to_esk': 'mean',
        'current_esk_families': 'first'
    }).round(1)
    
    fig = px.scatter(
        neighborhood_stats.reset_index(),
        x='distance_to_esk',
        y='price',
        size='current_esk_families',
        color='esk_suitability_score',
        hover_name='neighborhood',
        title="Neighborhood Overview: Distance vs Price vs ESK Score",
        labels={
            'distance_to_esk': 'Distance to ESK (km)',
            'price': 'Average Price (€)',
            'esk_suitability_score': 'ESK Score'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution
    st.subheader("💰 Price Distribution by Property Type")
    
    fig2 = px.box(
        df,
        x='property_type',
        y='price',
        color='neighborhood',
        title="Price Distribution by Property Type and Neighborhood"
    )
    st.plotly_chart(fig2, use_container_width=True)

def show_interactive_map():
    """Display interactive map with ESK properties and reference locations"""
    st.title("🗺️ Interactive Map")
    st.markdown("### Explore properties with key ESK reference locations")
    
    # Load data
    df = load_housing_data()
    
    # Get filters from session state (set by sidebar)
    filters = st.session_state.get('map_filters', {})
    
    # Use default values if filters not set
    max_distance = filters.get('max_distance', 8.0)
    min_score = filters.get('min_score', 60)
    max_price = filters.get('max_price', 800000)
    
    # Filter data for map
    map_df = df[
        (df['distance_to_esk'] <= max_distance) &
        (df['esk_suitability_score'] >= min_score) &
        (df['price'] <= max_price)
    ]
    
    if len(map_df) == 0:
        st.warning("No properties match your filter criteria. Please adjust the filters.")
        return
    
    # Create map centered between ESK and average property location
    center_lat = (ESK_LOCATION['lat'] + map_df['lat'].mean()) / 2
    center_lon = (ESK_LOCATION['lon'] + map_df['lon'].mean()) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add ESK marker (main reference)
    folium.Marker(
        [ESK_LOCATION['lat'], ESK_LOCATION['lon']],
        popup="""<b>🏫 European School Karlsruhe</b><br>
                Albert-Schweitzer-Str. 1<br>
                76139 Karlsruhe<br>
                <em>Your children's school!</em>""",
        icon=folium.Icon(color='red', icon='graduation-cap', prefix='fa')
    ).add_to(m)
    
    # Add major employers
    for employer, data in MAJOR_EMPLOYERS.items():
        folium.Marker(
            [data['lat'], data['lon']],
            popup=f"<b>💼 {employer}</b><br><em>Major employer in Karlsruhe region</em>",
            icon=folium.Icon(color=data['color'], icon='briefcase', prefix='fa')
        ).add_to(m)
    
    # Add reference points for orientation
    for point, data in MAJOR_POINTS.items():
        folium.Marker(
            [data['lat'], data['lon']],
            popup=f"<b>📍 {point}</b><br><em>Important reference point</em>",
            icon=folium.Icon(color=data['color'], icon='map-marker', prefix='fa')
        ).add_to(m)
    
    # Add property markers with color coding based on ESK score
    for idx, row in map_df.iterrows():
        # Color based on ESK suitability score
        if row['esk_suitability_score'] >= 80:
            color = 'green'
            score_category = 'Excellent'
        elif row['esk_suitability_score'] >= 70:
            color = 'orange'
            score_category = 'Good'
        elif row['esk_suitability_score'] >= 60:
            color = 'blue'
            score_category = 'Fair'
        else:
            color = 'gray'
            score_category = 'Basic'
            
        # Create detailed popup
        popup_html = f"""
        <div style="width: 250px;">
            <h4>🏠 {row['neighborhood']}</h4>
            <hr>
            <p><b>💰 Price:</b> €{row['price']:,}</p>
            <p><b>🛏️ Bedrooms:</b> {row['bedrooms']}</p>
            <p><b>📐 Area:</b> {row.get('area_sqm', row.get('sqft', 'N/A'))} m²</p>
            <p><b>🏫 Distance to ESK:</b> {row['distance_to_esk']:.1f} km</p>
            <p><b>⭐ ESK Score:</b> {row['esk_suitability_score']:.0f}/100 ({score_category})</p>
            <p><b>🏠 Type:</b> {row['property_type'].title()}</p>
            {f"<p><b>🌳 Garden:</b> {'Yes' if row.get('garden', False) else 'No'}</p>" if 'garden' in row else ""}
        </div>
        """
        
        folium.Marker(
            [row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon='home', prefix='fa')
        ).add_to(m)
    
    # Display map
    st_folium(m, width=800, height=600)
    
    # Map legend and statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🗺️ Map Legend:**
        - 🔴 **European School Karlsruhe** - Main reference point
        - � **Major Employers** - SAP, KIT, EnBW, dm, Ionos, Research Centers
        - ⚫ **Reference Points** - Hospital, Train Stations, Trade Fair
        - 🟢 **Excellent Properties** (ESK Score ≥ 80)
        - 🟠 **Good Properties** (ESK Score ≥ 70)  
        - 🔵 **Fair Properties** (ESK Score ≥ 60)
        - ⚫ **Basic Properties** (ESK Score < 60)
        """)
    
    with col2:
        st.markdown("**📊 Map Statistics:**")
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("🏠 Properties Shown", len(map_df))
            st.metric("⭐ Average ESK Score", f"{map_df['esk_suitability_score'].mean():.1f}/100")
        with col2b:
            st.metric("💰 Average Price", f"€{map_df['price'].mean():,.0f}")
            st.metric("🏫 Avg Distance to ESK", f"{map_df['distance_to_esk'].mean():.1f} km")
    
    # Highlight best properties
    st.subheader("🌟 Top Properties on Map")
    top_map_properties = map_df.nlargest(5, 'esk_suitability_score')[
        ['neighborhood', 'property_type', 'price', 'bedrooms', 'distance_to_esk', 'esk_suitability_score']
    ]
    st.dataframe(top_map_properties, use_container_width=True)

def main():
    """Main application function"""
    # Sidebar navigation
    with st.sidebar:
        st.title("🏫 ESKAR Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select Page",
            ["🏠 Welcome", "🔍 Property Search", "🗺️ Interactive Map", "🤖 AI Predictions", "📊 Market Analytics"]
        )
        
        st.markdown("---")
        
        # Show relevant filters for each page immediately after page selection
        if page == "🔍 Property Search":
            show_search_filters()
        elif page == "🗺️ Interactive Map":
            show_map_filters()
        
        # Move About section to bottom of sidebar
        st.markdown("---")
        st.markdown("### 🎯 About ESKAR")
        st.markdown("AI-powered housing finder for European School Karlsruhe families")
        
        st.markdown("**Key Features:**")
        st.markdown("• 🏫 ESK-optimized search")
        st.markdown("• 🤖 ML price predictions")  
        st.markdown("• 📊 Market insights")
        st.markdown("• 🗺️ Karlsruhe expertise")
    
    # Route to selected page with enhanced features
    if page == "🏠 Welcome":
        show_welcome_page()
    elif page == "🔍 Property Search":
        show_property_search()
        # Track search activity
        if feedback_system and 'session_id' in st.session_state:
            feedback_system.update_session_activity(st.session_state.session_id, 'search')
    elif page == "🗺️ Interactive Map":
        show_interactive_map()
        # Track map activity
        if feedback_system and 'session_id' in st.session_state:
            feedback_system.update_session_activity(st.session_state.session_id, 'view_map')
    elif page == "🤖 AI Predictions":
        show_ml_predictions()
        # Track prediction requests
        if feedback_system and 'session_id' in st.session_state:
            feedback_system.update_session_activity(st.session_state.session_id, 'request_prediction')
    elif page == "📊 Market Analytics":
        show_market_analytics()
    
    # Add feedback page
    if st.sidebar.button("💬 Give Feedback"):
        show_feedback_section()
    
    # Footer with production info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🏫 **ESKAR Housing Finder**")
        st.markdown("Built for European School Karlsruhe")
    with col2:
        st.markdown("🤖 **Production Features**")
        st.markdown("Advanced ML • Analytics • A/B Testing")
    with col3:
        st.markdown("📊 **Live Dashboard**")
        st.markdown("[Production Analytics](http://localhost:8502)")

def show_feedback_section():
    """Quick feedback collection - now with improved UX"""
    st.subheader("💬 Quick Feedback")
    
    # Always show feedback form, regardless of system status
    satisfaction = st.radio("How satisfied are you with ESKAR?", [1,2,3,4,5], index=3, 
                           help="1 = Very dissatisfied, 5 = Very satisfied")
    comments = st.text_input("Any suggestions or comments?", 
                            placeholder="Tell us what you think...")
    
    if st.button("Submit Feedback", type="primary"):
        # Try production system first, then fallback
        feedback_submitted = False
        
        if feedback_system:
            try:
                if 'session_id' in st.session_state:
                    feedback_system.collect_search_feedback(
                        st.session_state.session_id, satisfaction, {}, 0, comments
                    )
                    feedback_submitted = True
                    st.success("✅ Thank you for your feedback! (Production mode)")
            except Exception as e:
                st.warning(f"Production feedback failed: {e}")
        
        # Fallback for development/demo mode
        if not feedback_submitted:
            st.success("✅ Thank you for your feedback!")
            st.info(f"📊 Your rating: {satisfaction}/5")
            if comments:
                st.info(f"💬 Your comment: \"{comments}\"")
            st.balloons()  # Fun visual feedback
            
            # Log for development purposes
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"🕒 Feedback submitted at: {timestamp}")
    
    # Show some encouragement
    if st.button("🎯 Quick Survey", help="Optional 30-second survey"):
        st.write("**What brought you to ESKAR today?**")
        purpose = st.selectbox("Select one:", [
            "Looking for family housing near ESK",
            "Researching Karlsruhe neighborhoods", 
            "Comparing property prices",
            "Just exploring the app",
            "Other"
        ])
        if st.button("Submit Survey"):
            st.success(f"✅ Survey submitted: {purpose}")
            st.info("Thank you for helping us improve ESKAR! 🚀")

if __name__ == "__main__":
    main()
