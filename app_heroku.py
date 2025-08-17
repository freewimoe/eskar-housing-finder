# ESKAR Housing Finder - Heroku Production (Lightweight)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import lightgbm as lgb
from datetime import datetime, timedelta
import requests
import json
import logging
import sys
import os

# Page configuration
st.set_page_config(
    page_title="ESKAR Housing Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "ESKAR Housing Finder - Machine Learning powered housing recommendations for European School Karlsruhe families"
    }
)

# Configure logging for Heroku
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('ESKAR-Heroku')

# Simplified configuration for Heroku
class HerokuConfig:
    """Lightweight configuration for Heroku deployment"""
    ESK_COORDINATES = (49.04642435194822, 8.44610144968972)
    ESK_ADDRESS = "Albert-Schweitzer-Str. 1, 76139 Karlsruhe"
    
    NEIGHBORHOODS = [
        "Weststadt", "Südstadt", "Innenstadt-West", "Durlach", 
        "Oststadt", "Mühlburg", "Nordstadt", "Südweststadt",
        "Oberreut", "Knielingen", "Wolfartsweier", "Stupferich"
    ]
    
    EMPLOYERS = [
        {"name": "SAP SE", "coords": (49.293, 8.642), "type": "Technology"},
        {"name": "KIT Campus", "coords": (49.013, 8.404), "type": "Research"},
        {"name": "Ionos SE", "coords": (49.009, 8.424), "type": "Technology"},
    ]

# Lightweight data generator for Heroku
@st.cache_data(ttl=3600)
def generate_heroku_data(num_properties=100):
    """Generate lightweight dataset for Heroku deployment"""
    logger.info(f"Generating {num_properties} properties for Heroku")
    
    np.random.seed(42)
    config = HerokuConfig()
    
    data = []
    for i in range(num_properties):
        neighborhood = np.random.choice(config.NEIGHBORHOODS)
        property_type = np.random.choice(['house', 'apartment'], p=[0.4, 0.6])
        bedrooms = np.random.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1])
        
        # Generate realistic coordinates around Karlsruhe
        lat_offset = np.random.normal(0, 0.02)
        lon_offset = np.random.normal(0, 0.02)
        latitude = config.ESK_COORDINATES[0] + lat_offset
        longitude = config.ESK_COORDINATES[1] + lon_offset
        
        # Calculate distance to ESK
        distance_to_esk = np.sqrt(
            (latitude - config.ESK_COORDINATES[0])**2 + 
            (longitude - config.ESK_COORDINATES[1])**2
        ) * 111  # Approximate km conversion
        
        # Base price calculation
        base_price = 300000 + (bedrooms - 2) * 100000
        if property_type == 'house':
            base_price *= 1.3
        
        # Location-based price adjustments
        location_multiplier = max(0.8, 1.4 - distance_to_esk * 0.1)
        price = int(base_price * location_multiplier * np.random.uniform(0.85, 1.15))
        
        # Calculate ESK suitability score
        proximity_score = max(0, 100 - distance_to_esk * 15)
        feature_score = (bedrooms - 2) * 10
        esk_score = min(100, proximity_score + feature_score + np.random.randint(-10, 10))
        
        data.append({
            'property_id': f'ESK-{i+1:03d}',
            'neighborhood': neighborhood,
            'property_type': property_type,
            'bedrooms': bedrooms,
            'sqft': int(np.random.uniform(60, 250)),
            'garden': np.random.choice([True, False], p=[0.6, 0.4]),
            'garage': np.random.choice([True, False], p=[0.5, 0.5]),
            'price': price,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'distance_to_esk': round(distance_to_esk, 2),
            'esk_suitability_score': int(esk_score),
            'safety_score': round(np.random.uniform(7.0, 9.5), 1),
            'current_esk_families': np.random.randint(0, 5)
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated dataset with {len(df)} properties")
    return df

# Simplified ML model for Heroku
@st.cache_resource
def load_heroku_model():
    """Load lightweight ML model for Heroku"""
    logger.info("Loading lightweight model for Heroku")
    
    # Generate training data
    train_data = generate_heroku_data(200)
    
    # Prepare features
    X = train_data[['bedrooms', 'sqft', 'distance_to_esk', 'esk_suitability_score']]
    X['property_type_house'] = (train_data['property_type'] == 'house').astype(int)
    X['has_garden'] = train_data['garden'].astype(int)
    X['has_garage'] = train_data['garage'].astype(int)
    
    y = train_data['price']
    
    # Train LightGBM model (lightweight)
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    logger.info("Model training completed")
    return model

# Main Streamlit app
def main():
    """Main Heroku application"""
    
    # Header
    st.title("🏠 ESKAR Housing Finder")
    st.subheader("Machine Learning Housing Recommendations for ESK Families")
    st.caption("**Heroku Production Version** - Optimized for performance")
    
    # Load data and model
    with st.spinner("Loading ESKAR system..."):
        df = generate_heroku_data(100)
        model = load_heroku_model()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Welcome", "🔍 Property Search", "🤖 Price Predictions", "📊 Quick Analytics"]
        )
        
        st.markdown("---")
        st.info("**Heroku Version**\nOptimized for fast deployment")
    
    # Page routing
    if page == "🏠 Welcome":
        show_welcome(df)
    elif page == "🔍 Property Search":
        show_property_search(df)
    elif page == "🤖 Price Predictions":
        show_predictions(model)
    elif page == "📊 Quick Analytics":
        show_analytics(df)

def show_welcome(df):
    """Welcome page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to ESKAR Housing Finder! 🎯
        
        **Your AI-powered assistant for finding the perfect home near European School Karlsruhe.**
        
        #### 🎯 What ESKAR offers:
        - **Smart Property Search** with ESK-optimized filters
        - **AI Price Predictions** powered by machine learning
        - **Interactive Maps** showing ESK proximity
        - **Market Analytics** tailored for international families
        
        #### 🏫 About European School Karlsruhe:
        Located at Albert-Schweitzer-Str. 1, ESK serves international families
        working at major employers like SAP, KIT, and Ionos.
        """)
    
    with col2:
        st.metric("Properties Available", len(df))
        st.metric("Average ESK Distance", f"{df['distance_to_esk'].mean():.1f} km")
        st.metric("Neighborhoods Covered", df['neighborhood'].nunique())

def show_property_search(df):
    """Property search page"""
    st.header("🔍 Property Search")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_distance = st.slider("Max Distance to ESK (km)", 0.5, 8.0, 5.0, 0.5)
        bedrooms = st.multiselect("Bedrooms", [2, 3, 4, 5], default=[3, 4])
    
    with col2:
        price_range = st.slider("Price Range (€)", 
                               int(df['price'].min()), 
                               int(df['price'].max()), 
                               (400000, 800000), step=50000)
        property_types = st.multiselect("Property Type", 
                                      ['house', 'apartment'], 
                                      default=['house', 'apartment'])
    
    with col3:
        min_esk_score = st.slider("Min ESK Suitability Score", 0, 100, 50)
        needs_garden = st.checkbox("Garden Required")
    
    # Filter data
    filtered_df = df[
        (df['distance_to_esk'] <= max_distance) &
        (df['bedrooms'].isin(bedrooms)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1]) &
        (df['property_type'].isin(property_types)) &
        (df['esk_suitability_score'] >= min_esk_score)
    ]
    
    if needs_garden:
        filtered_df = filtered_df[filtered_df['garden'] == True]
    
    st.markdown(f"**Found {len(filtered_df)} properties matching your criteria**")
    
    if len(filtered_df) > 0:
        # Show results
        for idx, row in filtered_df.head(10).iterrows():
            with st.expander(f"🏠 {row['property_id']} - {row['neighborhood']} - €{row['price']:,}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {row['property_type'].title()}")
                    st.write(f"**Bedrooms:** {row['bedrooms']}")
                    st.write(f"**Size:** {row['sqft']} sqm")
                    st.write(f"**Garden:** {'✅' if row['garden'] else '❌'}")
                
                with col2:
                    st.write(f"**Distance to ESK:** {row['distance_to_esk']} km")
                    st.write(f"**ESK Score:** {row['esk_suitability_score']}/100")
                    st.write(f"**Safety Score:** {row['safety_score']}/10")
                    st.write(f"**ESK Families Nearby:** {row['current_esk_families']}")

def show_predictions(model):
    """Price prediction page"""
    st.header("🤖 AI Price Predictions")
    
    st.markdown("Use our machine learning model to predict property prices:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("Bedrooms", 2, 5, 3)
        sqft = st.number_input("Size (sqm)", 60, 300, 120)
        distance_to_esk = st.number_input("Distance to ESK (km)", 0.5, 8.0, 2.0, 0.1)
        esk_score = st.number_input("ESK Suitability Score", 0, 100, 75)
    
    with col2:
        property_type = st.selectbox("Property Type", ['house', 'apartment'])
        has_garden = st.checkbox("Has Garden", True)
        has_garage = st.checkbox("Has Garage", False)
    
    if st.button("🔮 Predict Price", type="primary"):
        # Prepare features
        features = pd.DataFrame({
            'bedrooms': [bedrooms],
            'sqft': [sqft],
            'distance_to_esk': [distance_to_esk],
            'esk_suitability_score': [esk_score],
            'property_type_house': [1 if property_type == 'house' else 0],
            'has_garden': [1 if has_garden else 0],
            'has_garage': [1 if has_garage else 0]
        })
        
        prediction = model.predict(features)[0]
        
        st.success(f"**Predicted Price: €{prediction:,.0f}**")
        
        # Show confidence interval (rough estimate)
        margin = prediction * 0.15
        st.info(f"**Confidence Range: €{prediction-margin:,.0f} - €{prediction+margin:,.0f}**")

def show_analytics(df):
    """Quick analytics page"""
    st.header("📊 Quick Market Analytics")
    
    # Price by neighborhood
    fig1 = px.box(df, x='neighborhood', y='price', 
                  title="Price Distribution by Neighborhood")
    fig1.update_xaxis(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # ESK score vs distance
    fig2 = px.scatter(df, x='distance_to_esk', y='esk_suitability_score',
                      color='property_type', size='price',
                      title="ESK Suitability vs Distance to School")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary stats
    st.subheader("Market Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"€{df['price'].mean():,.0f}")
    with col2:
        st.metric("Median Distance to ESK", f"{df['distance_to_esk'].median():.1f} km")
    with col3:
        st.metric("Average ESK Score", f"{df['esk_suitability_score'].mean():.0f}/100")
    with col4:
        st.metric("Properties with Garden", f"{(df['garden'].sum() / len(df) * 100):.0f}%")

if __name__ == "__main__":
    main()
