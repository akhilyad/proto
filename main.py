import db
import streamlit as st
import pandas as pd
import yaml
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from streamlit_folium import st_folium
import emissions
import visualization
import ui

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    st.error("Failed to load application configuration. Please check the logs.")
    st.stop()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.source_country = next(iter(CONFIG['locations']))
    st.session_state.source_city = next(iter(CONFIG['locations'][st.session_state.source_country]))
    st.session_state.dest_country = next(iter(CONFIG['locations']))
    st.session_state.dest_city = next(iter(CONFIG['locations'][st.session_state.dest_country]))
    st.session_state.weight_tons = 1.0
    st.session_state.initialized = True

def initialize_page():
    """Initialize common page elements and settings."""
    st.set_page_config(
        page_title="CO2 Emission Calculator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for better performance
    st.markdown("""
        <style>
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }
        .stButton > button {
            width: 100%;
        }
        .stSelectbox, .stNumberInput {
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def with_spinner(func):
    """Decorator to add a spinner for long-running operations."""
    def wrapper(*args, **kwargs):
        with st.spinner("Processing..."):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                st.error(f"An error occurred: {str(e)}")
                return None
    return wrapper

@with_spinner
def calculate_emissions_with_progress(source_country, source_city, dest_country, dest_city, weight_tons, transport_mode):
    progress_bar = st.progress(0)
    try:
        progress_bar.progress(25)
        distance_km = emissions.calculate_distance(
            source_country, source_city, dest_country, dest_city,
            get_coords_func=db.get_coordinates
        )
        progress_bar.progress(50)
        co2_kg = emissions.calculate_co2(transport_mode, distance_km, weight_tons)
        carbon_cost_eur = co2_kg * (CONFIG['carbon_price_eur_per_ton'] / 1000)  # Convert kg to tons
        trees_offset = int(co2_kg / 25)  # Assuming 25 kg CO2 per tree
        # Save emission (will fall back if no database)
        db.save_emission(source_country, source_city, dest_country, dest_city, transport_mode, weight_tons, co2_kg, distance_km, carbon_cost_eur, trees_offset)
        # ... (rest of the function, e.g., display results)
    except Exception as e:
        logger.error(f"Error in emissions calculation: {e}")
        st.error(f"An error occurred: {e}")
    finally:
        progress_bar.empty()

def reset_calc_source_city():
    st.session_state.calc_source_city = list(CONFIG['locations'][st.session_state.calc_source_country].keys())[0]

def reset_calc_dest_city():
    st.session_state.calc_dest_city = list(CONFIG['locations'][st.session_state.calc_dest_country].keys())[0]

def reset_opt_source_city():
    st.session_state.opt_source_city = list(CONFIG['locations'][st.session_state.opt_source_country].keys())[0]

def reset_opt_dest_city():
    st.session_state.opt_dest_city = list(CONFIG['locations'][st.session_state.opt_dest_country].keys())[0]

def page_calculate_emissions():
    """Calculate emissions page with improved performance."""
    st.header("Calculate CO2 Emissions")

    # --- Robust initialization for dropdowns ---
    if "calc_source_country" not in st.session_state:
        st.session_state["calc_source_country"] = list(CONFIG['locations'].keys())[0]
    if "calc_source_city" not in st.session_state or \
       st.session_state["calc_source_city"] not in CONFIG['locations'][st.session_state["calc_source_country"]]:
        st.session_state["calc_source_city"] = list(CONFIG['locations'][st.session_state["calc_source_country"]].keys())[0]
    if "calc_dest_country" not in st.session_state:
        st.session_state["calc_dest_country"] = list(CONFIG['locations'].keys())[0]
    if "calc_dest_city" not in st.session_state or \
       st.session_state["calc_dest_city"] not in CONFIG['locations'][st.session_state["calc_dest_country"]]:
        st.session_state["calc_dest_city"] = list(CONFIG['locations'][st.session_state["calc_dest_country"]].keys())[0]

    # --- Source country/city selectors (OUTSIDE the form) ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source")
        source_country = st.selectbox(
            "Source Country",
            list(CONFIG['locations'].keys()),
            index=list(CONFIG['locations'].keys()).index(st.session_state.calc_source_country),
            key="calc_source_country_selector"
        )
        if source_country != st.session_state.calc_source_country:
            st.session_state.calc_source_country = source_country
            st.session_state.calc_source_city = list(CONFIG['locations'][source_country].keys())[0]
        source_city = st.selectbox(
            "Source City",
            list(CONFIG['locations'][st.session_state.calc_source_country].keys()),
            index=list(CONFIG['locations'][st.session_state.calc_source_country].keys()).index(st.session_state.calc_source_city) if st.session_state.calc_source_city in CONFIG['locations'][st.session_state.calc_source_country] else 0,
            key="calc_source_city_selector"
        )
        st.session_state.calc_source_city = source_city
    with col2:
        st.subheader("Destination")
        dest_country = st.selectbox(
            "Destination Country",
            list(CONFIG['locations'].keys()),
            index=list(CONFIG['locations'].keys()).index(st.session_state.calc_dest_country),
            key="calc_dest_country_selector"
        )
        if dest_country != st.session_state.calc_dest_country:
            st.session_state.calc_dest_country = dest_country
            st.session_state.calc_dest_city = list(CONFIG['locations'][dest_country].keys())[0]
        dest_city = st.selectbox(
            "Destination City",
            list(CONFIG['locations'][st.session_state.calc_dest_country].keys()),
            index=list(CONFIG['locations'][st.session_state.calc_dest_country].keys()).index(st.session_state.calc_dest_city) if st.session_state.calc_dest_city in CONFIG['locations'][st.session_state.calc_dest_country] else 0,
            key="calc_dest_city_selector"
        )
        st.session_state.calc_dest_city = dest_city

    # --- Input form with validation (ONLY transport mode, weight, submit) ---
    with st.form("emissions_form"):
        col3, col4 = st.columns(2)
        with col3:
            transport_mode = st.selectbox(
                "Transport Mode",
                list(CONFIG['emission_factors'].keys()),
                help="Select the mode of transportation."
            )
        with col4:
            weight_tons = st.number_input(
                "Weight (tons)",
                min_value=0.1,
                max_value=100000.0,
                value=st.session_state.weight_tons,
                step=0.1,
                help="Enter the shipment weight in tons."
            )
        submitted = st.form_submit_button("Calculate Emissions")
        if submitted:
            try:
                st.session_state.calc_source_country = source_country
                st.session_state.calc_source_city = source_city
                st.session_state.calc_dest_country = dest_country
                st.session_state.calc_dest_city = dest_city
                st.session_state.weight_tons = weight_tons
                st.session_state['last_source_country'] = source_country
                st.session_state['last_source_city'] = source_city
                st.session_state['last_dest_country'] = dest_country
                st.session_state['last_dest_city'] = dest_city
                st.session_state['last_weight_tons'] = weight_tons
                results = calculate_emissions_with_progress(
                    st.session_state.calc_source_country,
                    st.session_state.calc_source_city,
                    st.session_state.calc_dest_country,
                    st.session_state.calc_dest_city,
                    weight_tons
                )
                if results:
                    st.subheader("Emission Results")
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("CO2 Emissions", f"{results['co2_kg']:.2f} kg")
                    with col6:
                        st.metric("Carbon Cost (EUR)", f"{results['carbon_cost_eur']:.2f}")
                    with col7:
                        st.metric("Distance", f"{results['distance_km']:.2f} km")
                    with col8:
                        st.metric("Trees to Offset", f"{int(results['trees_equivalent'])}")
                    db.save_emission(
                        f"{st.session_state.calc_source_city}, {st.session_state.calc_source_country}",
                        f"{st.session_state.calc_dest_city}, {st.session_state.calc_dest_country}",
                        transport_mode,
                        results['distance_km'],
                        results['co2_kg'],
                        weight_tons
                    )
                    with st.spinner("Generating map..."):
                        m = visualization.render_emission_map(
                            pd.DataFrame([{
                                'source_country': st.session_state.calc_source_country,
                                'source_city': st.session_state.calc_source_city,
                                'dest_country': st.session_state.calc_dest_country,
                                'dest_city': st.session_state.calc_dest_city,
                                'source': f"{st.session_state.calc_source_city}, {st.session_state.calc_source_country}",
                                'destination': f"{st.session_state.calc_dest_city}, {st.session_state.calc_dest_country}",
                                'co2_kg': results['co2_kg']
                            }]),
                            db.get_coordinates,
                            max_routes=1
                        )
                        st_folium(m, width=900, height=400) 
            except Exception as e:
                logger.error(f"Error in emissions calculation: {e}")
                st.error(f"An error occurred: {str(e)}")

def page_route_visualizer():
    st.header("Emission Hotspot Visualizer")
    try:
        emissions_df = db.get_emissions()
        if not emissions_df.empty:
            emissions_df['source_country'] = emissions_df['source'].apply(lambda x: x.split(', ')[1])
            emissions_df['source_city'] = emissions_df['source'].apply(lambda x: x.split(', ')[0])
            emissions_df['dest_country'] = emissions_df['destination'].apply(lambda x: x.split(', ')[1])
            emissions_df['dest_city'] = emissions_df['destination'].apply(lambda x: x.split(', ')[0])

            with st.spinner("Loading map..."):
                m = visualization.render_emission_map(emissions_df, db.get_coordinates)
                st_folium(m, width=1200, height=600)

            st.subheader("Route Analytics Dashboard")
            routes = [f"Route {idx + 1}: {row['source']} to {row['destination']}" for idx, row in emissions_df.iterrows()]
            selected_route = st.selectbox(
                "Select Route to Analyze",
                routes,
                help="Choose a route to view optimization details."
            )
            route_idx = int(selected_route.split(":")[0].split(" ")[1]) - 1
            row = emissions_df.iloc[route_idx]
            source_country = row['source_country']
            source_city = row['source_city']
            dest_country = row['dest_country']
            dest_city = row['dest_city']
            distance_km = row['distance_km']
            weight_tons = row['weight_tons']
            current_co2 = row['co2_kg']
            current_mode = row['transport_mode']
            max_delivery_days = st.number_input(
                "Max Delivery Days (Optional)",
                min_value=0.0,
                max_value=30.0,
                value=0.0,
                step=0.1,
                help="Set maximum delivery time in days (0 for no constraint)."
            )
            cost_weight = st.slider(
                "Cost vs CO2 Priority",
                0.0,
                1.0,
                0.2,
                help="0 = Prioritize CO2 reduction, 1 = Prioritize cost."
            )
            try:
                # NEW: Optimized route with flight option and cost consideration
                best_option, min_co2, breakdown, distances, costs, delivery_time = emissions.optimize_route_advanced(
                    source_country, source_city, dest_country, dest_city, distance_km, weight_tons,
                    max_delivery_days=max_delivery_days if max_delivery_days > 0 else None,
                    cost_weight=cost_weight
                )
                modes, ratios = best_option
                co2_values = breakdown
                dist_values = distances
                cost_values = costs
                savings = current_co2 - min_co2
                savings_pct = (savings / current_co2 * 100) if current_co2 != 0 else 0

                # Create route data for multi-modal visualization
                route_data = {
                    'source': {'country': source_country, 'city': source_city},
                    'destination': {'country': dest_country, 'city': dest_city},
                    'total_distance': distance_km,
                    'total_co2': min_co2,
                    'total_cost': sum(cost_values),
                    'delivery_time': delivery_time,
                    'segments': [
                        {
                            'mode': mode,
                            'ratio': ratio,
                            'distance': dist,
                            'co2': co2_val,
                            'cost': cost_val
                        }
                        for mode, ratio, dist, co2_val, cost_val in zip(modes, ratios, dist_values, co2_values, cost_values)
                        if ratio > 0
                    ]
                }

                # Render multi-modal route visualization
                with st.spinner("Rendering optimized route..."):
                    m = visualization.render_multi_modal_route(route_data, db.get_coordinates)
                    st_folium(m, width=1200, height=600)

                st.subheader("Key Performance Indicators (KPIs)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Distance", f"{distance_km:.2f} km")
                with col2:
                    st.metric("Optimized CO2 Emissions", f"{min_co2:.2f} kg")
                with col3:
                    st.metric("CO2 Savings", f"{savings:.2f} kg ({savings_pct:.1f}% reduction)")
                with col4:
                    st.metric("Total Cost (USD)", f"{sum(cost_values):.2f}")

                tab1, tab2, tab3 = st.tabs(["Route Breakdown", "Comparison Chart", "Mode Contribution"])
                with tab1:
                    st.write("**Optimized Route Breakdown**")
                    for seg in route_data['segments']:
                        st.write(f"- **{seg['mode']}**: {seg['distance']:.2f} km, CO2: {seg['co2']:.2f} kg, Cost: ${seg['cost']:.2f}")
                    st.write(f"**Delivery Time**: {delivery_time:.2f} days")
                with tab2:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[current_co2, min_co2],
                        y=['Current Route', 'Optimized Route'],
                        orientation='h',
                        name='CO2 Emissions (kg)',
                        marker_color=['#FF4B4B', '#36A2EB']
                    ))
                    fig.add_trace(go.Bar(
                        x=[distance_km, sum(dist_values)],
                        y=['Current Route', 'Optimized Route'],
                        orientation='h',
                        name='Distance (km)',
                        marker_color=['#FF9999', '#66B3FF']
                    ))
                    fig.update_layout(
                        title="Current vs Optimized Route Comparison",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with tab3:
                    import plotly.express as px
                    fig = px.pie(
                        values=[seg['co2'] for seg in route_data['segments']],
                        names=[seg['mode'] for seg in route_data['segments']],
                        title="CO2 Contribution by Mode"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                ui.show_error(str(e), f"Cannot optimize route: {str(e)}.")
        else:
            st.info("No emission routes to display. Calculate some emissions first!")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        ui.show_error(str(e), "Failed to load emission data.")

def page_supplier_lookup():
    st.header("Supplier Lookup Dashboard")
    LOCATIONS = CONFIG['locations']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        country = st.selectbox(
            "Country",
            ["All"] + list(LOCATIONS.keys()),
            help="Filter suppliers by country (select 'All' for no filter)."
        )
    with col2:
        cities = ["All"] + list(LOCATIONS.get(country, {}).keys()) if country != "All" else ["All"]
        city = st.selectbox(
            "City",
            cities,
            help="Filter suppliers by city (select 'All' for no filter)."
        )
    with col3:
        material = st.text_input(
            "Material (e.g., Steel, Electronics)",
            help="Enter a material to filter suppliers (case-insensitive)."
        )
    with col4:
        min_green_score = st.slider(
            "Minimum Green Score",
            0,
            100,
            50,
            help="Filter suppliers with a green score above this value (0-100)."
        )
        min_date = st.date_input(
            "Added After",
            value=None,
            help="Show suppliers added after this date (optional)."
        )
    try:
        suppliers = db.get_suppliers(
            country if country != "All" else None,
            city if city != "All" else None,
            material or None,
            min_green_score,
            min_date.strftime('%Y-%m-%d') if min_date else None
        )
        if not suppliers.empty:
            st.subheader("Key Performance Indicators (KPIs)")
            col4, col5, col6, col7 = st.columns(4)
            with col4:
                st.metric("Total Suppliers", len(suppliers))
            with col5:
                st.metric("Average Green Score", f"{suppliers['green_score'].mean():.1f}")
            with col6:
                st.metric("Total Capacity", f"{suppliers['annual_capacity_tons'].sum():,} tons")
            potential_savings = 0
            if 'source_country' in st.session_state and 'dest_country' in st.session_state:
                source_country = st.session_state.source_country
                dest_country = st.session_state.dest_country
                weight_tons = st.session_state.weight_tons
                if source_country != dest_country:
                    try:
                        from emissions import calculate_distance, EMISSION_FACTORS
                        distance_km = calculate_distance(
                            source_country, list(LOCATIONS[source_country].keys())[0],
                            dest_country, list(LOCATIONS[dest_country].keys())[0],
                            get_coords_func=db.get_coordinates
                        )
                        current_co2 = distance_km * weight_tons * EMISSION_FACTORS['Truck']
                        local_suppliers = suppliers[suppliers['country'] == dest_country]
                        if not local_suppliers.empty:
                            potential_savings = current_co2
                            st.success(
                                f"🌍 **Local Sourcing Opportunity**: Source from {dest_country} to save {potential_savings:.2f} kg CO2."
                            )
                        else:
                            st.info(f"No suppliers found in {dest_country}.")
                    except Exception as e:
                        ui.show_error(str(e), f"Cannot calculate savings: {str(e)}.")
                else:
                    st.info("Select different source and destination countries to calculate potential savings.")
            with col7:
                st.metric("Potential CO2 Savings", f"{potential_savings:.2f} kg")
            st.subheader("Supplier Insights")
            tab1, tab2, tab3 = st.tabs(["Supplier Distribution", "Material Availability", "Supplier Details"])
            with tab1:
                fig = visualization.bar_chart(
                    suppliers.groupby('country').size().reset_index(name='Count'),
                    x='country', y='Count', title="Suppliers by Country"
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                fig = visualization.bar_chart(
                    suppliers.groupby('material')['annual_capacity_tons'].sum().reset_index(),
                    x='material', y='annual_capacity_tons', title="Material Capacity"
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab3:
                st.dataframe(suppliers[['supplier_name', 'country', 'city', 'material', 'green_score', 'sustainable_practices', 'created_at']])
        else:
            st.info("No suppliers found for the given criteria.")
    except Exception as e:
        ui.show_error(str(e), "Failed to load supplier data.")

def page_reports():
    st.header("Emission Reports")
    try:
        emissions_df = db.get_emissions()
        if not emissions_df.empty:
            total_co2 = emissions_df['co2_kg'].sum()
            avg_co2 = emissions_df['co2_kg'].mean()
            total_shipments = len(emissions_df)
            total_savings = 0
            route_data = []
            for _, row in emissions_df.iterrows():
                source_country = row['source'].split(', ')[1]
                source_city = row['source'].split(', ')[0]
                dest_country = row['destination'].split(', ')[1]
                dest_city = row['destination'].split(', ')[0]
                distance_km = row['distance_km']
                weight_tons = row['weight_tons']
                current_co2 = row['co2_kg']
                current_mode = row['transport_mode']
                try:
                    best_option, min_co2, breakdown, distances, costs, delivery_time = emissions.optimize_route_advanced(
                        source_country, source_city, dest_country, dest_city, distance_km, weight_tons,
                        max_delivery_days=None, cost_weight=0.2
                    )
                    modes, ratios = best_option
                    co2_values = breakdown
                    dist_values = distances
                    cost_values = costs
                    savings = current_co2 - min_co2
                    total_savings += savings
                    route_data.append({
                        'Route': f"{source_city}, {source_country} to {dest_city}, {dest_country}",
                        'Old Mode': current_mode,
                        'Old Distance': distance_km,
                        'Old CO2': current_co2,
                        'New Modes': ' + '.join([m for m, r in zip(modes, ratios) if r > 0]),
                        'New Distances': ' + '.join([f"{d:.2f} km ({m})" for m, d, r in zip(modes, dist_values, ratios) if r > 0]),
                        'New CO2': min_co2,
                        'New Cost': sum(cost_values),
                        'Savings': savings
                    })
                except Exception as e:
                    st.warning(f"Skipping route optimization for {source_city} to {dest_city}: {e}")
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "CO2 Insights", "Route Optimization", "Detailed Data"])
            with tab1:
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total CO2 Emissions", f"{total_co2:.2f} kg")
                with col2:
                    st.metric("Total Shipments", f"{total_shipments}")
                with col3:
                    st.metric("Average CO2 per Shipment", f"{avg_co2:.2f} kg")
                with col4:
                    st.metric("Total CO2 Savings", f"{total_savings:.2f} kg")
                st.subheader("Emission Breakdown by Transport Mode")
                mode_summary = emissions_df.groupby('transport_mode')['co2_kg'].sum().reset_index()
                fig = visualization.pie_chart(mode_summary, values='co2_kg', names='transport_mode', title="CO2 by Mode")
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                st.subheader("CO2 Impact Insights")
                smartphone_charges = total_co2 * 1000 / 0.008
                ev_distance = total_co2 / 0.2
                trees_needed = total_co2 * 0.04
                st.write(f"**Environmental Impact**: The {total_co2:.2f} kg of CO2 could:")
                st.write(f"- Charge {int(smartphone_charges):,} smartphones.")
                st.write(f"- Power an EV for {ev_distance:.0f} km.")
                st.write(f"- Be offset by planting {int(trees_needed):,} trees.")
                st.subheader("Cost Savings Analysis")
                for currency, rate in CONFIG['exchange_rates'].items():
                    cost_savings = total_savings / 1000 * CONFIG['carbon_price_eur_per_ton'] * rate
                    st.write(f"- **{currency}**: {cost_savings:.2f}")
            with tab3:
                st.subheader("Route Optimization Summary")
                st.dataframe(pd.DataFrame(route_data))
            with tab4:
                st.subheader("Detailed Emission Data")
                st.dataframe(emissions_df)
                csv = emissions_df.to_csv(index=False)
                st.download_button(
                    label="Download Emission Data as CSV",
                    data=csv,
                    file_name="emissions_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No emission data available. Calculate some emissions first!")
    except Exception as e:
        ui.show_error(str(e), "Failed to load report data.")

def page_optimized_route_planning():
    st.header("Optimized Route Planning")
    LOCATIONS = CONFIG['locations']
    CARBON_PRICE_EUR_PER_TON = CONFIG['carbon_price_eur_per_ton']
    if 'last_source_country' in st.session_state:
        default_source_country = st.session_state['last_source_country']
    else:
        default_source_country = list(LOCATIONS.keys())[0]
    if 'last_source_city' in st.session_state and st.session_state['last_source_city'] in LOCATIONS.get(default_source_country, {}):
        default_source_city = st.session_state['last_source_city']
    else:
        default_source_city = list(LOCATIONS[default_source_country].keys())[0]
    if 'last_dest_country' in st.session_state:
        default_dest_country = st.session_state['last_dest_country']
    else:
        default_dest_country = list(LOCATIONS.keys())[0]
    if 'last_dest_city' in st.session_state and st.session_state['last_dest_city'] in LOCATIONS.get(default_dest_country, {}):
        default_dest_city = st.session_state['last_dest_city']
    else:
        default_dest_city = list(LOCATIONS[default_dest_country].keys())[0]
    if 'last_weight_tons' in st.session_state:
        default_weight_tons = st.session_state['last_weight_tons']
    else:
        default_weight_tons = 1.0
    if 'opt_source_country' not in st.session_state:
        st.session_state['opt_source_country'] = default_source_country
    if 'opt_source_city' not in st.session_state or st.session_state['opt_source_city'] not in LOCATIONS[st.session_state['opt_source_country']]:
        st.session_state['opt_source_city'] = default_source_city
    if 'opt_dest_country' not in st.session_state:
        st.session_state['opt_dest_country'] = default_dest_country
    if 'opt_dest_city' not in st.session_state or st.session_state['opt_dest_city'] not in LOCATIONS[st.session_state['opt_dest_country']]:
        st.session_state['opt_dest_city'] = default_dest_city
    if 'weight_tons' not in st.session_state:
        st.session_state.weight_tons = default_weight_tons
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source")
        source_country = st.selectbox(
            "Source Country",
            list(LOCATIONS.keys()),
            index=list(LOCATIONS.keys()).index(st.session_state.opt_source_country),
            key="opt_source_country_selector"
        )
        if source_country != st.session_state.opt_source_country:
            st.session_state.opt_source_country = source_country
            st.session_state.opt_source_city = list(LOCATIONS[source_country].keys())[0]
        source_city = st.selectbox(
            "Source City",
            list(LOCATIONS[st.session_state.opt_source_country].keys()),
            index=list(LOCATIONS[st.session_state.opt_source_country].keys()).index(st.session_state.opt_source_city) if st.session_state.opt_source_city in LOCATIONS[st.session_state.opt_source_country] else 0,
            key="opt_source_city_selector"
        )
        st.session_state.opt_source_city = source_city
        st.subheader("Destination")
        dest_country = st.selectbox(
            "Destination Country",
            list(LOCATIONS.keys()),
            index=list(LOCATIONS.keys()).index(st.session_state.opt_dest_country),
            key="opt_dest_country_selector"
        )
        if dest_country != st.session_state.opt_dest_country:
            st.session_state.opt_dest_country = dest_country
            st.session_state.opt_dest_city = list(LOCATIONS[dest_country].keys())[0]
        dest_city = st.selectbox(
            "Destination City",
            list(LOCATIONS[st.session_state.opt_dest_country].keys()),
            index=list(LOCATIONS[st.session_state.opt_dest_country].keys()).index(st.session_state.opt_dest_city) if st.session_state.opt_dest_city in LOCATIONS[st.session_state.opt_dest_country] else 0,
            key="opt_dest_city_selector"
        )
        st.session_state.opt_dest_city = dest_city
    with col2:
        weight_tons = st.number_input(
            "Weight (tons)",
            min_value=0.1,
            max_value=100000.0,
            value=st.session_state.weight_tons,
            step=0.1,
            help="Enter the shipment weight in tons."
        )
        max_delivery_days = st.number_input(
            "Max Delivery Days (Optional)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.1,
            help="Set maximum delivery time in days (0 for no constraint)."
        )
        cost_weight = st.slider(
            "Cost vs CO2 Priority",
            0.0,
            1.0,
            0.2,
            help="0 = Prioritize CO2 reduction, 1 = Prioritize cost."
        )
        st.session_state.weight_tons = weight_tons
        try:
            distance_km = emissions.calculate_distance(
                st.session_state.opt_source_country,
                st.session_state.opt_source_city,
                st.session_state.opt_dest_country,
                st.session_state.opt_dest_city,
                get_coords_func=db.get_coordinates
            )
            st.write(f"Estimated Distance: {distance_km} km")
        except ValueError as e:
            ui.show_error(str(e), f"Cannot calculate distance: {str(e)}. Please select different locations.")
            distance_km = 0.0
    if st.button("Optimize Route") and distance_km > 0:
        try:
            best_option, min_co2, breakdown, distances, costs, delivery_time = emissions.optimize_route_advanced(
                st.session_state.opt_source_country,
                st.session_state.opt_source_city,
                st.session_state.opt_dest_country,
                st.session_state.opt_dest_city,
                distance_km, weight_tons,
                max_delivery_days=max_delivery_days if max_delivery_days > 0 else None,
                cost_weight=cost_weight
            )
            modes, ratios = best_option
            co2_values = breakdown
            dist_values = distances
            cost_values = costs
            savings = emissions.calculate_co2('Truck', distance_km, weight_tons) - min_co2
            savings_pct = (savings / min_co2 * 100) if min_co2 != 0 else 0
            cost_savings_eur = savings / 1000 * CARBON_PRICE_EUR_PER_TON
            trees_equivalent = savings * 0.04

            route_data = {
                'source': {
                    'country': st.session_state.opt_source_country,
                    'city': st.session_state.opt_source_city
                },
                'destination': {
                    'country': st.session_state.opt_dest_country,
                    'city': st.session_state.opt_dest_city
                },
                'total_distance': distance_km,
                'total_co2': min_co2,
                'total_cost': sum(cost_values),
                'delivery_time': delivery_time,
                'segments': [
                    {
                        'mode': mode,
                        'ratio': ratio,
                        'distance': dist,
                        'co2': co2_val,
                        'cost': cost_val
                    }
                    for mode, ratio, dist, co2_val, cost_val in zip(modes, ratios, dist_values, co2_values, cost_values)
                    if ratio > 0
                ]
            }
            with st.spinner("Rendering optimized route..."):
                m = visualization.render_multi_modal_route(route_data, db.get_coordinates)
                st_folium(m, width=900, height=400)
            st.subheader("Optimization Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Optimized CO2 Emissions", f"{min_co2:.2f} kg")
            with col2:
                st.metric("CO2 Savings", f"{savings:.2f} kg ({savings_pct:.1f}%)")
            with col3:
                st.metric("Cost Savings (EUR)", f"{cost_savings_eur:.2f}")
            with col4:
                st.metric("Delivery Time", f"{delivery_time:.2f} days")
            tab1, tab2, tab3 = st.tabs(["Route Breakdown", "CO2 Comparison", "Mode Contribution"])
            with tab1:
                st.write("**Optimized Route Breakdown**")
                for seg in route_data['segments']:
                    st.write(f"- **{seg['mode']}**: {seg['distance']:.2f} km, CO2: {seg['co2']:.2f} kg, Cost: ${seg['cost']:.2f}")
                st.write(f"**Delivery Time**: {delivery_time:.2f} days")
            with tab2:
                import plotly.express as px
                fig = px.bar(
                    x=[emissions.calculate_co2('Truck', distance_km, weight_tons), min_co2],
                    y=['Current Truck Route', 'Optimized Route'],
                    title="CO2 Emissions Comparison",
                    labels={'x': 'CO2 Emissions (kg)', 'y': 'Route'}
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab3:
                import plotly.express as px
                fig = px.pie(
                    values=[seg['co2'] for seg in route_data['segments']],
                    names=[seg['mode'] for seg in route_data['segments']],
                    title="CO2 Contribution by Mode"
                )
                st.plotly_chart(fig, use_container_width=True)
        except ValueError as e:
            ui.show_error(str(e), f"Cannot optimize route: {str(e)}.")

def page_green_warehousing():
    st.header("Green Warehousing Analysis")
    if 'warehouse_size_m2' not in st.session_state:
        st.session_state.warehouse_size_m2 = 1000.0
    if 'led_percentage' not in st.session_state:
        st.session_state.led_percentage = 0.5
    if 'solar_percentage' not in st.session_state:
        st.session_state.solar_percentage = 0.3
    col1, col2 = st.columns(2)
    with col1:
        warehouse_size_m2 = st.number_input(
            "Warehouse Size (m²)",
            min_value=100.0,
            max_value=100000.0,
            value=st.session_state.warehouse_size_m2,
            step=100.0,
            help="Enter the warehouse size in square meters."
        )
        led_percentage = st.slider(
            "LED Lighting Usage (%)",
            0.0,
            100.0,
            st.session_state.led_percentage * 100,
            help="Percentage of lighting using LED technology."
        ) / 100
        solar_percentage = st.slider(
            "Solar Panel Usage (%)",
            0.0,
            100.0,
            st.session_state.solar_percentage * 100,
            help="Percentage of energy from solar panels."
        ) / 100
        st.session_state.warehouse_size_m2 = warehouse_size_m2
        st.session_state.led_percentage = led_percentage
        st.session_state.solar_percentage = solar_percentage
    with col2:
        try:
            co2_savings_kg, energy_savings_kwh = emissions.calculate_warehouse_savings(warehouse_size_m2, led_percentage, solar_percentage)
            energy_cost_savings = energy_savings_kwh * 0.15
            car_miles_equivalent = co2_savings_kg / 0.4
            st.subheader("Savings Metrics")
            st.metric("CO2 Savings", f"{co2_savings_kg:.2f} kg")
            st.metric("Energy Savings", f"{energy_savings_kwh:.2f} kWh")
            st.metric("Cost Savings (USD)", f"{energy_cost_savings:.2f}")
            st.metric("Car Miles Equivalent", f"{int(car_miles_equivalent)} miles")
        except ValueError as e:
            ui.show_error(str(e), f"Calculation failed: {str(e)}.")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Calculate Savings"):
            try:
                tab1, tab2 = st.tabs(["Savings Breakdown", "Trend Analysis"])
                with tab1:
                    import plotly.express as px
                    fig = px.bar(
                        x=['LED Lighting', 'Solar Panels'],
                        y=[led_percentage * co2_savings_kg, solar_percentage * co2_savings_kg],
                        title="CO2 Savings by Technology",
                        labels={'x': 'Technology', 'y': 'CO2 Savings (kg)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    sizes = range(100, int(warehouse_size_m2) + 1000, 1000)
                    savings = [emissions.calculate_warehouse_savings(size, led_percentage, solar_percentage)[0] for size in sizes]
                    fig = visualization.line_chart(
                        pd.DataFrame({'Warehouse Size (m²)': list(sizes), 'CO2 Savings (kg)': savings}),
                        x='Warehouse Size (m²)', y='CO2 Savings (kg)', title="CO2 Savings vs Warehouse Size"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                ui.show_error(str(e), f"Visualization failed: {str(e)}.")
    with col_btn2:
        if st.button("Reset Inputs"):
            st.session_state.warehouse_size_m2 = 1000.0
            st.session_state.led_percentage = 0.5
            st.session_state.solar_percentage = 0.3
            st.rerun()

def page_sustainable_packaging():
    st.header("Sustainable Packaging Analysis")
    PACKAGING_EMISSIONS = CONFIG['packaging_emissions']
    PACKAGING_COSTS = CONFIG['packaging_costs']
    if 'material_type' not in st.session_state:
        st.session_state.material_type = 'Plastic'
    if 'weight_kg' not in st.session_state:
        st.session_state.weight_kg = 1.0
    col1, col2 = st.columns(2)
    with col1:
        material_type = st.selectbox(
            "Packaging Material",
            list(PACKAGING_EMISSIONS.keys()),
            index=list(PACKAGING_EMISSIONS.keys()).index(st.session_state.material_type),
            help="Select the packaging material."
        )
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=0.1,
            max_value=10000.0,
            value=st.session_state.weight_kg,
            step=0.1,
            help="Enter the packaging weight in kilograms."
        )
        st.session_state.material_type = material_type
        st.session_state.weight_kg = weight_kg
    with col2:
        try:
            co2_kg = weight_kg * PACKAGING_EMISSIONS[material_type]
            cost_impact = weight_kg * PACKAGING_COSTS[material_type]
            biodegradable_co2 = weight_kg * PACKAGING_EMISSIONS['Biodegradable']
            potential_savings = co2_kg - biodegradable_co2
            plastic_bottles_equivalent = co2_kg / 0.082
            st.subheader("Packaging Metrics")
            st.metric("CO2 Emissions", f"{co2_kg:.2f} kg")
            st.metric("Potential CO2 Savings", f"{potential_savings:.2f} kg")
            st.metric("Cost Impact (USD)", f"{cost_impact:.2f}")
            st.metric("Plastic Bottles Equivalent", f"{int(plastic_bottles_equivalent)} bottles")
            db.save_packaging(material_type, weight_kg, co2_kg)
            if material_type != 'Biodegradable':
                st.success(f"Switch to Biodegradable to save {potential_savings:.2f} kg CO2!")
        except Exception as e:
            ui.show_error(str(e), f"Calculation failed: {str(e)}.")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Analyze Packaging"):
            try:
                packaging = db.get_packaging()
                tab1, tab2 = st.tabs(["Material Comparison", "Historical Trends"])
                with tab1:
                    import plotly.express as px
                    fig = px.bar(
                        x=list(PACKAGING_EMISSIONS.keys()),
                        y=[weight_kg * PACKAGING_EMISSIONS[mat] for mat in PACKAGING_EMISSIONS],
                        title="CO2 Emissions by Material",
                        labels={'x': 'Material', 'y': 'CO2 Emissions (kg)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    if not packaging.empty:
                        fig = visualization.line_chart(
                            packaging,
                            x='timestamp',
                            y='co2_kg',
                            title="Historical Packaging Emissions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No historical packaging data available.")
            except Exception as e:
                ui.show_error(str(e), f"Visualization failed: {str(e)}.")
    with col_btn2:
        if st.button("Reset Inputs"):
            st.session_state.material_type = 'Plastic'
            st.session_state.weight_kg = 1.0
            st.rerun()

def page_efficient_load_management():
    st.header("Efficient Load Management")
    if 'weight_tons' not in st.session_state:
        st.session_state.weight_tons = 10.0
    if 'vehicle_capacity_tons' not in st.session_state:
        st.session_state.vehicle_capacity_tons = 20.0
    if 'avg_trip_distance_km' not in st.session_state:
        st.session_state.avg_trip_distance_km = 100.0
    col1, col2 = st.columns(2)
    with col1:
        weight_tons = st.number_input(
            "Total Weight (tons)",
            min_value=0.1,
            max_value=100000.0,
            value=st.session_state.weight_tons,
            step=0.1,
            help="Enter the total weight to transport in tons."
        )
        vehicle_capacity_tons = st.number_input(
            "Vehicle Capacity (tons)",
            min_value=0.1,
            max_value=1000.0,
            value=st.session_state.vehicle_capacity_tons,
            step=0.1,
            help="Enter the vehicle capacity in tons."
        )
        avg_trip_distance_km = st.number_input(
            "Average Trip Distance (km)",
            min_value=1.0,
            max_value=10000.0,
            value=st.session_state.avg_trip_distance_km,
            step=10.0,
            help="Enter the average trip distance in kilometers."
        )
        st.session_state.weight_tons = weight_tons
        st.session_state.vehicle_capacity_tons = vehicle_capacity_tons
        st.session_state.avg_trip_distance_km = avg_trip_distance_km
    with col2:
        try:
            trips_saved, co2_savings_kg = emissions.calculate_load_optimization(weight_tons, vehicle_capacity_tons, avg_trip_distance_km)
            fuel_cost_savings = co2_savings_kg * 0.05
            flights_equivalent = co2_savings_kg / 90
            st.subheader("Load Optimization Metrics")
            st.metric("Trips Saved", f"{trips_saved}")
            st.metric("CO2 Savings", f"{co2_savings_kg:.2f} kg")
            st.metric("Fuel Cost Savings (USD)", f"{fuel_cost_savings:.2f}")
            st.metric("Flights Equivalent", f"{int(flights_equivalent)} flights")
        except ValueError as e:
            ui.show_error(str(e), f"Calculation failed: {str(e)}.")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Optimize Load"):
            try:
                tab1, tab2 = st.tabs(["Savings Breakdown", "Weight Sensitivity"])
                with tab1:
                    import plotly.express as px
                    fig = px.bar(
                        x=['Trips Saved', 'CO2 Savings'],
                        y=[trips_saved, co2_savings_kg],
                        title="Load Optimization Savings",
                        labels={'x': 'Metric', 'y': 'Value'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    weights = range(int(weight_tons / 2), int(weight_tons * 2), max(1, int(weight_tons / 10)))
                    savings = [emissions.calculate_load_optimization(w, vehicle_capacity_tons, avg_trip_distance_km)[1] for w in weights]
                    fig = visualization.line_chart(
                        pd.DataFrame({'Total Weight (tons)': list(weights), 'CO2 Savings (kg)': savings}),
                        x='Total Weight (tons)', y='CO2 Savings (kg)', title="CO2 Savings vs Total Weight"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                ui.show_error(str(e), f"Visualization failed: {str(e)}.")
    with col_btn2:
        if st.button("Reset Inputs"):
            st.session_state.weight_tons = 10.0
            st.session_state.vehicle_capacity_tons = 20.0
            st.session_state.avg_trip_distance_km = 100.0
            st.rerun()

def page_energy_conservation():
    st.header("Energy Conservation Analysis")
    if 'facility_size_m2' not in st.session_state:
        st.session_state.facility_size_m2 = 1000.0
    if 'smart_system_usage' not in st.session_state:
        st.session_state.smart_system_usage = 0.5
    col1, col2 = st.columns(2)
    with col1:
        facility_size_m2 = st.number_input(
            "Facility Size (m²)",
            min_value=100.0,
            max_value=100000.0,
            value=st.session_state.facility_size_m2,
            step=100.0,
            help="Enter the facility size in square meters."
        )
        smart_system_usage = st.slider(
            "Smart System Usage (%)",
            0.0,
            100.0,
            st.session_state.smart_system_usage * 100,
            help="Percentage of energy managed by smart systems."
        ) / 100
        st.session_state.facility_size_m2 = facility_size_m2
        st.session_state.smart_system_usage = smart_system_usage
    with col2:
        try:
            traditional_energy_kwh = facility_size_m2 * 150
            energy_savings_kwh = traditional_energy_kwh * smart_system_usage * 0.4
            co2_savings_kg = energy_savings_kwh * 0.5
            cost_savings = energy_savings_kwh * 0.15
            household_equivalent = energy_savings_kwh / 10000
            st.subheader("Energy Conservation Metrics")
            st.metric("CO2 Savings", f"{co2_savings_kg:.2f} kg")
            st.metric("Energy Savings", f"{energy_savings_kwh:.2f} kWh")
            st.metric("Cost Savings (USD)", f"{cost_savings:.2f}")
            st.metric("Household Equivalent", f"{int(household_equivalent)} households")
        except ValueError as e:
            ui.show_error(str(e), f"Calculation failed: {str(e)}")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Analyze Energy Savings"):
            try:
                tab1, tab2 = st.tabs(["Savings Breakdown", "Size Sensitivity"])
                with tab1:
                    import plotly.express as px
                    fig = px.bar(
                        x=['Smart Systems'],
                        y=[co2_savings_kg],
                        title="CO2 Savings from Smart Systems",
                        labels={'x': 'Technology', 'y': 'CO2 Savings (kg)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    sizes = range(100, int(facility_size_m2) + 1000, 1000)
                    savings = [size * 150 * smart_system_usage * 0.4 * 0.5 for size in sizes]
                    fig = visualization.line_chart(
                        pd.DataFrame({'Facility Size (m²)': list(sizes), 'CO2 Savings (kg)': savings}),
                        x='Facility Size (m²)', y='CO2 Savings (kg)', title="CO2 Savings vs Facility Size"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                ui.show_error(str(e), f"Visualization failed: {str(e)}")
    with col_btn2:
        if st.button("Reset Inputs"):
            st.session_state.facility_size_m2 = 1000.0
            st.session_state.smart_system_usage = 0.5
            st.rerun()

def main():
    initialize_page()
    st.title("CO2 Emission Calculator & Sustainability Analytics")
    
    @st.cache_data(ttl=3600)
    def get_available_pages():
        return {
            "Calculate Emissions": page_calculate_emissions,
            "Route Visualizer": page_route_visualizer,
            "Supplier Lookup": page_supplier_lookup,
            "Reports": page_reports,
            "Optimized Route Planning": page_optimized_route_planning,
            "Green Warehousing": page_green_warehousing,
            "Sustainable Packaging": page_sustainable_packaging,
            "Efficient Load Management": page_efficient_load_management,
            "Energy Conservation": page_energy_conservation,
        }
    
    pages = get_available_pages()
    page = st.sidebar.selectbox("Select a page", list(pages.keys()))
    
    try:
        pages[page]()
    except Exception as e:
        logger.error(f"Error in page {page}: {e}")
        st.error(f"An error occurred while loading the page. Please try again or contact support.")
        st.stop()

if __name__ == "__main__":
    main()
