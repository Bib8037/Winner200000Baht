import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import date, timedelta
#test

# --- Page Configuration ---
st.set_page_config(
    page_title="MMA Sales Optimizer",
    page_icon="📈",
    layout="wide"
)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("MMA Sales Optimizer")
    
    # Add icons to the navigation
    page_icons = {
        "Dashboard": "📊",
        "Order Processing": "📋",
        "Logistics": "🚚",
        "Sales Strategy": "⭐",
        "Truck Management": "🛻",
        "OCR": "📝",
        "Logistics & TruckOptimization": "🚛"
    }
    
    st.markdown("---")
    st.markdown("### Navigate to:")

    def safe_rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    
    if st.button(f'{page_icons["Dashboard"]} Dashboard', use_container_width=True):
        st.session_state.page = "Dashboard"
        # st.experimental_rerun()
        safe_rerun()
    if st.button(f'{page_icons["Order Processing"]} Order Processing', use_container_width=True):
        st.session_state.page = "Order Processing"
        # st.experimental_rerun()
        safe_rerun()
    if st.button(f'{page_icons["Logistics"]} Logistics', use_container_width=True):
        st.session_state.page = "Logistics"
        # st.experimental_rerun()
        safe_rerun()
    if st.button(f'{page_icons["Sales Strategy"]} Sales Strategy', use_container_width=True):
        st.session_state.page = "Sales Strategy"
        # st.experimental_rerun()
        safe_rerun()
    if st.button(f'{page_icons["Truck Management"]} Truck Management', use_container_width=True):
        st.session_state.page = "Truck Management"
        # st.experimental_rerun()
        safe_rerun()
    if st.button(f'{page_icons["OCR"]} OCR', use_container_width=True):
        st.session_state.page = "OCR"
        # st.experimental_rerun()
        safe_rerun()
    if st.button(f'{page_icons["Logistics Optimization"]} Logistics Optimization', use_container_width=True):
        st.session_state.page = "Logistics Optimization"
        # st.experimental_rerun()
        safe_rerun()
# =================================================================================
# --- OCR PAGE ---
# =================================================================================
def show_ocr():
    st.title("PDF Text & OCR Extractor")
    st.write("Upload a PDF to extract text. If no text is found, OCR will be used.")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        try:
            import fitz  # PyMuPDF
            import pytesseract
            from PIL import Image
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted_text = ""
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    extracted_text += f"\n--- Page {i+1} (Text) ---\n" + text
                else:
                    # Convert page to image and run OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang='eng')
                    extracted_text += f"\n--- Page {i+1} (OCR) ---\n" + ocr_text
            st.subheader("Extracted Text")
            st.text_area("PDF Text & OCR", extracted_text, height=400)
        except Exception as e:
            st.error(f"Text/OCR extraction failed: {e}")
        
    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 MMA Corp.")

# Use session state to control the page
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'


# =================================================================================
# --- 1. DASHBOARD PAGE ---
# =================================================================================
def show_dashboard():
    st.title("Dashboard")

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Real-time Sales Volume (Units)",
            value="335",
            delta="5.2% vs last month"
        )
    with col2:
        st.metric(
            label="Real-time Margin (THB)",
            value="฿100,000",
            delta="8.1% vs last month"
        )
    with col3:
        st.metric(
            label="Pending Orders",
            value="3"
        )
    with col4:
        st.metric(
            label="Fleet Utilization",
            value="75%",
            delta="-3% vs yesterday",
            delta_color="inverse"
        )
    
    st.markdown("---")

    # --- Charts ---
    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("Monthly Performance")
        # --- Mock Data for Monthly Performance Chart ---
        perf_data = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
            'Sales Volume': [210, 320, 500, 480, 680, 580, 650],
            'Profit Margin': [3000, 2000, 9800, 5000, 8500, 6000, 7500]
        }
        df_perf = pd.DataFrame(perf_data)

        # --- Create Dual-Axis Line Chart with Plotly ---
        fig_perf = go.Figure()

        # Add Sales Volume trace (Primary Y-axis)
        fig_perf.add_trace(go.Scatter(
            x=df_perf['Month'],
            y=df_perf['Sales Volume'],
            name='Sales Volume',
            mode='lines+markers',
            line=dict(color='#4A90E2'),
            yaxis='y1'
        ))

        # Add Profit Margin trace (Secondary Y-axis)
        fig_perf.add_trace(go.Scatter(
            x=df_perf['Month'],
            y=df_perf['Profit Margin'],
            name='Profit Margin',
            mode='lines+markers',
            line=dict(color='#2ECC71'),
            yaxis='y2'
        ))

        # Update layout for dual axes
        fig_perf.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            yaxis=dict(title='Sales Volume (Units)', color='#4A90E2', range=[0, 800]),
            yaxis2=dict(title='Profit Margin (THB)', overlaying='y', side='right', color='#2ECC71', range=[0, 10000]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with chart2:
        st.subheader("Sales by Top Customers")
        # --- Mock Data for Customer Sales Chart ---
        cust_data = {
            'Customer': ['Central Retail', 'SCG Home', 'Thai Watsadu', 'Global House', 'Boonthavorn'],
            'Volume (Units)': [380, 290, 350, 200, 270]
        }
        df_cust = pd.DataFrame(cust_data).sort_values('Volume (Units)', ascending=True)

        # --- Create Horizontal Bar Chart with Plotly ---
        fig_cust = px.bar(
            df_cust,
            x='Volume (Units)',
            y='Customer',
            orientation='h',
            text='Volume (Units)',
            labels={'Volume (Units)': 'Volume (Units)'}
        )
        fig_cust.update_traces(textposition='inside', marker_color='#4A90E2')
        fig_cust.update_layout(
            yaxis_title=None,
            xaxis_title="Volume (Units)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cust, use_container_width=True)


# =================================================================================
# --- 2. ORDER PROCESSING PAGE ---
# =================================================================================
def show_order_processing():
    st.title("Automated Order Processing")

    with st.container(border=True):
        st.markdown("##### Upload Purchase Order")
        st.markdown("Upload a customer's PO (PDF, PNG, JPG). Our AI will automatically extract the details, reducing manual entry and errors.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Select a PO File", type=['pdf', 'png', 'jpg'], label_visibility="collapsed")
        with col2:
            if st.button("Process PO", use_container_width=True, type="primary"):
                if uploaded_file:
                    with st.spinner('AI is processing the document...'):
                        # Simulate processing
                        import time
                        time.sleep(3)
                    st.success("PO Processed Successfully! Details added to Recent Orders.")
                else:
                    st.warning("Please upload a file first.")

    st.markdown("---")

    st.subheader("Recent Orders")
    
    # --- Mock Data for Recent Orders Table ---
    order_data = {
        'ORDER ID': [f'PO-00{i}' for i in range(1, 7)],
        'CUSTOMER': ['Central Retail', 'SCG Home', 'Thai Watsadu', 'Global House', 'Boonthavorn', 'Central Retail'],
        'VOLUME (UNITS)': [50, 75, 60, 40, 80, 30],
        'MARGIN (THB)': ['15,000', '22,000', '18,000', '11,500', '24,000', '9,500'],
        'DATE': [
            (date.today() - timedelta(days=4)).strftime('%Y-%m-%d'),
            (date.today() - timedelta(days=3)).strftime('%Y-%m-%d'),
            (date.today() - timedelta(days=2)).strftime('%Y-%m-%d'),
            (date.today() - timedelta(days=1)).strftime('%Y-%m-%d'),
            date.today().strftime('%Y-%m-%d'),
            date.today().strftime('%Y-%m-%d')
        ],
        'STATUS': ['Delivered', 'Scheduled', 'Verified', 'Pending', 'Pending', 'Pending']
    }
    df_orders = pd.DataFrame(order_data)
    
    # Displaying the table. Note: Styling individual cells (like the status pills) is limited in st.dataframe.
    # For exact visual match, a custom component or complex HTML injection would be needed.
    st.dataframe(df_orders, use_container_width=True, hide_index=True)


# =================================================================================
# --- 3. LOGISTICS OPTIMIZATION PAGE ---
# =================================================================================
def show_logistics():
    st.title("Logistics Optimization")
    
    col1, col2 = st.columns([3, 1])
    with col1:
      st.markdown("Efficiently plan dispatches using our decision support system. The tool considers truck capacity, order volume, and routes to suggest the most cost-effective delivery schedule, maximizing fleet utilization.")
    with col2:
      st.button("Optimize All Routes", type="primary", use_container_width=True)
      
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Order Queue")
        # --- Mock Data for Order Queue ---
        order_queue = [
            {"id": "PO-003 - Thai Watsadu", "details": "60 units of Cement Type A"},
            {"id": "PO-004 - Global House", "details": "40 units of Dry Mortar"},
            {"id": "PO-005 - Boonthavorn", "details": "80 units of Concrete Mix B"},
            {"id": "PO-006 - Central Retail", "details": "30 units of Cement Type C"},
        ]
        
        for order in order_queue:
            with st.container(border=True):
                c1, c2 = st.columns([3,1])
                with c1:
                    st.markdown(f"**{order['id']}**")
                    st.caption(order['details'])
                with c2:
                    st.button("Assign", key=f"assign_{order['id']}", use_container_width=True)


    with col2:
        st.subheader("Fleet Status")
        # --- Mock Data for Fleet ---
        fleet_status = [
            {"id": "TR-S01 (Small)", "driver": "Somchai", "load": 0, "capacity": 5, "status": "Idle"},
            {"id": "TR-S02 (Small)", "driver": "Somsak", "load": 4, "capacity": 5, "status": "En Route"},
            {"id": "TR-S03 (Small)", "driver": "Somkiat", "load": 0, "capacity": 5, "status": "Idle"},
            {"id": "TR-S04 (Small)", "driver": "Sombat", "load": 0, "capacity": 5, "status": "Idle"},
            {"id": "TR-S05 (Small)", "driver": "Somphong", "load": 0, "capacity": 5, "status": "Maintenance"},
            {"id": "TR-L01 (Large)", "driver": "Prasert", "load": 18, "capacity": 20, "status": "En Route"},
        ]

        status_colors = {
            "Idle": "green",
            "En Route": "blue",
            "Maintenance": "red"
        }

        for i in range(0, len(fleet_status), 2):
            r1, r2 = st.columns(2)
            # Truck 1
            with r1:
                truck = fleet_status[i]
                with st.container(border=True):
                    sc1, sc2 = st.columns([2,1])
                    with sc1:
                        st.markdown(f"**{truck['id']}**")
                    with sc2:
                        st.markdown(f"<span style='color:{status_colors[truck['status']]}'>● {truck['status']}</span>", unsafe_allow_html=True)
                    st.caption(f"Driver: {truck['driver']}")
                    st.progress(truck['load'] / truck['capacity'], text=f"Load: {truck['load']} / {truck['capacity']} units")

            # Truck 2 (if exists)
            if i + 1 < len(fleet_status):
                with r2:
                    truck = fleet_status[i+1]
                    with st.container(border=True):
                        sc1, sc2 = st.columns([2,1])
                        with sc1:
                            st.markdown(f"**{truck['id']}**")
                        with sc2:
                            st.markdown(f"<span style='color:{status_colors[truck['status']]}'>● {truck['status']}</span>", unsafe_allow_html=True)
                        st.caption(f"Driver: {truck['driver']}")
                        st.progress(truck['load'] / truck['capacity'], text=f"Load: {truck['load']} / {truck['capacity']} units")


# =================================================================================
# --- 4. SALES STRATEGY PAGE ---
# =================================================================================
def show_sales_strategy():
    st.title("AI-Powered Sales Strategy")
    
    # Tabs for different strategy tools
    tab1, tab2 = st.tabs(["🎯 Sales Prioritization", "🚚 Delivery Optimization"])
    
    with tab1:
        with st.container(border=True):
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown("##### Sales Prioritization Tool")
                st.markdown("Leverage AI to analyze customer data and answer the critical question: \"Who should we sell to first for maximum profit and strategic advantage?\"")
            with col2:
                st.button("⭐ Get AI Recommendations", type="primary", use_container_width=True)
                
        st.markdown("---")
        
        st.subheader("Top 3 Customer Priorities")
        
        # --- Mock Data for Strategy ---
        priorities = [
            {"rank": 1, "customer": "Central Retail", "reason": "Highest current margin and volume, representing the most significant profit potential.", "margin": 240000},
            {"rank": 2, "customer": "Thai Watsadu", "reason": "Second highest margin and strong volume, indicating consistent profitability and strategic importance.", "margin": 200000},
            {"rank": 3, "customer": "SCG Home", "reason": "Third highest margin and substantial volume, making them a key contributor to overall revenue and profit.", "margin": 180000},
        ]

        cols = st.columns(3)

        for i, priority in enumerate(priorities):
            with cols[i]:
                with st.container(border=True, height=280):
                    # Rank Badge
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h5 style="margin: 0;">{priority['customer']}</h5>
                        <div style="background-color: #FFB347; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                            {priority['rank']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<small>{priority['reason']}</small>", unsafe_allow_html=True)
                    
                    # Use a spacer or specific height in container to align elements at bottom
                    st.markdown("<br>", unsafe_allow_html=True) 

                    st.markdown("**Potential Margin**")
                    st.markdown(f"### ฿{priority['margin']:,}")

        with tab2:
            # ---------- Demo scenario (deterministic mock) ----------
            def generate_demo_scenario(seed=42):
                rng = np.random.default_rng(seed)
                customers = [
                    "Central Retail","Thai Watsadu","SCG Home","Global House","Boonthavorn",
                    "Mega Home","HomePro","DoHome","Amorn","ThaiTool"
                ]
                # volumes ~ [10..80], distances ~ [8..40]
                vols = rng.integers(low=10, high=81, size=len(customers))
                dists = rng.integers(low=8, high=41, size=len(customers))
                return {c: {"volume": int(v), "distance": int(d)}
                    for c, v, d in zip(customers, vols, dists)}

            def default_truck_config():
                return {
                    'small_capacity': 5,     # units
                    'large_capacity': 20,    # units
                    'fuel_cost': 5.0,        # THB/km
                    'small_fixed_cost': 200, # THB
                    'large_fixed_cost': 500  # THB
                }

            if 'optimization_done' not in st.session_state:
                st.session_state.optimization_done = False
            if 'customer_orders' not in st.session_state:
                st.session_state.customer_orders = generate_demo_scenario()
            if 'truck_config' not in st.session_state:
                st.session_state.truck_config = default_truck_config()

            with st.container(border=True):
                st.markdown("##### Smart Delivery Route Optimization")
                st.markdown("Configure or load a demo scenario, then compare **Before** vs **After** optimization.")

                demo_col1, demo_col2 = st.columns([2,1])
                with demo_col1:
                    if st.button("🎲 Load Demo Scenario", use_container_width=True):
                        st.session_state.customer_orders = generate_demo_scenario()
                        st.session_state.truck_config = default_truck_config()
                        st.session_state.optimization_done = False
                        st.rerun()
                with demo_col2:
                    if st.button("♻️ Reset Inputs", use_container_width=True):
                        st.session_state.customer_orders = generate_demo_scenario(123)
                        st.session_state.truck_config = default_truck_config()
                        st.session_state.optimization_done = False
                        st.rerun()

            st.markdown("---")
            st.subheader("Configuration")

            input_col1, input_col2 = st.columns(2)

            with input_col1:
                st.markdown("**Customer Orders & Distances**")
                customer_orders = {}
                for customer, defaults in st.session_state.customer_orders.items():
                    col_vol, col_dist = st.columns(2)
                    with col_vol:
                        volume = st.number_input(
                            f"{customer} Volume",
                            min_value=0, max_value=200, value=int(defaults["volume"]),
                            key=f"vol_{customer}"
                        )
                    with col_dist:
                        distance = st.number_input(
                            f"{customer} Distance (km)",
                            min_value=0, max_value=200, value=int(defaults["distance"]),
                            key=f"dist_{customer}"
                        )
                    customer_orders[customer] = {"volume": volume, "distance": distance}

            with input_col2:
                st.markdown("**Fleet Configuration**")
                small_truck_capacity = st.number_input("Small Truck Capacity (units)", 1, 50, st.session_state.truck_config['small_capacity'])
                large_truck_capacity = st.number_input("Large Truck Capacity (units)", 1, 100, st.session_state.truck_config['large_capacity'])

                st.markdown("**Cost Parameters**")
                fuel_cost_per_km = st.number_input("Fuel Cost per KM (THB)", 0.0, 100.0, float(st.session_state.truck_config['fuel_cost']), step=0.5)
                small_truck_fixed_cost = st.number_input("Small Truck Fixed Cost (THB)", 0, 5000, st.session_state.truck_config['small_fixed_cost'])
                large_truck_fixed_cost = st.number_input("Large Truck Fixed Cost (THB)", 0, 10000, st.session_state.truck_config['large_fixed_cost'])

            run_col1, _ = st.columns([2,1])
            with run_col1:
                if st.button("🚚 Optimize Fleet & Routes", type="primary", use_container_width=True):
                    st.session_state.optimization_done = True
                    st.session_state.customer_orders = customer_orders
                    st.session_state.truck_config = {
                        'small_capacity': small_truck_capacity,
                        'large_capacity': large_truck_capacity,
                        'fuel_cost': fuel_cost_per_km,
                        'small_fixed_cost': small_truck_fixed_cost,
                        'large_fixed_cost': large_truck_fixed_cost
                    }
                    st.rerun()

            st.markdown("---")

            # ---------- Planning helpers ----------
            def summarize(routes):
                if not routes:
                    return 0, 0, 0, 0, 0
                total_trucks = len([r for r in routes if r['customers']])
                total_cost = sum(r['total_cost'] for r in routes)
                total_distance = sum(r['total_distance'] for r in routes)
                total_volume = sum(r['total_volume'] for r in routes)
                avg_eff = np.mean([r['efficiency'] for r in routes if r['efficiency'] > 0])
                return total_trucks, total_cost, total_distance, total_volume, avg_eff

            # Greedy bin-pack style assignment (used by optimizer)
            def assign_customers_to_trucks(customers, num_large, num_small, config):
                trucks = []
                for i in range(num_large):
                    trucks.append({'id': f'Large-{i+1}','type':'Large','capacity': config['large_capacity'],
                        'fixed_cost': config['large_fixed_cost'],'customers':[],'vol':0,'dist':0})
                for i in range(num_small):
                    trucks.append({'id': f'Small-{i+1}','type':'Small','capacity': config['small_capacity'],
                        'fixed_cost': config['small_fixed_cost'],'customers':[],'vol':0,'dist':0})

                # Sort customers by volume/distance ratio for better packing
                customers_sorted = sorted(customers, key=lambda x: x[1]['volume'] / max(x[1]['distance'], 1), reverse=True)
                
                assigned = []
                for name, data in customers_sorted:
                    best = None
                    best_fit = float('inf')
                    
                    for t in trucks:
                        if t['vol'] + data['volume'] <= t['capacity']:
                            # Find truck with best fit (smallest remaining capacity after assignment)
                            remaining_capacity = t['capacity'] - (t['vol'] + data['volume'])
                            if remaining_capacity < best_fit:
                                best_fit = remaining_capacity
                                best = t
                    
                    if best:
                        best['customers'].append(name)
                        best['vol'] += data['volume']
                        best['dist'] += data['distance']
                        assigned.append((name, data))

                unassigned = [n for (n, d) in customers if (n, d) not in assigned]
                routes = []
                for t in trucks:
                    if t['customers']:
                        fuel = t['dist'] * config['fuel_cost']
                        total = t['fixed_cost'] + fuel
                        eff = (t['vol'] / t['capacity']) * 100
                        routes.append({
                            'truck_id': t['id'],
                            'truck_type': 'Large Truck' if t['type']=='Large' else 'Small Truck',
                            'customers': t['customers'],
                            'total_volume': t['vol'],
                            'capacity': t['capacity'],
                            'total_distance': t['dist'],
                            'fuel_cost': fuel,
                            'fixed_cost': t['fixed_cost'],
                            'total_cost': total,
                            'efficiency': eff
                        })
                return routes, unassigned

            # Optimizer: tries combinations and picks lowest cost
            def optimize_fleet_and_routes(customer_orders, config):
                active = [(n, d) for n, d in customer_orders.items() if d['volume'] > 0]
                if not active:
                    return [], [], 0, 0

                total_vol = sum(d['volume'] for _, d in active)
                max_trucks = len(active)

                best = None
                best_cost = float('inf')
                
                # Try different combinations of trucks
                for nl in range(0, min(8, max_trucks) + 1):
                    for ns in range(0, min(12, max_trucks) + 1):
                        if nl == 0 and ns == 0:
                            continue
                        
                        capacity = nl * config['large_capacity'] + ns * config['small_capacity']
                        if capacity < total_vol:
                            continue
                        
                        routes, unassigned = assign_customers_to_trucks(active, nl, ns, config)
                        
                        # Calculate cost with penalty for unassigned customers
                        cost = sum(r['total_cost'] for r in routes)
                        if unassigned:
                            cost += len(unassigned) * 10000  # Heavy penalty for unassigned
                        
                        if cost < best_cost:
                            best_cost = cost
                            best = (routes, unassigned)
                
                if best is None:
                    # Fallback: use enough large trucks to handle all volume
                    min_large_trucks = (total_vol + config['large_capacity'] - 1) // config['large_capacity']
                    routes, unassigned = assign_customers_to_trucks(active, min_large_trucks, 0, config)
                    best = (routes, unassigned)
                    
                routes, unassigned = best
                t, c, _, _, _ = summarize(routes)
                return routes, unassigned, c, t

            # Baseline: simple/naive plan (Before)
            def baseline_plan(customer_orders, config):
                routes = []
                truck_counter = 1
                
                for name, data in customer_orders.items():
                    vol = data['volume']
                    dist = data['distance']
                    if vol <= 0: 
                        continue

                    remaining = vol
                    customer_truck_num = 1
                    
                    while remaining > 0:
                        if remaining <= config['small_capacity']:
                            cap = config['small_capacity']
                            t_id = f"Small-{truck_counter}"
                            t_type = 'Small Truck'
                            fixed = config['small_fixed_cost']
                        else:
                            cap = config['large_capacity']
                            t_id = f"Large-{truck_counter}"
                            t_type = 'Large Truck'
                            fixed = config['large_fixed_cost']

                        use_vol = min(remaining, cap)
                        fuel = dist * config['fuel_cost']
                        total = fixed + fuel
                        eff = (use_vol / cap) * 100

                        routes.append({
                            'truck_id': t_id,
                            'truck_type': t_type,
                            'customers': [name],
                            'total_volume': use_vol,
                            'capacity': cap,
                            'total_distance': dist,
                            'fuel_cost': fuel,
                            'fixed_cost': fixed,
                            'total_cost': total,
                            'efficiency': eff
                        })
                        
                        remaining -= use_vol
                        truck_counter += 1
                        customer_truck_num += 1
                        
                return routes

            # Define variables from session state
            orders = st.session_state.customer_orders
            cfg = st.session_state.truck_config

            # Always compute baseline
            routes_before = baseline_plan(orders, cfg)
            b_trucks, b_cost, b_dist, _, b_eff = summarize(routes_before)

            if not st.session_state.optimization_done:
                st.subheader("Current Configuration Preview")
                total_volume = sum(d['volume'] for d in orders.values() if d['volume'] > 0)
                total_distance = sum(d['distance'] for d in orders.values() if d['volume'] > 0)

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Total Volume", f"{total_volume} units")
                with c2: st.metric("Total Distance", f"{total_distance} km")
                with c3: st.metric("Active Customers", f"{len([d for d in orders.values() if d['volume'] > 0])}")
                st.info("Click **Optimize Fleet & Routes** to calculate the optimized plan and compare with the baseline.")
            else:
                # Compute optimized results
                routes_after, unassigned, _, _ = optimize_fleet_and_routes(orders, cfg)
                a_trucks, a_cost, a_dist, _, a_eff = summarize(routes_after)

                st.subheader("Before vs After (Key Metrics)")
                k1, k2, k3, k4 = st.columns(4)
                with k1: 
                    delta_trucks = a_trucks - b_trucks
                    st.metric("Trucks Used", f"{a_trucks}", delta=f"{delta_trucks:+d}" if delta_trucks != 0 else None)
                with k2: 
                    delta_cost = a_cost - b_cost
                    st.metric("Total Cost (THB)", f"฿{a_cost:,.0f}", delta=f"฿{delta_cost:+,.0f}" if delta_cost != 0 else None)
                with k3: 
                    delta_eff = a_eff - b_eff
                    st.metric("Avg Utilization", f"{a_eff:.1f}%", delta=f"{delta_eff:+.1f}%" if abs(delta_eff) > 0.1 else None)
                with k4: 
                    delta_dist = a_dist - b_dist
                    st.metric("Total Distance", f"{a_dist:.0f} km", delta=f"{delta_dist:+.0f}" if abs(delta_dist) > 0.1 else None)

                st.markdown("---")
                st.subheader("Plan Details (Before vs After)")
                
                # Build comparison table
                def to_rows(tag, routes):
                    rows = []
                    for r in routes:
                        rows.append({
                            "Plan": tag,
                            "Truck": r['truck_id'],
                            "Type": r['truck_type'],
                            "Customers": ", ".join(r['customers']),
                            "Volume/Cap": f"{r['total_volume']}/{r['capacity']}",
                            "Distance (km)": round(r['total_distance'], 1),
                            "Fixed (THB)": round(r['fixed_cost'], 0),
                            "Fuel (THB)": round(r['fuel_cost'], 0),
                            "Total (THB)": round(r['total_cost'], 0),
                            "Utilization %": round(r['efficiency'], 1),
                        })
                    return rows

                df_cmp = pd.DataFrame(to_rows("Before", routes_before) + to_rows("After", routes_after))
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

                if unassigned:
                    st.warning(f"⚠️ Unassigned customers: {', '.join(unassigned)}. Consider increasing fleet capacity.")

                # Summary comparison
                st.markdown("---")
                st.subheader("Summary Comparison")
                summary_data = {
                    "Metric": ["Total Trucks", "Total Cost (THB)", "Total Distance (km)", "Avg Utilization (%)"],
                    "Before": [b_trucks, f"฿{b_cost:,.0f}", f"{b_dist:.0f}", f"{b_eff:.1f}%"],
                    "After": [a_trucks, f"฿{a_cost:,.0f}", f"{a_dist:.0f}", f"{a_eff:.1f}%"],
                    "Savings": [
                        f"{b_trucks - a_trucks:+d}",
                        f"฿{b_cost - a_cost:+,.0f}",
                        f"{b_dist - a_dist:+.0f} km",
                        f"{a_eff - b_eff:+.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Visual Comparison")
                
                # Create comparison charts
                cA, cB = st.columns(2)
                with cA:
                    st.markdown("**Utilization Comparison**")
                    if routes_before and routes_after:
                        util_df = pd.DataFrame({
                            "Truck": [r['truck_id'] for r in routes_before] + [r['truck_id'] for r in routes_after],
                            "Utilization %": [r['efficiency'] for r in routes_before] + [r['efficiency'] for r in routes_after],
                            "Plan": (["Before"] * len(routes_before)) + (["After"] * len(routes_after))
                        })
                        fig_u = px.bar(util_df, x="Truck", y="Utilization %", color="Plan", barmode="group",
                                 color_discrete_map={"Before": "#ff6b6b", "After": "#4ecdc4"})
                        fig_u.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_tickangle=-45)
                        st.plotly_chart(fig_u, use_container_width=True)
                        
                with cB:
                    st.markdown("**Cost Comparison**")
                    if routes_before and routes_after:
                        cost_df = pd.DataFrame({
                            "Truck": [r['truck_id'] for r in routes_before] + [r['truck_id'] for r in routes_after],
                            "Total Cost (THB)": [r['total_cost'] for r in routes_before] + [r['total_cost'] for r in routes_after],
                            "Plan": (["Before"] * len(routes_before)) + (["After"] * len(routes_after))
                        })
                        fig_c = px.bar(cost_df, x="Truck", y="Total Cost (THB)", color="Plan", barmode="group",
                                 color_discrete_map={"Before": "#ff6b6b", "After": "#4ecdc4"})
                        fig_c.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_tickangle=-45)
                        st.plotly_chart(fig_c, use_container_width=True)

# =================================================================================
# --- Logistic Route optimization ---
# =================================================================================
# 🚚 Truck fleet setup
TRUCK_FLEET = {
    "lorry_14t": {"capacity": 14, "count": 5},
    "lorry_24t": {"capacity": 24, "count": 7},
    "lorry_29t": {"capacity": 29, "count": 2},
}

# 📍 Provinces with multiple trips/day
MULTI_TRIP_PROVINCES = {
    "Rayong": 2
}

def logistics_optimization_page():
    st.title("🚛 Logistics Optimization for Polymers")

    # Ensure state variables exist
    if "all_orders" not in st.session_state or not isinstance(st.session_state.all_orders, list):
        st.session_state.all_orders = []  # store every order
    if "schedule" not in st.session_state or not isinstance(st.session_state.schedule, dict):
        st.session_state.schedule = {}
    if "moved_orders" not in st.session_state or not isinstance(st.session_state.moved_orders, list):
        st.session_state.moved_orders = []

    # ===============================
    # 🗑️ CLEAR OPTIONS
    # ===============================
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Clear All Orders"):
            st.session_state.all_orders = []
            st.session_state.schedule = {}
            st.session_state.moved_orders = []
            st.success("✅ All orders have been cleared.")

    with col2:
        clear_date = st.date_input("Select date to clear bookings", value=date.today())
        if st.button("❌ Clear Selected Date Orders"):
            st.session_state.all_orders = [
                o for o in st.session_state.all_orders if o["desired"] != clear_date
            ]
            run_optimization()
            st.success(f"✅ Cleared all orders on {clear_date}")

    # ===============================
    # 📋 NEW ORDER FORM
    # ===============================
    st.subheader("📋 Enter New Order")
    with st.form("order_form"):
        customer = st.text_input("Customer Company Name")
        location = st.text_input("Delivery Location (Province)")
        qty = st.number_input("Quantity of Polymers (tons)", min_value=0.1, step=0.1)
        desired_date = st.date_input("Earliest / Exact Delivery Date", min_value=date.today())
        max_date = st.date_input(
            "Latest Acceptable Delivery Date",
            value=desired_date, min_value=desired_date
        )
        submit = st.form_submit_button("Add Order")

    if submit:
        truck_type = auto_select_truck(qty)
        order = {
            "customer": customer,
            "location": location,
            "qty": qty,
            "truck_type": truck_type,
            "desired": desired_date,
            "max": max_date
        }
        st.session_state.all_orders.append(order)
        run_optimization()
        st.success(f"✅ Order for {customer} added using **{truck_type}**")

    # ===============================
    # 📅 SHOW SCHEDULE
    # ===============================
    st.subheader("📅 Allocated Schedule (Optimized)")
    schedule_df = get_schedule_dataframe_v2()
    st.dataframe(schedule_df)

    # ===============================
    # 💡 RECOMMENDATIONS
    # ===============================
    recommendations = recommend_moves()
    if recommendations:
        st.subheader("💡 Recommendations to Resolve Overbookings")
        for r in recommendations:
            if r["to_date"]:
                st.write(f"📢 Move **{r['customer']}** from {r['from_date']} → {r['to_date']}")
            else:
                st.write(f"⚠️ Could not schedule **{r['customer']}** (needs manual decision)")
    
    # ===============================
    # 📦 FINAL DELIVERY PLAN SUMMARY
    # ===============================
    st.subheader("📦 Final Delivery Plan After Optimization")
    summary_df = get_final_delivery_summary()
    if not summary_df.empty:
        st.dataframe(summary_df)
    else:
        st.info("No deliveries scheduled yet.")

# -------------------------------------
# HELPER FUNCTIONS
# -------------------------------------

def auto_select_truck(qty):
    """Pick smallest truck type that can handle qty."""
    if qty <= TRUCK_FLEET["lorry_14t"]["capacity"]:
        return "lorry_14t"
    elif qty <= TRUCK_FLEET["lorry_24t"]["capacity"]:
        return "lorry_24t"
    else:
        return "lorry_29t"

def run_optimization():
    """Greedy optimization to assign orders without exceeding truck limits."""
    st.session_state.schedule = {}
    moved_orders = []

    # Sort by least flexibility first (keep least flexible on desired date if possible)
    orders_sorted = sorted(
        st.session_state.all_orders,
        key=lambda o: (o["max"] - o["desired"]).days
    )

    for order in orders_sorted:
        if not isinstance(order, dict) or "truck_type" not in order:
            continue

        trip_multiplier = MULTI_TRIP_PROVINCES.get(order["location"].title().strip(), 1)
        assigned = False

        for d in pd.date_range(order["desired"], order["max"]):
            used = sum(1 for o in st.session_state.schedule.get(d.date(), [])
                       if isinstance(o, dict) and o.get("truck_type") == order["truck_type"])
            total_can_use = TRUCK_FLEET[order["truck_type"]]["count"] * trip_multiplier

            if used < total_can_use:
                # Assign this order to schedule
                if d.date() not in st.session_state.schedule:
                    st.session_state.schedule[d.date()] = []
                st.session_state.schedule[d.date()].append(order.copy())
                if d.date() != order["desired"]:
                    moved_orders.append({
                        "customer": order["customer"],
                        "from_date": order["desired"],
                        "to_date": d.date()
                    })
                assigned = True
                break

        if not assigned:
            moved_orders.append({
                "customer": order["customer"],
                "from_date": order["desired"],
                "to_date": None  # Could not schedule within window
            })

    st.session_state.moved_orders = moved_orders

def get_schedule_dataframe_v2():
    """Return DataFrame showing truck usage and availability."""
    data = []
    for offset in range(0, 30):
        day = date.today() + timedelta(days=offset)
        row = {"Date": day}
        for t in TRUCK_FLEET.keys():
            used = sum(1 for o in st.session_state.schedule.get(day, [])
                       if isinstance(o, dict) and o.get("truck_type") == t)
            total = TRUCK_FLEET[t]["count"]
            row[f"{t}_used"] = used
            row[f"{t}_available"] = total - used
        data.append(row)
    return pd.DataFrame(data)

def get_final_delivery_summary():
    """Return DataFrame of all deliveries after optimization."""
    rows = []
    for day, orders in sorted(st.session_state.schedule.items()):
        for order in orders:
            rows.append({
                "Final Delivery Date": day,
                "Customer": order.get("customer"),
                "Location": order.get("location"),
                "Quantity (t)": order.get("qty"),
                "Truck Type": order.get("truck_type"),
                "Original Desired Date": order.get("desired"),
            })
    return pd.DataFrame(rows)

def recommend_moves():
    """Show the move recommendations stored from last optimization."""
    return st.session_state.get("moved_orders", [])

recommendations = recommend_moves()
if recommendations:
    st.subheader("💡 Recommendations to Resolve Overbookings")
    for r in recommendations:
        if r["to_date"]:
            st.write(f"📢 Move **{r['customer']}** from {r['from_date']} → {r['to_date']}")
        else:
            st.write(f"⚠️ Could not schedule **{r['customer']}** (needs manual decision)")

# =================================================================================
# --- Truck Management Page ---
# =================================================================================
def show_truck_management():
    import calendar
    from datetime import datetime
    # --- Truck resources ---
    TRUCKS = {
        "Lorry A": {"capacity": 14, "count": 5},
        "Lorry B": {"capacity": 24, "count": 7},
        "Lorry C": {"capacity": 29, "count": 2},
    }
    CAPACITY_TO_TRUCK = {14: "Lorry A", 24: "Lorry B", 29: "Lorry C"}
    TRUCK_TYPES = ["Lorry A", "Lorry B", "Lorry C"]

    st.set_page_config(page_title="Lorry Truck Scheduler", layout="wide")
    st.title("🚚 Lorry Truck Optimization Scheduler")

    # --- Month selection (default = current year/month) ---
    today = datetime.today()
    months = {calendar.month_name[i]: i for i in range(1, 13)}
    col_m1, col_m2 = st.columns([2,1])
    with col_m1:
        selected_month = st.selectbox("Select Month", options=list(months.keys()), index=today.month - 1)
    with col_m2:
        selected_year = st.number_input("Year", min_value=2000, max_value=2100, value=today.year, step=1)

    month_num = months[selected_month]
    days_in_month = calendar.monthrange(selected_year, month_num)[1]

    # --- Orders memory ---
    if "orders" not in st.session_state:
        # each order: {day:int, load:int, customer:str}
        st.session_state["orders"] = []

    st.sidebar.subheader("Actions")
    if st.sidebar.button("🧹 Clear all orders"):
        st.session_state["orders"] = []
        st.sidebar.success("All orders cleared for this session.")

    # --- Input form (fixed loads + customer) ---
    st.subheader("Add Required Loads")
    st.markdown("Allowed loads: **14, 24, 29 tons**. Please specify customer/destination.")

    with st.form("add_order"):
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            day = st.number_input("Day of Month", min_value=1, max_value=days_in_month, value=1)
        with c2:
            load = st.selectbox("Required Load (tons)", options=[14, 24, 29], index=0)
        with c3:
            customer = st.text_input("Customer / Destination", placeholder="e.g., ACME Co. (Bangna)")

        submitted = st.form_submit_button("Add Order")

        if submitted:
            customer = customer.strip()
            if not customer:
                st.warning("Please enter a customer/destination.")
            else:
                truck_type = CAPACITY_TO_TRUCK[load]
                max_available = TRUCKS[truck_type]["count"]

                # Count existing orders for this truck type on that day
                existing = sum(
                    1 for o in st.session_state["orders"]
                    if o["day"] == day and CAPACITY_TO_TRUCK[o["load"]] == truck_type
                )

                if existing + 1 > max_available:
                    st.warning(
                        f"❌ Cannot add order: {truck_type} on day {day} would exceed daily availability "
                        f"({max_available} trucks)."
                    )
                else:
                    st.session_state["orders"].append({"day": day, "load": load, "customer": customer})
                    st.success(f"✅ Added {load}T on day {day} ({truck_type}) → {customer}")

    # --- Orders table (raw) ---
    orders_df = pd.DataFrame(st.session_state["orders"])
    if not orders_df.empty and len(st.session_state["orders"]) > 0:
        st.subheader("Orders")
        st.dataframe(orders_df.sort_values(["day", "load"]), use_container_width=True, hide_index=True)

    # --- Build per-day, per-truck-type list of customers ---
    def group_customers_by_day_and_type(orders, days_in_month):
        # dict: day -> truck_type -> [customers...]
        grouped = {d: {t: [] for t in TRUCK_TYPES} for d in range(1, days_in_month + 1)}
        for o in sorted(orders, key=lambda x: (x["day"], x["load"], x["customer"].lower())):
            t = CAPACITY_TO_TRUCK[o["load"]]
            grouped[o["day"]][t].append(o["customer"])
        return grouped

    grouped = group_customers_by_day_and_type(st.session_state["orders"], days_in_month)

    # --- Helper: render sockets line like "🔴 CustA | 🔴 CustB | ⚪ | ⚪ ..." ---
    def sockets_line(used_customers, capacity):
        used = [f"🔴 {c}" for c in used_customers]
        avail_count = max(capacity - len(used_customers), 0)
        avail = ["⚪"] * avail_count
        return " | ".join(used + avail) if capacity > 0 else ""

    # --- Build visualization table dataframe ---
    viz_rows = []
    for day in range(1, days_in_month + 1):
        row = {"Day": day}
        for t in TRUCK_TYPES:
            cap = TRUCKS[t]["count"]
            custs = grouped[day][t]
            row[f"{t} ({TRUCKS[t]['capacity']}T)"] = sockets_line(custs, cap)
        viz_rows.append(row)

    viz_df = pd.DataFrame(viz_rows)

    st.subheader("Schedule (Slots Table)")
    st.caption("⚪ = available slot, 🔴 = booked slot (shows customer)")

    # ---- Show FULL table, hide the left index (keep Day column) ----
    styled = (
        viz_df
        .style
        .hide(axis="index")  # hides the left unnamed index column
        .set_properties(**{"white-space": "pre-wrap", "line-height": "1.3"})
    )

    st.table(styled)  # st.table renders the full table (no scroll)

# =================================================================================
# --- Main App Logic ---
# =================================================================================
if st.session_state.page == "Dashboard":
    show_dashboard()
elif st.session_state.page == "Order Processing":
    show_order_processing()
elif st.session_state.page == "Logistics":
    show_logistics()
elif st.session_state.page == "Sales Strategy":
    show_sales_strategy()
elif st.session_state.page == "Truck Management":
    show_truck_management()
elif st.session_state.page == "OCR":
    show_ocr()
elif st.session_state.page == "Logistics Optimization":
    logistics_optimization_page()