import streamlit as st
import pandas as pd
import subprocess
import os

# Configure the Streamlit page
st.set_page_config(page_title="Indian Economy ABM Dashboard", layout="wide")
st.title("🇮🇳 Indian Economy ABM: Policy Laboratory")
st.markdown("Inject macroeconomic shocks and monitor the distributional impact across sectors and the wealth gap.")

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
RESULTS_PATH = os.path.join(ROOT_DIR, "data", "processed", "simulation_results.csv")
SIMULATION_SCRIPT = os.path.join(ROOT_DIR, "src", "engine", "run_simulation.py")

# Sidebar Controls & Policy Shocks
st.sidebar.header("Policy Shock Laboratory")
st.sidebar.markdown("Inject shocks to see how the economy reacts.")

repo_shock = st.sidebar.slider("RBI Repo Rate Shock", min_value=-0.05, max_value=0.10, value=0.0, step=0.005, format="%.3f", help="Additive shock. e.g. 0.02 = +200 bps hike")
gst_shock = st.sidebar.slider("GST Rate Shock", min_value=-0.10, max_value=0.10, value=0.0, step=0.01, format="%.2f", help="Additive shock. e.g. -0.05 = -5% tax cut")
exchange_shock = st.sidebar.slider("Exchange Rate Shock (INR)", min_value=-0.20, max_value=0.20, value=0.0, step=0.01, format="%.2f", help="Additive shock. > 0 means INR depreciates, boosting exports.")

st.sidebar.markdown("---")

# Function to run the simulation with args
def run_simulation(repo, gst, exchange):
    with st.spinner('Running 10-year ABM Simulation with Policy Shocks...'):
        try:
            result = subprocess.run([
                'python', SIMULATION_SCRIPT,
                '--repo_shock', str(repo),
                '--gst_shock', str(gst),
                '--exchange_shock', str(exchange)
            ], capture_output=True, text=True, cwd=ROOT_DIR)
            
            if result.returncode == 0:
                st.sidebar.success("Simulation completed successfully!")
            else:
                st.sidebar.error(f"Simulation failed: {result.stderr}")
        except Exception as e:
            st.sidebar.error(f"Error executing simulation: {str(e)}")

if st.sidebar.button("Inject Shocks & Re-run Simulation", type="primary"):
    run_simulation(repo_shock, gst_shock, exchange_shock)

# Load Data
@st.cache_data(ttl=5)
def load_data():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None

df = load_data()

if df is not None and not df.empty:
    st.subheader("Final Tick Macroeconomic KPIs")
    
    initial_row = df.iloc[0]
    final_row = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Nominal Output (Crores)", 
                  value=f"₹{final_row['Total_Output']:,.0f}", 
                  delta=f"{((final_row['Total_Output'] - initial_row['Total_Output']) / initial_row['Total_Output']) * 100:.1f}%")
    with col2:
        st.metric(label="Total Net Profit", 
                  value=f"₹{final_row['Total_Profit']:,.0f}")
    with col3:
        st.metric(label="Total Tax Revenue", 
                  value=f"₹{final_row['Total_Tax_Revenue']:,.0f}", 
                  delta="Govt Income")
    with col4:
        st.metric(label="Bankruptcies (Total)", 
                  value=f"{final_row['Bankruptcies']:,.0f} firms", 
                  delta_color="inverse")

    st.markdown("---")
    
    # Create Tabs for different analytical views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Macroeconomy", "🏭 Sectoral Divergence", "⚖️ Inequality & Bankruptcies", "🗺️ Regional Economy", "🔗 Supply Chain Contagion"])
    
    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Nominal Output & Tax Revenue")
            chart_data_op = df.set_index("Tick")[["Total_Output", "Total_Tax_Revenue"]]
            st.line_chart(chart_data_op)
            
        with colB:
            st.subheader("Corporate Debt vs Profit Trajectory")
            chart_data_cl = df.set_index("Tick")[["Total_Debt", "Total_Profit"]]
            st.line_chart(chart_data_cl)
            
    with tab2:
        colC, colD = st.columns(2)
        with colC:
            st.subheader("Output by Sector")
            chart_data_sect_out = df.set_index("Tick")[["Agri_Output", "Mfg_Output", "Svc_Output"]]
            st.line_chart(chart_data_sect_out)
        with colD:
            st.subheader("Profit by Sector")
            chart_data_sect_prof = df.set_index("Tick")[["Agri_Profit", "Mfg_Profit", "Svc_Profit"]]
            st.line_chart(chart_data_sect_prof)
            
    with tab3:
        st.subheader("Wealth Inequality (Gini Coefficient of Capital)")
        st.markdown("A value of 0 means perfect equality. A value closer to 1 means extreme inequality (few mega-corporations own all capital).")
        chart_data_gini = df.set_index("Tick")[["Gini_Coefficient"]]
        st.line_chart(chart_data_gini, color="#ff4b4b")
        
    with tab4:
        st.subheader("State GDP Leaderboard (Final Year)")
        st.markdown("Comparing Nominal Output across Indian States.")
        
        # Filter columns that start with 'State_'
        state_cols = [c for c in df.columns if c.startswith('State_')]
        if state_cols:
            # Get the values for the final tick
            state_data = final_row[state_cols].sort_values(ascending=False)
            
            # Show top 10 states
            top_states = state_data.head(10)
            
            # Clean up index names for chart (remove 'State_' prefix)
            top_states.index = top_states.index.str.replace('State_', '')
            
            st.bar_chart(top_states, horizontal=True)
            
            with st.expander("View All States GDP"):
                all_states = state_data.copy()
                all_states.index = all_states.index.str.replace('State_', '')
                st.dataframe(all_states.to_frame(name="Nominal Output (Crores)").style.format("{:,.0f}"))
        else:
            st.warning("Regional data not found in simulation results.")
            
    with tab5:
        st.subheader("Supply Chain Scarcity (Intermediate Input Costs)")
        st.markdown("Tracks the cost multiplier for procuring intermediate goods. A value > 1.0 means the sector experienced bankruptcies/shortages, driving up prices for downstream firms.")
        
        # Check if price multipliers exist
        sc_cols = ["Agri_Price_Multiplier", "Mfg_Price_Multiplier", "Svc_Price_Multiplier"]
        available_cols = [c for c in sc_cols if c in df.columns]
        
        if available_cols:
            chart_data_sc = df.set_index("Tick")[available_cols]
            st.line_chart(chart_data_sc)
        else:
            st.warning("Supply Chain pricing data not found in simulation results.")
            
    with st.expander("View Raw Data Table"):
        st.dataframe(df.style.format("{:,.3f}"))
else:
    st.warning("No simulation results found. Please set policy parameters and click 'Inject Shocks & Re-run Simulation' in the sidebar.")
