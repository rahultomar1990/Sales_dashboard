import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import os

# --- 1. PAGE SETUP & BRANDING ---
st.set_page_config(layout="wide", page_title="Bikano Sales Intelligence", page_icon="📈")

# Try to load logo (make sure 'bikano_logo.png' is in the same directory)
LOGO_PATH = "bikano_logo.png"
try:
    # st.logo is a newer feature for top-left branding
    st.logo(LOGO_PATH, icon_image=LOGO_PATH)
except Exception:
    pass # Continue if logo is missing

# --- 2. EXECUTIVE STYLING (GLASSMORPHISM & POLISH) ---
st.markdown("""
    <style>
        /* APP BACKGROUND - Subtle Gradient for depth */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.5);
        }

        /* MAIN Title Styling */
        h1 { 
            text-align: left;
            background: -webkit-linear-gradient(45deg, #d32f2f, #fbc02d); /* Bikano Red/Yellow hues */
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.2rem; font-weight: 900; margin-bottom: 15px;
        }

        /* GLASSMORPHISM METRIC CARDS */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.65); /* Translucent white */
            backdrop-filter: blur(12px); /* The frosted glass effect */
            border: 1px solid rgba(255, 255, 255, 0.4); /* Subtle border */
            padding: 20px !important;
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10); /* Soft shadow */
            transition: all 0.3s ease;
            height: 100%;
        }
        div[data-testid="metric-container"]:hover {
             transform: translateY(-5px);
             box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
             border-color: rgba(251, 192, 45, 0.8); /* Yellow glow on hover */
        }

        /* METRIC VALUES */
        div[data-testid="stMetricValue"] {
            font-size: 26px; font-weight: 800;
            color: #333;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 600; color: #555;
        }

        /* CHAT AREA STYLING */
        .stChatInput {
            border-radius: 20px !important;
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        /* CUSTOM BUTTONS FOR AI */
        .stButton button {
            background: linear-gradient(to right, #d32f2f, #e57373);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton button:hover {
             background: linear-gradient(to right, #b71c1c, #d32f2f);
             transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. CONFIGURATION ---
GROQ_API_KEY = "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"
CACHE_FILE = "sales_data_cache.parquet"

# --- 4. DATA ENGINE ---
@st.cache_data(show_spinner=False)
def get_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_parquet(CACHE_FILE)
            if 'Order Date' in df.columns:
                df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
                df['Year'] = df['Order Date'].dt.year
                df['Month'] = df['Order Date'].dt.month_name()
            
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            if 'Month' in df.columns:
                df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
            
            df = df.sort_values(by='Order Date')
            return df
        except Exception as e:
            return f"Error reading cache: {str(e)}"
    else:
        return "⚠️ CRITICAL: 'sales_data_cache.parquet' not found!"

# --- LOAD DATA ---
with st.spinner("🚀 Launching Bikano Analytics..."):
    data_result = get_data()

if isinstance(data_result, str):
    st.error(data_result)
    st.stop()
else:
    df = data_result

# --- 5. CHART ENGINE ---
def generate_chart(prompt, df):
    try:
        chart_llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        code_prompt = f"""
        You are a Python Data Visualization Expert using Plotly Express.
        The user wants a visual based on this dataframe: {df.columns.tolist()}
        User Query: "{prompt}"
        INSTRUCTIONS:
        1. INTELLIGENTLY CHOOSE CHART TYPE.
        2. STYLE: Use `template='plotly_white'`. Use BOLD colors.
        3. OUTPUT: Write ONLY the raw Python code for figure named 'fig'. Use 'df'.
        """
        response = chart_llm.invoke(code_prompt)
        cleaned_code = response.content.replace("```python", "").replace("```", "").strip()
        local_vars = {"df": df, "px": px}
        exec(cleaned_code, globals(), local_vars)
        return local_vars.get('fig', None)
    except Exception:
        return None

# --- 6. CALCULATIONS ---
min_year = int(df['Year'].min()) if 'Year' in df.columns else 2020
max_year = int(df['Year'].max()) if 'Year' in df.columns else 2024
current_year = max_year
last_year = current_year - 1
cy_sales = df[df['Year'] == current_year]['Sales'].sum() if 'Year' in df.columns else 0
ly_sales = df[df['Year'] == last_year]['Sales'].sum() if 'Year' in df.columns else 0
yoy_growth = ((cy_sales - ly_sales) / ly_sales) * 100 if ly_sales > 0 else 0
orders_count = df['Order ID'].nunique() if 'Order ID' in df.columns else 0
units_count = df['Units'].sum() if 'Units' in df.columns else 0

# --- 7. MAIN LAYOUT (50/50 Split) ---
col_dash, col_chat = st.columns([1, 1], gap="large")

# ================= LEFT COLUMN: DASHBOARD =================
with col_dash:
    # Header with Bikano theme colors
    st.markdown(f"<h1>Bikano Executive Overview <span style='font-size:1.2rem; color:#555'>| FY {current_year}</span></h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # GLASSMORPHISM TILES (3x2 Grid)
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1: st.metric("💰 Total Revenue", f"${df['Sales'].sum():,.0f}")
    with row1_c2: st.metric(f"📅 CY {current_year} Sales", f"${cy_sales:,.0f}")
    with row1_c3: st.metric(f"⏮️ LY {last_year} Sales", f"${ly_sales:,.0f}", delta=f"${cy_sales-ly_sales:,.0f}")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True) # Spacer

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1: st.metric("📈 YoY Growth", f"{yoy_growth:.2f}%", delta=f"{yoy_growth:.2f}%")
    with row2_c2: st.metric("📦 Total Orders", f"{orders_count:,}")
    with row2_c3: st.metric("🔢 Units Sold", f"{units_count:,.0f}")
    
    st.markdown("---")
    
    # TREND CHART (Clean Line Chart)
    if 'Year' in df.columns:
        st.subheader(f"📅 Sales Performance Trend ({last_year} vs {current_year})")
        trend_df = df[df['Year'].isin([current_year, last_year])]
        monthly_sales = trend_df.groupby(['Month', 'Year'])['Sales'].sum().reset_index()
        
        # Using a bold red/gold palette for Bikano
        custom_colors = ['#d32f2f', '#fbc02d'] 
        fig_line = px.line(monthly_sales, x='Month', y='Sales', color='Year', markers=True,
                           color_discrete_sequence=custom_colors)
        
        fig_line.update_layout(xaxis_title=None, yaxis_title="Revenue ($)", legend_title="Fiscal Year", 
                               height=400, template="plotly_white",
                               hovermode="x unified")
        fig_line.update_traces(line=dict(width=4), marker=dict(size=10, line=dict(width=2, color='white')))
        st.plotly_chart(fig_line, use_container_width=True)

# ================= RIGHT COLUMN: AI ANALYST =================
with col_chat:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    # Styled header for AI
    st.markdown("<h3 style='color:#d32f2f;'>🤖 Bikano AI Data Analyst</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;'>Instant insights from your sales data.</p>", unsafe_allow_html=True)

    # --- NEW: EXECUTIVE QUICK ACTION BUTTONS ---
    # These buttons allow one-click querying
    qa_c1, qa_c2, qa_c3 = st.columns(3)
    prompt_to_run = None
    
    if qa_c1.button("🔥 Top 5 Products", use_container_width=True, help="Click to see best sellers CY"):
        prompt_to_run = f"Show me a table of the top 5 Products by Sales in {current_year}."
    if qa_c2.button("🏙️ Region Performance", use_container_width=True, help="Sales breakdown by region/city"):
        prompt_to_run = f"Show total sales broken down by City or Region for {current_year}, as a bar chart."
    if qa_c3.button("📉 Slow Movers CY", use_container_width=True, help="Identify bottom performing items"):
        prompt_to_run = f"Which 5 products have the lowest Sales in {current_year}? Show in a table."

    # AI Agent Setup
    if "messages" not in st.session_state: st.session_state.messages = []
    if "agent" not in st.session_state:
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        prefix = f"""You are an expert Bikano Data Analyst. Dataset years: {min_year}-{max_year}. 'CY'={current_year}. Always provide Final Answer as nicely formatted Markdown Tables."""
        st.session_state.agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True, max_iterations=10, prefix=prefix)

    # Chat History Display
    chat_container = st.container(height=500)
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], dict):
                    if "text" in msg["content"]: st.markdown(msg["content"]["text"])
                    if "chart" in msg["content"]: 
                        st.plotly_chart(msg["content"]["chart"], use_container_width=True, key=f"chart_{i}")
                else:
                    st.markdown(msg["content"])

    # Input handling: Either from text input OR quick action button
    user_input = st.chat_input("Ex: 'Compare sales of Product A vs B'")
    
    # Determine final prompt to process
    final_prompt = prompt_to_run if prompt_to_run else user_input

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(final_prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Bikano AI is crunching the numbers..."):
                    text_resp = ""
                    try:
                        style = "\nIMPORTANT: Format numbers with commas. Use Markdown Tables. Start response with 'Final Answer:'."
                        resp = st.session_state.agent.invoke(final_prompt + style)
                        text_resp = resp['output']
                        st.markdown(text_resp)
                    except Exception as e:
                        text_resp = f"Could not format data. Error: {e}"
                        st.error(text_resp)

                    # Intelligent Chart Triggering
                    need_chart = any(k in final_prompt.lower() for k in ["chart", "plot", "graph", "trend", "compare"])
                    fig = None
                    if need_chart and "Error" not in text_resp:
                        fig = generate_chart(final_prompt, df)
                    
                    if fig:
                        st.session_state.messages.append({"role": "assistant", "content": {"text": text_resp, "chart": fig}})
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_new_{len(st.session_state.messages)}")
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": text_resp})
