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

# --- 2. EXECUTIVE STYLING (IMPROVED CONTRAST) ---
st.markdown("""
    <style>
        /* APP BACKGROUND - Deep Royal Gradient for Contrast */
        .stApp {
            background: linear-gradient(160deg, #1a237e 0%, #4a148c 100%);
            background-attachment: fixed;
        }
        
        /* SIDEBAR STYLING - Clean White Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #ddd;
        }
        [data-testid="stSidebar"] * {
            color: #333 !important;
        }

        /* MAIN HEADERS (H1, H2, H3) - White Text to pop against Dark Background */
        h1, h2, h3 { 
            color: #ffffff !important;
            font-family: 'Helvetica Neue', sans-serif;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        }
        p, label {
            color: #e0e0e0 !important; /* Light grey for standard text on dark bg */
        }

        /* METRIC CARDS (The Glass Tiles) */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.9); /* Higher opacity for readability */
            border: 1px solid rgba(255, 255, 255, 1);
            padding: 20px !important;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); /* Stronger shadow */
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
             transform: translateY(-5px);
             border-left: 5px solid #d32f2f; /* Bikano Red Accent */
        }

        /* TEXT INSIDE METRICS - Force Black for Contrast */
        div[data-testid="stMetricValue"] {
            color: #d32f2f !important; /* Bikano Red for numbers */
            font-size: 28px !important;
            font-weight: 800 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #333333 !important; /* Dark Grey for labels */
            font-weight: 700 !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #333333 !important; /* Ensure delta is visible */
        }

        /* CHAT & INPUT AREAS */
        .stChatInput {
            background-color: white !important;
            border-radius: 20px !important;
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            color: #000;
        }
        /* Fix text color inside chat bubbles */
        [data-testid="stChatMessage"] p {
            color: #000 !important;
        }
        
        /* CUSTOM BUTTONS */
        .stButton button {
            background: linear-gradient(to right, #fbc02d, #f57f17); /* Bikano Yellow/Gold */
            color: #000;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            transition: all 0.2s;
        }
        .stButton button:hover {
             transform: scale(1.02);
             box-shadow: 0 4px 10px rgba(0,0,0,0.2);
             color: #000;
        }
        
        /* EXPANDER TEXT FIX */
        .streamlit-expanderHeader {
            color: #333 !important;
            background-color: #fff !important;
            border-radius: 5px;
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
        2. STYLE: Use `template='plotly_white'`.
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
    # Header with White Text (controlled by CSS)
    st.markdown(f"<h1>Bikano Sales Intelligence <span style='font-size:1.2rem; opacity:0.8'>| FY {current_year}</span></h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # GLASSMORPHISM TILES (3x2 Grid)
    # The CSS above forces the text inside these metrics to be Dark Grey/Red so they are readable on the white cards.
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1: st.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    with row1_c2: st.metric(f"CY {current_year} Sales", f"${cy_sales:,.0f}")
    with row1_c3: st.metric(f"LY {last_year} Sales", f"${ly_sales:,.0f}", delta=f"${cy_sales-ly_sales:,.0f}")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True) # Spacer

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1: st.metric("YoY Growth", f"{yoy_growth:.2f}%", delta=f"{yoy_growth:.2f}%")
    with row2_c2: st.metric("Total Orders", f"{orders_count:,}")
    with row2_c3: st.metric("Units Sold", f"{units_count:,.0f}")
    
    st.markdown("---")
    
    # TREND CHART
    if 'Year' in df.columns:
        st.subheader(f"📅 Sales Trend ({last_year} vs {current_year})")
        trend_df = df[df['Year'].isin([current_year, last_year])]
        monthly_sales = trend_df.groupby(['Month', 'Year'])['Sales'].sum().reset_index()
        
        # High contrast colors for the dark background
        custom_colors = ['#ffeb3b', '#00e5ff'] # Bright Yellow & Cyan for dark mode visibility
        
        fig_line = px.line(monthly_sales, x='Month', y='Sales', color='Year', markers=True,
                           color_discrete_sequence=px.colors.qualitative.Set1) # Using Set1 for distinct colors
        
        fig_line.update_layout(
            xaxis_title=None, 
            yaxis_title="Revenue ($)", 
            legend_title="Fiscal Year", 
            height=400, 
            template="plotly_white", # Keep chart background white for readability
            hovermode="x unified",
            paper_bgcolor='rgba(255,255,255,0.9)', # Semi-transparent white backing for chart
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_line.update_traces(line=dict(width=4), marker=dict(size=10, line=dict(width=2, color='white')))
        st.plotly_chart(fig_line, use_container_width=True)

# ================= RIGHT COLUMN: AI ANALYST =================
with col_chat:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    # Header
    st.markdown("<h3>🤖 AI Executive Assistant</h3>", unsafe_allow_html=True)
    st.markdown("<p>Ask questions about your sales data naturally.</p>", unsafe_allow_html=True)

    # QUICK ACTION BUTTONS
    qa_c1, qa_c2, qa_c3 = st.columns(3)
    prompt_to_run = None
    
    if qa_c1.button("🔥 Top Products", use_container_width=True):
        prompt_to_run = f"Show me a table of the top 5 Products by Sales in {current_year}."
    if qa_c2.button("🏙️ Region Sales", use_container_width=True):
        prompt_to_run = f"Show total sales broken down by City or Region for {current_year}, as a bar chart."
    if qa_c3.button("📉 Low Growth", use_container_width=True):
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

    # Input handling
    user_input = st.chat_input("Ex: 'Compare sales of Product A vs B'")
    
    final_prompt = prompt_to_run if prompt_to_run else user_input

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(final_prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    text_resp = ""
                    try:
                        style = "\nIMPORTANT: Format numbers with commas. Use Markdown Tables. Start response with 'Final Answer:'."
                        resp = st.session_state.agent.invoke(final_prompt + style)
                        text_resp = resp['output']
                        st.markdown(text_resp)
                    except Exception as e:
                        text_resp = f"Could not format data. Error: {e}"
                        st.error(text_resp)

                    need_chart = any(k in final_prompt.lower() for k in ["chart", "plot", "graph", "trend", "compare"])
                    fig = None
                    if need_chart and "Error" not in text_resp:
                        fig = generate_chart(final_prompt, df)
                    
                    if fig:
                        st.session_state.messages.append({"role": "assistant", "content": {"text": text_resp, "chart": fig}})
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_new_{len(st.session_state.messages)}")
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": text_resp})
