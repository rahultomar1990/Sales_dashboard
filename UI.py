import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_groq import ChatGroq
import os

# --- 1. PAGE SETUP & BRANDING ---
st.set_page_config(layout="wide", page_title="Bikano Sales Intelligence", page_icon="📈")

# LOGO SETUP
LOGO_PATH = "Bikano new logo.jpg"
try:
    st.logo(LOGO_PATH, icon_image=LOGO_PATH)
except Exception:
    pass 

# --- 2. CSS FOR PERFECT ALIGNMENT & DARK MODE ---
st.markdown("""
    <style>
        /* 1. MAXIMIZE SCREEN USAGE */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* 2. DARK MODE THEME */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* 3. SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #262730;
            border-right: 1px solid #333;
        }

        /* 4. HEADERS */
        h1 { 
            font-size: 1.8rem !important; 
            margin-bottom: 5px !important;
            padding-bottom: 0px !important;
            color: #FFFFFF !important;
        }
        h3 {
            font-size: 1.3rem !important;
            margin-bottom: 10px !important;
            color: #FFFFFF !important;
        }
        
        /* 5. KPI TILES */
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 10px 15px !important; 
            border-radius: 8px;
            border-left: 4px solid #D32F2F; /* BIKANO RED */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stMetricLabel"] {
            color: #B0B0B0 !important;
            font-size: 13px !important;
        }
        div[data-testid="stMetricValue"] {
            color: #FFCA28 !important; /* BIKANO YELLOW */
            font-size: 24px !important;
        }

        /* 6. CHAT INPUT */
        .stChatInput {
            bottom: 10px !important; 
        }
        [data-testid="stChatMessage"] {
            background-color: #1E1E1E !important;
            border: 1px solid #333;
        }
        
        /* 7. CUSTOM BUTTONS */
        .stButton button {
            background: linear-gradient(45deg, #D32F2F, #B71C1C);
            color: white;
            border: none;
            border-radius: 6px;
            height: 38px;
            font-size: 14px;
            width: 100%;
        }
        .stButton button:hover {
            border: 1px solid #FFCA28;
            color: #fff;
        }

        /* 8. CHART TRANSPARENCY */
        .js-plotly-plot .plotly .main-svg {
            background: rgba(0,0,0,0) !important;
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
with st.spinner("🚀 Loading Dashboard..."):
    data_result = get_data()

if isinstance(data_result, str):
    st.error(data_result)
    st.stop()
else:
    df = data_result

# --- 5. CORE AI ENGINE (No-Loop Direct Execution) ---
def ask_ai(prompt, df):
    """
    Directly converts a question to Pandas code and executes it.
    This prevents 'Thinking' loops and is much faster.
    """
    try:
        # Use the Stable Llama 3 Model
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)
        
        # 1. Determine if we need a chart or just data
        code_prompt = f"""
        You are a Python Data Analyst. 
        DataFrame `df` has columns: {df.columns.tolist()}.
        User Question: "{prompt}"
        
        INSTRUCTIONS:
        1. Write PYTHON CODE to calculate the answer.
        2. Assign the final answer (string, number, or markdown table) to a variable named `result`.
        3. If the user asks for a plot/chart/graph, create a Plotly Express figure named `fig`.
        4. Wrap code in ```python blocks.
        5. DO NOT print anything. Just assign `result` or `fig`.
        
        Example:
        result = df['Sales'].sum()
        """
        
        response = llm.invoke(code_prompt)
        cleaned_code = response.content.replace("```python", "").replace("```", "").strip()
        
        # Execute Code safely
        local_vars = {"df": df, "px": px, "result": "I couldn't calculate that.", "fig": None}
        exec(cleaned_code, globals(), local_vars)
        
        return local_vars.get('result', "No result found."), local_vars.get('fig', None)
        
    except Exception as e:
        return f"Error processing query: {str(e)}", None

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
col_dash, col_chat = st.columns([1, 1], gap="medium") 

# ================= LEFT COLUMN: DASHBOARD =================
with col_dash:
    st.markdown(f"<h1>Bikano Sales Intelligence <span style='font-size:1.2rem; color:#FFCA28'>| FY {current_year}</span></h1>", unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # METRIC TILES
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1: st.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    with row1_c2: st.metric(f"CY {current_year} Sales", f"${cy_sales:,.0f}")
    with row1_c3: st.metric(f"LY {last_year} Sales", f"${ly_sales:,.0f}", delta=f"${cy_sales-ly_sales:,.0f}")
    
    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1: st.metric("YoY Growth", f"{yoy_growth:.2f}%", delta=f"{yoy_growth:.2f}%")
    with row2_c2: st.metric("Total Orders", f"{orders_count:,}")
    with row2_c3: st.metric("Units Sold", f"{units_count:,.0f}")
    
    st.markdown("<hr style='margin: 15px 0px; border-color: #333;'>", unsafe_allow_html=True)
    
    # TREND CHART
    if 'Year' in df.columns:
        st.markdown(f"### 📅 Sales Trend ({last_year} vs {current_year})")
        trend_df = df[df['Year'].isin([current_year, last_year])]
        monthly_sales = trend_df.groupby(['Month', 'Year'])['Sales'].sum().reset_index()
        
        custom_colors = ['#FFD600', '#FF5252'] # Yellow & Red
        
        fig_line = px.line(monthly_sales, x='Month', y='Sales', color='Year', 
                           markers=True, text='Sales',
                           color_discrete_sequence=custom_colors)
        
        fig_line.update_layout(
            xaxis_title=None, 
            yaxis_title=None, 
            legend_title=None, 
            height=340, 
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_line.update_traces(
            line=dict(width=3), 
            marker=dict(size=8),
            texttemplate='%{y:.2s}', 
            textposition='top center'
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ================= RIGHT COLUMN: AI ANALYST =================
with col_chat:
    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3>🤖 AI Executive Assistant</h3>", unsafe_allow_html=True)

    # ACTION BUTTONS
    qa_c1, qa_c2, qa_c3 = st.columns(3)
    prompt_to_run = None
    
    if qa_c1.button("🔥 Top Products", use_container_width=True):
        prompt_to_run = f"Show me a table of the top 5 Products by Sales in {current_year}."
    if qa_c2.button("🏙️ Region Sales", use_container_width=True):
        prompt_to_run = f"Show total sales broken down by City or Region for {current_year}, as a bar chart."
    if qa_c3.button("📉 Low Growth", use_container_width=True):
        prompt_to_run = f"Which 5 products have the lowest Sales in {current_year}? Show in a table."

    # CHAT HISTORY
    if "messages" not in st.session_state: st.session_state.messages = []
    
    chat_container = st.container(height=580)
    with chat_container:
        if not st.session_state.messages:
            st.info(f"👋 Ready! I can analyze data instantly.")
        
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], dict):
                    if "text" in msg["content"]: st.markdown(msg["content"]["text"])
                    if "chart" in msg["content"]: 
                        st.plotly_chart(msg["content"]["chart"], use_container_width=True, key=f"chart_{i}")
                else:
                    st.markdown(msg["content"])

    # INPUT
    user_input = st.chat_input("Ask about sales...")
    
    final_prompt = prompt_to_run if prompt_to_run else user_input

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(final_prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Calculating..."):
                    # CALL THE NEW DIRECT ENGINE
                    text_resp, fig = ask_ai(final_prompt, df)

                    # Display Response
                    if fig:
                        st.session_state.messages.append({"role": "assistant", "content": {"text": str(text_resp), "chart": fig}})
                        st.markdown(text_resp)
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_new_{len(st.session_state.messages)}")
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": str(text_resp)})
                        st.markdown(text_resp)
