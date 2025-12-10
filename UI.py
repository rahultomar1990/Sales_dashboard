import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import os

# --- 1. PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Sales Data Analysis", page_icon="📈")

# --- 2. CUSTOM CSS (MODERN & COMPACT) ---
st.markdown("""
    <style>
        /* 1. COMPACT LAYOUT - SHIFT EVERYTHING UP */
        .block-container {
            padding-top: 1.5rem !important; /* Minimum top padding */
            padding-bottom: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* 2. MODERN TITLE */
        h1 {
            text-align: left;
            background: -webkit-linear-gradient(45deg, #0b5394, #662d91);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0px !important;
            padding-bottom: 10px;
        }
        
        /* 3. MODERN SCORECARDS */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff, #f0f2f6);
            border: 1px solid #e0e0e0;
            padding: 10px;
            border-radius: 10px;
            border-left: 5px solid #6C63FF;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricValue"] {
            font-size: 22px;
            font-weight: 700;
            color: #0b5394;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 13px;
            font-weight: 600;
            color: #555;
        }

        /* 4. MODERN CHAT BUBBLES */
        /* User Message */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #f0f2f6;
            border-radius: 15px;
            padding: 10px;
            border: 1px solid #e0e0e0;
        }
        /* AI Message */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 10px;
            border: 1px solid #d1d5db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        /* 5. INPUT BOX STYLING */
        .stChatInput {
            border-radius: 20px !important;
        }
        
        /* 6. HIDE DEFAULT FOOTER */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. CONFIGURATION ---
# Use secrets in real app, hardcoded here for easy deployment
GROQ_API_KEY = "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" 
CACHE_FILE = "sales_data_cache.parquet"

# --- 4. CLOUD DATA ENGINE ---
@st.cache_data(show_spinner=False)
def get_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_parquet(CACHE_FILE)
            
            # RE-APPLY TYPES (Critical for Parquet)
            if 'Sales' in df.columns: df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
            if 'Units' in df.columns: df['Units'] = pd.to_numeric(df['Units'], errors='coerce').fillna(0)
            if 'Order Date' in df.columns:
                df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
                df['Year'] = df['Order Date'].dt.year
                df['Month'] = df['Order Date'].dt.month_name()
                
            # SORTING (Fixes Chart Ordering)
            df = df.sort_values(by='Order Date')
            
            # CATEGORICAL MONTHS (Fixes Chart Sorting)
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            if 'Month' in df.columns:
                df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
                
            return df
        except Exception as e:
            return f"Error reading cache: {str(e)}"
    else:
        return "⚠️ Data File Missing! Please upload 'sales_data_cache.parquet' to GitHub."

# --- LOAD DATA ---
with st.spinner("🚀 Booting AI Analyst..."):
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
        User Query: "{prompt}"
        Data Columns: {df.columns.tolist()}
        INSTRUCTIONS:
        1. Create a figure named 'fig'.
        2. **MANDATORY**: For Bar Charts, Set `color` to the x-axis column.
        3. Style: `template='plotly_white'`, `color_discrete_sequence=px.colors.qualitative.Bold`.
        4. OUTPUT: RAW CODE ONLY. NO MARKDOWN.
        """
        response = chart_llm.invoke(code_prompt)
        cleaned_code = response.content.replace("```python", "").replace("```", "").strip()
        local_vars = {"df": df, "px": px}
        exec(cleaned_code, globals(), local_vars)
        return local_vars.get('fig', None)
    except:
        return None

# --- 6. CALCULATIONS ---
current_year = int(df['Year'].max()) if 'Year' in df.columns else 2024
last_year = current_year - 1
cy_sales = df[df['Year'] == current_year]['Sales'].sum() if 'Year' in df.columns else 0
ly_sales = df[df['Year'] == last_year]['Sales'].sum() if 'Year' in df.columns else 0
yoy_growth = ((cy_sales - ly_sales) / ly_sales) * 100 if ly_sales > 0 else 0

# --- 7. MAIN LAYOUT ---
col_dash, col_chat = st.columns([1.3, 1], gap="small")

# --- LEFT COLUMN: DASHBOARD ---
with col_dash:
    st.title("Sales Data Analysis")
    
    # METRICS ROW 1
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Sales", f"${df['Sales'].sum():,.0f}")
    c2.metric(f"📅 CY {current_year}", f"${cy_sales:,.0f}")
    c3.metric(f"⏮️ LY {last_year}", f"${ly_sales:,.0f}", delta=f"{cy_sales-ly_sales:,.0f}")
    
    # METRICS ROW 2 (Compact)
    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    c4.metric("📈 YoY Growth", f"{yoy_growth:.2f}%", delta=f"{yoy_growth:.2f}%")
    c5.metric("📦 Orders", f"{df['Order ID'].nunique() if 'Order ID' in df.columns else 0:,}")
    c6.metric("🔢 Units", f"{df['Units'].sum() if 'Units' in df.columns else 0:,.0f}")
    
    st.markdown("---")
    
    # CHARTS (Side-by-Side)
    ch1, ch2 = st.columns(2)
    
    with ch1:
        if 'Year' in df.columns:
            st.caption(f"📅 Sales Trend ({last_year} vs {current_year})")
            trend_df = df[df['Year'].isin([current_year, last_year])]
            monthly_sales = trend_df.groupby(['Month', 'Year'])['Sales'].sum().reset_index()
            
            fig_bar = px.bar(monthly_sales, x='Month', y='Sales', color='Year', barmode='group', 
                             color_discrete_sequence=px.colors.qualitative.Bold)
            # FIX: Adjust margins so labels aren't cut off
            fig_bar.update_layout(
                xaxis_title=None, 
                yaxis_title=None, 
                legend_title=None, 
                height=280, # Compact Height
                margin=dict(l=0, r=0, t=0, b=0),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        if 'Division' in df.columns:
            st.caption("🏢 Sales by Division")
            div_sales = df.groupby('Division')['Sales'].sum().reset_index()
            fig_pie = px.pie(div_sales, values='Sales', names='Division', hole=0.6, 
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pie.update_traces(textposition='inside', textinfo='percent')
            fig_pie.update_layout(
                height=280, # Compact Height
                margin=dict(l=0, r=0, t=0, b=0),
                template="plotly_white",
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("📄 View Raw Data"):
        st.dataframe(df.head(5), use_container_width=True)

# --- RIGHT COLUMN: CHAT BOT ---
with col_chat:
    st.markdown("### 🤖 AI Analyst")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    
    if "agent" not in st.session_state:
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        prefix = f"""
        You are an expert Data Analyst using Pandas.
        - Dataset Years: {int(df['Year'].min())} to {int(df['Year'].max())}.
        - CY = {current_year}, LY = {last_year}.
        - Output Markdown Tables.
        """
        st.session_state.agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True, max_iterations=10, prefix=prefix
        )

    # Chat Container
    chat_container = st.container(height=520)
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], dict):
                    st.markdown(msg["content"]["text"])
                    if "chart" in msg["content"]:
                        with st.expander("📊 Visual", expanded=True):
                            st.plotly_chart(msg["content"]["chart"], use_container_width=True, key=f"hist_{i}")
                else:
                    st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about sales, trends, or top products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        resp = st.session_state.agent.invoke(prompt)
                        ans = resp['output']
                        st.markdown(ans)
                        
                        if any(k in prompt.lower() for k in ["chart", "plot", "graph"]):
                            fig = generate_chart(prompt, df)
                            if fig:
                                st.session_state.messages.append({"role": "assistant", "content": {"text": ans, "chart": fig}})
                                with st.expander("📊 Visual", expanded=True):
                                    st.plotly_chart(fig, use_container_width=True, key=f"new_{len(st.session_state.messages)}")
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": ans})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": ans})
                    except Exception as e:
                        st.error(str(e))
