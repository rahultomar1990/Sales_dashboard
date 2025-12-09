import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import os

# --- 1. PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Sales Data Analysis", page_icon="📈")

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
        .block-container { padding: 1rem 1rem 3rem 1rem; }
        h1 { 
            text-align: left;
            background: -webkit-linear-gradient(45deg, #0b5394, #662d91);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.0rem; font-weight: 800; margin-bottom: 10px;
        }
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 242, 246, 0.8));
            border: 1px solid #e0e0e0; padding: 10px; border-radius: 12px;
            border-left: 6px solid #6C63FF; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-4px); box-shadow: 0 10px 15px rgba(0,0,0,0.1); border-left: 6px solid #FF6584;
        }
        div[data-testid="stMetricValue"] {
            font-size: 20px; font-weight: 700;
            background: -webkit-linear-gradient(90deg, #0052cc, #00a3ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        div[data-testid="column"]:nth-of-type(1) {
            border-right: 2px solid transparent;
            border-image: linear-gradient(to bottom, #f0f0f0 0%, #6C63FF 50%, #f0f0f0 100%);
            border-image-slice: 1; padding-right: 25px;
        }
        @media (min-width: 768px) {
            .stChatInput { position: fixed; bottom: 20px; width: 28% !important; z-index: 1000; border-radius: 20px; box-shadow: 0 -4px 20px rgba(0,0,0,0.1); }
        }
        @media (max-width: 768px) {
            .stChatInput { position: fixed; bottom: 0px; left: 0px; width: 100% !important; z-index: 1000; border-radius: 0px; padding-bottom: 10px; background-color: white; }
            div[data-testid="column"]:nth-of-type(1) { border-right: none; padding-right: 0px; }
        }
        .streamlit-expanderHeader { font-weight: 700; color: #0b5394; background-color: #f0f4f8; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. CONFIGURATION ---
# IMPORTANT: In a real app, use st.secrets for this key. For this demo, we keep it here.
GROQ_API_KEY = "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"
CACHE_FILE = "sales_data_cache.parquet"

# --- 4. DATA ENGINE (CLOUD ONLY) ---
@st.cache_data(show_spinner=False)
def get_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_parquet(CACHE_FILE)
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            if 'Month' in df.columns:
                df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
            return df
        except Exception as e:
            return f"Error reading cache: {str(e)}"
    else:
        return "⚠️ Data File Missing! Please upload 'sales_data_cache.parquet' to your GitHub repository."

with st.spinner("🚀 Loading Cloud Data..."):
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
        1. INTELLIGENTLY CHOOSE CHART TYPE (Bar, Line, Pie).
        2. **MANDATORY COLOR RULE**: 
           - For Bar Charts, you MUST set `color` to the same column as `x`. 
             Example: `px.bar(df, x='City', y='Sales', color='City', ...)`
        3. **STYLE**:
           - Use `template='plotly_white'`.
           - Set `color_discrete_sequence=px.colors.qualitative.Vivid`.
        4. OUTPUT: 
           - Write ONLY the raw Python code to create a figure named 'fig'. 
           - Do NOT include markdown ticks.
           - Use 'df' as the dataframe variable.
        """
        response = chart_llm.invoke(code_prompt)
        cleaned_code = response.content.replace("```python", "").replace("```", "").strip()
        local_vars = {"df": df, "px": px}
        exec(cleaned_code, globals(), local_vars)
        return local_vars.get('fig', None)
    except Exception as e:
        return None

# --- 6. CALCULATIONS & LAYOUT ---
min_year = int(df['Year'].min()) if 'Year' in df.columns else 2020
max_year = int(df['Year'].max()) if 'Year' in df.columns else 2024
current_year = max_year
last_year = current_year - 1
cy_sales = df[df['Year'] == current_year]['Sales'].sum() if 'Year' in df.columns else 0
ly_sales = df[df['Year'] == last_year]['Sales'].sum() if 'Year' in df.columns else 0
yoy_growth = ((cy_sales - ly_sales) / ly_sales) * 100 if ly_sales > 0 else 0

col_dash, col_chat = st.columns([2.5, 1.2], gap="medium")

# LEFT COLUMN
with col_dash:
    st.title("Sales Data Analysis with AI Assistant")
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("💰 Total Sales", f"${df['Sales'].sum():,.0f}")
    m2.metric(f"📅 CY {current_year}", f"${cy_sales:,.0f}")
    m3.metric(f"⏮️ LY {last_year}", f"${ly_sales:,.0f}", delta=f"{cy_sales-ly_sales:,.0f}")
    m4.metric("📈 YoY Growth", f"{yoy_growth:.2f}%", delta=f"{yoy_growth:.2f}%")
    m5.metric("📦 Orders", f"{df['Order ID'].nunique() if 'Order ID' in df.columns else 0:,}")
    m6.metric("🔢 Units", f"{df['Units'].sum() if 'Units' in df.columns else 0:,.0f}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if 'Year' in df.columns:
            st.markdown(f"#### 📅 Sales Trend ({last_year} vs {current_year})")
            trend_df = df[df['Year'].isin([current_year, last_year])]
            monthly_sales = trend_df.groupby(['Month', 'Year'])['Sales'].sum().reset_index()
            fig_bar = px.bar(monthly_sales, x='Month', y='Sales', color='Year', barmode='group', 
                             text_auto='.2s', color_discrete_sequence=px.colors.qualitative.Bold)
            fig_bar.update_layout(xaxis_title=None, yaxis_title=None, legend_title="Year", height=350, template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        if 'Division' in df.columns:
            st.markdown("#### 🏢 Sales by Division")
            div_sales = df.groupby('Division')['Sales'].sum().reset_index()
            fig_pie = px.pie(div_sales, values='Sales', names='Division', hole=0.5, 
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("📄 View Raw Data Snippet"):
        st.dataframe(df.head(10), use_container_width=True)

# RIGHT COLUMN
with col_chat:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("🤖 AI Data Analyst (Groq)")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    if "agent" not in st.session_state:
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        prefix = f"Data: {min_year}-{max_year}. CY={current_year}, LY={last_year}. Output: Markdown Table."
        st.session_state.agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True, max_iterations=10, prefix=prefix
        )

    chat_container = st.container(height=580)
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], dict):
                    st.markdown(msg["content"]["text"])
                    with st.expander("📊 View Chart", expanded=True):
                        st.plotly_chart(msg["content"]["chart"], use_container_width=True, key=f"chart_{i}")
                else:
                    st.markdown(msg["content"])

    if prompt := st.chat_input("Ask: 'Top 5 cities sales'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        resp = st.session_state.agent.invoke(prompt)
                        ans = resp['output']
                        st.markdown(ans)
                        
                        if any(k in prompt.lower() for k in ["chart", "plot", "graph", "pie", "bar"]):
                            fig = generate_chart(prompt, df)
                            if fig:
                                st.session_state.messages.append({"role": "assistant", "content": {"text": ans, "chart": fig}})
                                with st.expander("📊 View Chart", expanded=True):
                                    st.plotly_chart(fig, use_container_width=True, key=f"new_{len(st.session_state.messages)}")
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": ans})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": ans})
                    except Exception as e:
                        st.error(str(e))