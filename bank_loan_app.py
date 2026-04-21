import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="AI Loan Intelligence Engine", page_icon="📈", layout="wide")

# Advanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
    }
    
    .main {
        background-color: #0f172a;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
    }

    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    .stMetric {
        background: #1e293b;
        padding: 24px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        border-bottom: 4px solid #3b82f6;
    }
    
    .card-container {
        background: #1e293b;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    
    .status-banner {
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    }
    
    .eligible-gradient {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    
    .declined-gradient {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    }
    
    .big-stat {
        font-size: 54px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .label-text {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 14px;
        font-weight: 600;
        opacity: 0.9;
    }

    /* Force text color for Streamlit components */
    .stMarkdown, .stSlider, .stNumberInput, .stRadio, .stButton, .stHeader, h1, h2, h3, p, span, label {
        color: #f8fafc !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data for EDA
@st.cache_data
def load_eda_data():
    if os.path.exists('loan_data.csv'):
        return pd.read_csv('loan_data.csv')
    return None

# Load models
@st.cache_resource(ttl=3600) # Added TTL and will change key to force reset
def load_models_v2():
    # Use absolute paths to ensure files are found regardless of how the app is launched
    base_path = os.path.dirname(os.path.abspath(__file__))
    clf_path = os.path.join(base_path, 'models', 'classification_pipeline.joblib')
    reg_path = os.path.join(base_path, 'models', 'regression_pipeline.joblib')
    
    try:
        if not os.path.exists(clf_path) or not os.path.exists(reg_path):
            return None, None
            
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
        return clf, reg
    except Exception as e:
        st.error(f"Internal Load Error: {e}")
        return None, None

df_eda = load_eda_data()
clf_pipeline, reg_pipeline = load_models_v2()

if not clf_pipeline:
    import sklearn
    import joblib as jl
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    st.error(f"""
    ⚠️ **AI Underwriting Engine Offline**
    
    The cloud environment is having trouble locating the model files.
    
    **System Diagnostics:**
    - **Expected Path**: `{os.path.join(base_path, 'models')}`
    - **Current Folder Content**: `{os.listdir(base_path)}`
    - **Sklearn Version**: `{sklearn.__version__}`
    - **Joblib Version**: `{jl.__version__}`
    
    **How to fix:**
    1. Check if the `models/` folder is visible in your GitHub repository.
    2. Click **'Reboot App'** in the Streamlit Cloud dashboard.
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Intelligence Input")
    st.markdown("---")
    
    credit_score = st.slider("Target Credit Score", 300, 850, 720)
    age = st.slider("Applicant Age", 18, 70, 35)
    monthly_income = st.number_input("Gross Monthly Income ($)", 1000, 30000, 7500, step=500)
    previous_loan = st.radio("Existing Credit Facilities", ["Yes", "No"], horizontal=True)
    
    st.markdown("---")
    analyze = st.button("🚀 EXECUTE AI ANALYSIS", use_container_width=True, type="primary")

# Main Page Layout
st.title("🛡️ AI Loan Underwriting Intelligence")
tabs = st.tabs(["🎯 Decision Engine", "📊 Market Insights", "🧬 Model Explainability"])

# Tab 1: Decision Engine
with tabs[0]:
    if analyze:
        input_df = pd.DataFrame({
            'Credit_Score': [credit_score],
            'Age': [age],
            'Monthly_Income': [monthly_income],
            'Previous_Loan': [previous_loan]
        })
        
        # Predictions
        prob = clf_pipeline.predict_proba(input_df)[0][1]
        is_eligible = clf_pipeline.predict(input_df)[0]
        
        # Header Status
        status_class = "eligible-gradient" if is_eligible == 1 else "declined-gradient"
        status_text = "CREDIT APPROVED" if is_eligible == 1 else "CREDIT DECLINED"
        
        st.markdown(f"""
            <div class="status-banner {status_class}">
                <div class="label-text">{status_text}</div>
                <div class="big-stat">{prob:.1%}</div>
                <div class="label-text">Underwriting Confidence Index</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Risk Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score Spectrum", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 40], 'color': "#ef4444"},
                    {'range': [40, 70], 'color': "#f59e0b"},
                    {'range': [70, 100], 'color': "#10b981"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", height=300, margin=dict(t=0, b=0), template="plotly_dark")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if is_eligible == 1:
            terms = reg_pipeline.predict(input_df)[0]
            amt, rate, tenure = terms
            
            st.subheader("💰 Proposed Facility Terms")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Max Limit", f"${amt:,.0f}", help="Calculated based on DTI and Score")
            with c2: st.metric("APR", f"{rate:.2f}%", delta=f"{rate-12:.1f}% vs Avg", delta_color="inverse")
            with c3: st.metric("Tenure", f"{int(tenure)} Mo")
            
            st.success("✅ Applicant meets all algorithmic safety requirements for this facility.")
        else:
            st.error("❌ Profile falls outside current risk appetite. High correlation with historical defaults found.")
            
    else:
        st.info("👋 System Idle. Please configure applicant parameters in the sidebar and execute analysis.")

# Tab 2: Market Insights
with tabs[1]:
    st.subheader("📊 Comparative Market Analysis")
    if df_eda is not None:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Credit Score vs. Approval Density**")
            fig1 = px.histogram(df_eda, x="Credit_Score", color="Is_Eligible", 
                                barmode="overlay", color_discrete_sequence=["#ef4444", "#10b981"],
                                labels={"Credit_Score": "FICO Score"}, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.markdown("**Income Distribution vs. Loan Amount**")
            fig2 = px.scatter(df_eda[df_eda['Is_Eligible']=='Yes'], x="Monthly_Income", y="Loan_Amount", 
                              color="Interest_Rate", size="Age", opacity=0.6,
                              color_continuous_scale="RdYlGn_r", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("**Correlation Matrix of Portfolio Features**")
        numeric_df = df_eda.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Historical data not available for visualization.")

# Tab 3: Model Explainability
with tabs[2]:
    st.subheader("🧬 Neural/Tree Logic Breakdown")
    if analyze:
        try:
            model = clf_pipeline.named_steps['classifier']
            preprocessor = clf_pipeline.named_steps['preprocessor']
            input_preprocessed = preprocessor.transform(input_df)
            
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['Previous_Loan'])
            feature_names = ['Credit Score', 'Age', 'Monthly Income'] + list(cat_features)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_preprocessed)
            
            # Using Plotly for a better SHAP representation
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_values[1][0]
            }).sort_values('Impact', ascending=True)
            
            fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h',
                             color='Impact', color_continuous_scale='RdYlGn',
                             title="Feature Contribution to Approval", template="plotly_dark")
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("How to read: Bars to the right (green) increased approval chance. Bars to the left (red) decreased it.")
            
        except Exception as e:
            st.error(f"Explainability module error: {e}")
    else:
        st.info("Run analysis to view the logic behind the decision.")

# Footer
st.markdown("---")
st.caption("Underwriting Engine v2.4.0 | Powered by Random Forest Ensembles | Local Time: 2026-04-18")
