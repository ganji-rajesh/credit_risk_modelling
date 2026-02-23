"""
Credit Risk Scoring App — Streamlit Deployment
Author: Senior Data Scientist
Date: 2026-02-23

A production-ready web application for real-time credit risk assessment.
Scores loans using calibrated PD (Probability of Default) and LGD (Loss Given Default) models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title='Credit Risk Scorer',
    page_icon='💰',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-medium { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_models():
    """Load pre-trained PD and LGD models."""
    try:
        pd_model = joblib.load('pd_model.pkl')
        lgd_model = joblib.load('lgd_model.pkl')
        return pd_model, lgd_model
    except FileNotFoundError:
        st.error('❌ Models not found. Ensure pd_model.pkl and lgd_model.pkl are in the app directory.')
        return None, None

pd_model, lgd_model = load_models()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURES = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti',
    'fico_score', 'term', 'grade', 'emp_length',
    'delinq_2yrs', 'inq_last_6mths', 'revol_util', 'open_acc'
]

FEATURE_CONFIG = {
    'loan_amnt': {
        'label': 'Loan Amount ($)',
        'type': 'slider',
        'min': 500,
        'max': 40000,
        'step': 500,
        'default': 15000,
        'help': 'Principal loan amount requested'
    },
    'int_rate': {
        'label': 'Interest Rate (%)',
        'type': 'slider',
        'min': 5.0,
        'max': 30.0,
        'step': 0.5,
        'default': 12.0,
        'help': 'Annual interest rate on the loan'
    },
    'annual_inc': {
        'label': 'Annual Income ($)',
        'type': 'slider',
        'min': 20000,
        'max': 200000,
        'step': 5000,
        'default': 60000,
        'help': 'Self-reported annual income'
    },
    'dti': {
        'label': 'Debt-to-Income Ratio (%)',
        'type': 'slider',
        'min': 0.0,
        'max': 100.0,
        'step': 1.0,
        'default': 20.0,
        'help': 'Monthly debt obligations / monthly income'
    },
    'fico_score': {
        'label': 'FICO Score',
        'type': 'slider',
        'min': 300,
        'max': 850,
        'step': 5,
        'default': 720,
        'help': 'Credit bureau FICO score (higher = better)'
    },
    'term': {
        'label': 'Loan Term (months)',
        'type': 'radio',
        'options': [36, 60],
        'default': 36,
        'help': '36 or 60 month loan term'
    },
    'grade': {
        'label': 'Risk Grade',
        'type': 'radio',
        'options': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'default': 'C',
        'help': 'LendingClub grade (A=best, G=worst)'
    },
    'emp_length': {
        'label': 'Employment Length (years)',
        'type': 'slider',
        'min': 0,
        'max': 10,
        'step': 1,
        'default': 5,
        'help': 'Years at current employer (capped at 10)'
    },
    'delinq_2yrs': {
        'label': 'Delinquencies in Last 2 Years',
        'type': 'slider',
        'min': 0,
        'max': 10,
        'step': 1,
        'default': 0,
        'help': 'Number of delinquency events in past 24 months'
    },
    'inq_last_6mths': {
        'label': 'Inquiries in Last 6 Months',
        'type': 'slider',
        'min': 0,
        'max': 10,
        'step': 1,
        'default': 1,
        'help': 'Number of credit inquiries in past 6 months'
    },
    'revol_util': {
        'label': 'Revolving Credit Utilization (%)',
        'type': 'slider',
        'min': 0.0,
        'max': 100.0,
        'step': 1.0,
        'default': 40.0,
        'help': 'Revolving credit used / revolving credit limit'
    },
    'open_acc': {
        'label': 'Open Accounts',
        'type': 'slider',
        'min': 0,
        'max': 30,
        'step': 1,
        'default': 8,
        'help': 'Number of open credit accounts'
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def encode_inputs(inputs_dict):
    """Encode categorical inputs to match model training."""
    encoded = inputs_dict.copy()
    
    # Encode grade (A=1, B=2, ..., G=7)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    encoded['grade'] = grade_map[encoded['grade']]
    
    return encoded

def get_risk_label(pd_score, lgd_score):
    """Classify overall risk level."""
    risk_product = pd_score * lgd_score
    
    if risk_product > 0.20:
        return '🔴 HIGH RISK', 'risk-high'
    elif risk_product > 0.08:
        return '🟡 MEDIUM RISK', 'risk-medium'
    else:
        return '🟢 LOW RISK', 'risk-low'

def create_risk_gauge(pd_score):
    """Create a gauge chart for PD visualization."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=pd_score * 100,
        title={'text': 'Probability of Default (%)'},
        delta={'reference': 5},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 5], 'color': 'lightgray'},
                {'range': [5, 15], 'color': 'yellow'},
                {'range': [15, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 15
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_input_form():
    """Create a form for feature inputs."""
    col1, col2, col3 = st.columns(3)
    inputs = {}
    
    # Organize features across columns
    feature_list = list(FEATURE_CONFIG.keys())
    columns = [col1, col2, col3]
    
    for idx, feature in enumerate(feature_list):
        col = columns[idx % 3]
        config = FEATURE_CONFIG[feature]
        
        with col:
            if config['type'] == 'slider':
                inputs[feature] = st.slider(
                    config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['default'],
                    step=config['step'],
                    help=config['help']
                )
            elif config['type'] == 'radio':
                inputs[feature] = st.radio(
                    config['label'],
                    options=config['options'],
                    index=config['options'].index(config['default']),
                    help=config['help']
                )
    
    return inputs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header & Description
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title('💰 Credit Risk Scoring Engine')
st.markdown("""
    **Real-time assessment of loan default probability and loss severity**
    
    This application uses machine learning models trained on LendingClub loan data to estimate:
    - **PD (Probability of Default)**: Likelihood the borrower will default within the loan term
    - **LGD (Loss Given Default)**: Expected loss as % of loan amount if default occurs
    - **EL (Expected Loss)**: Dollar amount expected to lose = PD × LGD × Loan Amount
""")

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if pd_model and lgd_model:
    # Sidebar for mode selection
    st.sidebar.header('⚙️ Settings')
    mode = st.sidebar.radio(
        'Scoring Mode',
        ['Single Loan', 'Batch Upload'],
        help='Score one loan at a time or upload a CSV with multiple loans'
    )
    
    if mode == 'Single Loan':
        # ─────────────────────────────────────────────────────────────────────
        # SINGLE LOAN SCORING
        # ─────────────────────────────────────────────────────────────────────
        st.header('📋 Loan Scoring')
        
        # Create preset templates
        col_template_a, col_template_b, col_template_c = st.columns(3)
        
        with col_template_a:
            if st.button('📌 Prime Borrower', use_container_width=True):
                st.session_state.preset = 'prime'
        
        with col_template_b:
            if st.button('📌 Standard Borrower', use_container_width=True):
                st.session_state.preset = 'standard'
        
        with col_template_c:
            if st.button('📌 Subprime Borrower', use_container_width=True):
                st.session_state.preset = 'subprime'
        
        st.markdown('---')
        
        # Apply presets if selected
        if 'preset' in st.session_state:
            if st.session_state.preset == 'prime':
                preset_values = {
                    'loan_amnt': 25000, 'int_rate': 8.0, 'annual_inc': 120000,
                    'dti': 15.0, 'fico_score': 750, 'term': 60,
                    'grade': 'A', 'emp_length': 8,
                    'delinq_2yrs': 0, 'inq_last_6mths': 0, 'revol_util': 20.0, 'open_acc': 10
                }
                st.info('✨ Prime borrower preset applied')
            elif st.session_state.preset == 'standard':
                preset_values = {
                    'loan_amnt': 15000, 'int_rate': 14.5, 'annual_inc': 60000,
                    'dti': 25.0, 'fico_score': 680, 'term': 36,
                    'grade': 'C', 'emp_length': 5,
                    'delinq_2yrs': 0, 'inq_last_6mths': 1, 'revol_util': 45.0, 'open_acc': 8
                }
                st.info('📖 Standard borrower preset applied')
            else:  # subprime
                preset_values = {
                    'loan_amnt': 8000, 'int_rate': 22.0, 'annual_inc': 35000,
                    'dti': 35.0, 'fico_score': 620, 'term': 36,
                    'grade': 'E', 'emp_length': 2,
                    'delinq_2yrs': 2, 'inq_last_6mths': 3, 'revol_util': 85.0, 'open_acc': 5
                }
                st.warning('⚠️ Subprime borrower preset applied')
            
            # Override feature config defaults
            for feat, value in preset_values.items():
                FEATURE_CONFIG[feat]['default'] = value
        
        # Loan feature input form
        inputs = create_feature_input_form()
        
        st.markdown('---')
        
        # Score the loan
        col_score_btn, col_reset_btn = st.columns(2)
        
        with col_score_btn:
            score_btn = st.button('🎯 Score Loan', type='primary', use_container_width=True)
        
        with col_reset_btn:
            if st.button('🔄 Reset', use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        if score_btn:
            # Encode inputs
            encoded_inputs = encode_inputs(inputs)
            
            # Create DataFrame for prediction
            X_input = pd.DataFrame([encoded_inputs], columns=FEATURES)
            
            # Make predictions
            pd_score = pd_model.predict_proba(X_input)[0, 1]
            lgd_score = np.clip(lgd_model.predict(X_input)[0], 0, 1)
            
            # Calculate Expected Loss
            ead = inputs['loan_amnt']  # Exposure = loan amount
            el = pd_score * lgd_score * ead
            
            # Get risk classification
            risk_label, risk_class = get_risk_label(pd_score, lgd_score)
            
            # Display Results
            st.divider()
            st.header('📊 Scoring Results')
            
            # Risk classification banner
            st.markdown(f"### {risk_label}", unsafe_allow_html=True)
            
            # Metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label='Probability of Default',
                    value=f'{pd_score*100:.2f}%',
                    delta='Risk of borrower defaulting',
                    delta_color='inverse'
                )
            
            with metric_col2:
                st.metric(
                    label='Loss Given Default',
                    value=f'{lgd_score*100:.2f}%',
                    delta='Loss if default occurs',
                    delta_color='inverse'
                )
            
            with metric_col3:
                st.metric(
                    label='Expected Loss ($)',
                    value=f'${el:,.2f}',
                    delta=f'{100*el/ead:.2f}% of loan amount',
                    delta_color='inverse'
                )
            
            st.markdown('---')
            
            # Calculate expected loss percentage
            el_pct = el / ead * 100
            
            # Detailed metrics table
            st.subheader('📋 Detailed Metrics')
            metrics_table = pd.DataFrame({
                'Metric': ['Loan Amount', 'Interest Rate', 'Annual Income', 'FICO Score', 
                          'DTI', 'Grade', 'Default Probability', 'Loss If Default',
                          'Expected Loss ($)', 'Expected Loss (%)'],
                'Value': [f'${inputs["loan_amnt"]:,}', f'{inputs["int_rate"]:.1f}%',
                         f'${inputs["annual_inc"]:,}', f'{inputs["fico_score"]}',
                         f'{inputs["dti"]:.1f}%', inputs['grade'],
                         f'{pd_score*100:.2f}%', f'{lgd_score*100:.2f}%',
                         f'${el:,.2f}', f'{el_pct:.2f}%']
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
    
    else:
        # ─────────────────────────────────────────────────────────────────────
        # BATCH SCORING
        # ─────────────────────────────────────────────────────────────────────
        st.header('📤 Batch Loan Scoring')
        
        uploaded_file = st.file_uploader('Upload CSV with loan data', type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                missing_cols = [f for f in FEATURES if f not in df.columns]
                if missing_cols:
                    st.error(f'❌ Missing columns: {missing_cols}')
                else:
                    st.success(f'✅ Found all {len(FEATURES)} required features')
                    
                    # Encode grade if present
                    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
                    if 'grade' in df.columns and df['grade'].dtype == 'object':
                        df['grade'] = df['grade'].map(grade_map)
                    
                    # Score all loans
                    X_batch = df[FEATURES]
                    
                    pd_scores = pd_model.predict_proba(X_batch)[:, 1]
                    lgd_scores = np.clip(lgd_model.predict(X_batch), 0, 1)
                    el_scores = pd_scores * lgd_scores * df['loan_amnt']
                    
                    # Add results to dataframe
                    results_df = df.copy()
                    results_df['pd_probability'] = pd_scores
                    results_df['lgd_loss_rate'] = lgd_scores
                    results_df['expected_loss_amt'] = el_scores
                    results_df['expected_loss_pct'] = (el_scores / df['loan_amnt']) * 100
                    
                    # Risk classification
                    results_df['risk_level'] = results_df.apply(
                        lambda row: 'HIGH' if row['expected_loss_pct'] > 10
                        else ('MEDIUM' if row['expected_loss_pct'] > 5 else 'LOW'),
                        axis=1
                    )
                    
                    # Display summary stats
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric('Total Loans', len(results_df))
                    
                    with col_stat2:
                        avg_pd = results_df['pd_probability'].mean()
                        st.metric('Avg PD', f'{avg_pd*100:.2f}%')
                    
                    with col_stat3:
                        avg_el = results_df['expected_loss_amt'].sum()
                        st.metric('Total Expected Loss', f'${avg_el:,.0f}')
                    
                    with col_stat4:
                        high_risk_count = (results_df['risk_level'] == 'HIGH').sum()
                        st.metric('High Risk', high_risk_count)
                    
                    st.markdown('---')
                    
                    # Display results table
                    st.subheader('📊 Scoring Results')
                    display_cols = ['loan_amnt', 'int_rate', 'fico_score', 'grade',
                                   'pd_probability', 'lgd_loss_rate', 'expected_loss_amt',
                                   'expected_loss_pct', 'risk_level']
                    
                    st.dataframe(
                        results_df[display_cols].round(4),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label='📥 Download Scored Results (CSV)',
                        data=csv,
                        file_name=f'scored_loans_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
            
            except Exception as e:
                st.error(f'❌ Error processing file: {str(e)}')

else:
    st.error('❌ Could not load models. Please check that pd_model.pkl and lgd_model.pkl exist.')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Credit Risk Scoring Engine | Basel-aligned PD & LGD Models | LendingClub Data (2007-2018)</p>
    <p>⚠️ <strong>Disclaimer</strong>: This tool is for educational and reference purposes only.
    Actual lending decisions should incorporate additional factors and human judgment.</p>
    </div>
""", unsafe_allow_html=True)
