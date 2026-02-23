# Credit Risk Scoring Engine — Streamlit Deployment

A production-ready web application for real-time credit risk assessment using Basel-aligned PD (Probability of Default) and LGD (Loss Given Default) models.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Ensure Models Exist

Verify that both trained models are in the app directory:
```
├── app.py
├── pd_model.pkl          # Probability of Default model
├── lgd_model.pkl         # Loss Given Default model
├── requirements_streamlit.txt
└── README.md
```

If models are missing, run the training notebook first:
```bash
jupyter notebook credit_risk_model_v2.ipynb
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📋 Features

### Single Loan Scoring
- **Real-time predictions** for individual loans
- **Preset templates** (Prime, Standard, Subprime borrowers)
- **Interactive sliders & radio buttons** for 12 loan features
- **Risk gauge visualization** for PD (Probability of Default)
- **Lending recommendation** (APPROVE / REVIEW / DECLINE)
- **Expected Loss breakdown** with decision rationale

### Batch Scoring
- **CSV upload** for bulk loan assessment
- **Instant scoring** of multiple loans
- **Risk classification** (HIGH / MEDIUM / LOW)
- **Summary statistics** (total exposure, avg PD, high-risk count)
- **Download results** as CSV for further analysis

---

## 🎯 Input Features

The app accepts 12 loan features:

| Feature | Range | Description |
|---------|-------|-------------|
| **loan_amnt** | $500 – $40,000 | Principal loan amount |
| **int_rate** | 5% – 30% | Annual interest rate |
| **annual_inc** | $20k – $200k | Self-reported annual income |
| **dti** | 0% – 100% | Debt-to-income ratio |
| **fico_score** | 300 – 850 | Credit bureau FICO score |
| **term** | 36 / 60 | Loan term in months |
| **grade** | A – G | LendingClub risk grade |
| **emp_length** | 0 – 10 | Years at current employer |
| **delinq_2yrs** | 0 – 10 | Delinquencies in last 24 months |
| **inq_last_6mths** | 0 – 10 | Credit inquiries in last 6 months |
| **revol_util** | 0% – 100% | Revolving credit utilization |
| **open_acc** | 0 – 30 | Number of open credit accounts |

---

## 📊 Output Metrics

The application returns three key risk indicators:

### 1. **Probability of Default (PD)**
- Calibrated prediction of loan default probability
- Range: 0% – 100%
- Lower = better

### 2. **Loss Given Default (LGD)**
- Expected loss as percentage of loan amount if default occurs
- Range: 0% – 100%
- Lower = better

### 3. **Expected Loss (EL)**
- **EL = PD × LGD × Loan Amount (EAD)**
- Dollar amount of expected loss
- Used for lending decision thresholds

---

## 🔴 Risk Classification

| Expected Loss % | Decision | Rationale |
|-----------------|----------|-----------|
| **< 2%** | 🟢 APPROVE | Low risk, strong approval candidate |
| **2% – 5%** | 🟡 APPROVE WITH CONDITIONS | Moderate risk, consider higher rates |
| **5% – 10%** | 🟠 REVIEW | Elevated risk, manual review recommended |
| **> 10%** | 🔴 DECLINE | High risk, not recommended for approval |

---

## 🔧 Customization

### Modify Risk Thresholds

Edit the `get_risk_label()` function in `app.py`:

```python
def get_risk_label(pd_score, lgd_score):
    risk_product = pd_score * lgd_score
    
    if risk_product > 0.20:          # Adjust these thresholds
        return '🔴 HIGH RISK', 'risk-high'
    elif risk_product > 0.08:
        return '🟡 MEDIUM RISK', 'risk-medium'
    else:
        return '🟢 LOW RISK', 'risk-low'
```

### Add New Preset Templates

Add borrower profiles in the sidebar section:

```python
if st.session_state.preset == 'my_profile':
    preset_values = {
        'loan_amnt': 20000,
        'int_rate': 10.0,
        # ... add all 12 features
    }
```

### Modify Feature Ranges

Update `FEATURE_CONFIG` dictionary to change slider ranges or defaults:

```python
FEATURE_CONFIG = {
    'fico_score': {
        'label': 'FICO Score',
        'type': 'slider',
        'min': 300,      # Change here
        'max': 850,      # Change here
        'default': 720,  # Change here
    },
    # ... other features
}
```

---

## 🌐 Deployment Options

### **Local Development**
```bash
streamlit run app.py
```

### **Streamlit Cloud** (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo and deploy
4. App is live at: `https://yourname-creditrisk.streamlit.app`

### **Docker** (Production)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY app.py .
COPY pd_model.pkl .
COPY lgd_model.pkl .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

Build & run:
```bash
docker build -t credit-risk-app .
docker run -p 8501:8501 credit-risk-app
```

### **AWS / Azure / GCP**
- Streamlit works on any cloud platform with Python
- Use managed container services (ECS, Cloud Run, App Service)
- Store models in cloud storage (S3, GCS, Blob Storage) if needed

---

## 📈 Performance Considerations

- **Model Loading**: Models are cached with `@st.cache_resource` (loads once on app start)
- **Batch Processing**: Handles ~10k loans in <5 seconds
- **Memory Usage**: ~500 MB for models + data
- **Latency**: Single prediction: <100ms

---

## ⚠️ Important Notes

1. **Model Assumptions**
   - Trained on LendingClub data (2007–2018)
   - FICO score scaled to [300, 850] range
   - Grade encoded as A=1, B=2, ..., G=7

2. **Feature Encoding**
   - Automatic encoding of `grade` feature (A–G → 1–7)
   - Other features should match training data distributions

3. **Data Quality**
   - Remove NaN values before batch upload
   - Ensure numeric columns are properly formatted
   - Validate feature ranges match historical data

4. **Model Updates**
   - Retrain models quarterly to capture concept drift
   - Monitor actual vs expected loss divergence
   - Update `pd_model.pkl` and `lgd_model.pkl` with new versions

---

## 📞 Support

For model questions, refer to the original notebook:
- `credit_risk_model_v2.ipynb`

For Streamlit documentation:
- https://docs.streamlit.io

---

## 📜 License & Disclaimer

⚠️ **Disclaimer**: This tool is for educational and reference purposes only. Actual lending decisions should incorporate:
- Additional risk factors (collateral, guarantees, macroeconomic conditions)
- Human judgment and expertise
- Regulatory compliance (Fair Lending, Truth in Lending, Basel III, etc.)
- Regular model validation and backtesting

---

**Last Updated**: February 23, 2026  
**Model Version**: basel-aligned-v1.0  
**Framework**: Streamlit 1.28+
