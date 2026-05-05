import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedCost AI — Medical Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — chamkeele / vibrant palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #24243e 100%);
    color: #f0f0f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e5f 0%, #2d2d8f 100%);
    border-right: 2px solid #7c3aed;
}
[data-testid="stSidebar"] * { color: #e2d9f3 !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stRadio > label {
    color: #c4b5fd !important; font-weight: 600;
    font-size: 0.85rem; letter-spacing: 0.5px;
}

.hero-banner {
    background: linear-gradient(90deg, #7c3aed, #db2777, #f59e0b);
    border-radius: 16px; padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.4);
}
.hero-banner h1 { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0; letter-spacing: -0.5px; }
.hero-banner p  { font-size: 1rem; color: #fde68a; margin: 0.4rem 0 0; font-weight: 400; }

.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    text-align: center; backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(124,58,237,0.35); }
.metric-card .label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #a78bfa; margin-bottom: 0.4rem; }
.metric-card .value { font-size: 1.9rem; font-weight: 800; color: #fff; }
.metric-card .unit  { font-size: 0.75rem; color: #c4b5fd; margin-top: 0.2rem; }

.result-card {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 60%, #f59e0b 100%);
    border-radius: 20px; padding: 2.2rem; text-align: center;
    box-shadow: 0 10px 40px rgba(219,39,119,0.45);
    animation: pulse-glow 2s ease-in-out infinite alternate;
}
@keyframes pulse-glow {
    from { box-shadow: 0 10px 40px rgba(219,39,119,0.4); }
    to   { box-shadow: 0 14px 50px rgba(245,158,11,0.6); }
}
.result-card .result-label { font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #fde68a; margin-bottom: 0.5rem; }
.result-card .result-value { font-size: 3.4rem; font-weight: 800; color: #fff; margin: 0; line-height: 1; }
.result-card .result-sub   { font-size: 0.9rem; color: #fde68a; margin-top: 0.6rem; }

.risk-badge {
    display: inline-block; padding: 0.4rem 1.2rem;
    border-radius: 50px; font-size: 0.85rem;
    font-weight: 700; letter-spacing: 0.5px; margin-top: 0.8rem;
}
.risk-low    { background: #065f46; color: #6ee7b7; border: 1px solid #6ee7b7; }
.risk-medium { background: #78350f; color: #fcd34d; border: 1px solid #fcd34d; }
.risk-high   { background: #7f1d1d; color: #fca5a5; border: 1px solid #fca5a5; }

.section-header {
    font-size: 1rem; font-weight: 700; color: #a78bfa;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 1.2rem 0 0.8rem;
    border-bottom: 1px solid rgba(167,139,250,0.25); padding-bottom: 0.4rem;
}

.profile-pill {
    display: inline-block;
    background: rgba(124,58,237,0.25); border: 1px solid #7c3aed;
    border-radius: 50px; padding: 0.3rem 0.9rem;
    font-size: 0.78rem; font-weight: 600; color: #c4b5fd; margin: 0.2rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: #fff; font-family: 'Poppins', sans-serif;
    font-weight: 700; font-size: 1rem; letter-spacing: 0.5px;
    border: none; border-radius: 12px; padding: 0.75rem 2rem;
    width: 100%; cursor: pointer;
    transition: opacity 0.2s ease, transform 0.15s ease;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}
div.stButton > button:hover { opacity: 0.9; transform: translateY(-2px); }

.info-box {
    background: rgba(16,185,129,0.12); border-left: 4px solid #10b981;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #6ee7b7; margin-top: 1rem;
}

hr { border: none; border-top: 1px solid rgba(167,139,250,0.2); margin: 1.5rem 0; }

.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 10px !important; color: #fff !important;
}

.feat-bar-wrap { margin: 0.4rem 0; }
.feat-label { font-size: 0.78rem; color: #c4b5fd; font-weight: 600; margin-bottom: 0.15rem; }
.feat-bar-bg { background: rgba(255,255,255,0.08); border-radius: 50px; height: 10px; overflow: hidden; }
.feat-bar-fill { height: 10px; border-radius: 50px; background: linear-gradient(90deg, #7c3aed, #f59e0b); }

.footer { text-align: center; font-size: 0.72rem; color: #6b7280; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(167,139,250,0.15); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL  — joblib fixes sklearn version mismatch
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("medical_cost.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ─────────────────────────────────────────────
#  SIDEBAR — INPUT CONTROLS
#  Column names kept EXACTLY as in dataset:
#  age, sex, bmi, children, smoker, region
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem;">
        <span style="font-size:3rem;">🏥</span>
        <h2 style="color:#c4b5fd; margin:0.3rem 0 0; font-size:1.2rem; font-weight:700;">Patient Profile</h2>
        <p style="font-size:0.75rem; color:#7c6fab; margin:0;">Enter details to predict medical charges</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🧬 Demographics</div>', unsafe_allow_html=True)

    age      = st.slider("age",      min_value=18, max_value=80,   value=35,  step=1)
    bmi      = st.slider("bmi",      min_value=10.0, max_value=55.0, value=27.5, step=0.1,
                         help="Body Mass Index = weight(kg) / height(m)²")
    children = st.slider("children", min_value=0, max_value=10,    value=1)

    st.markdown('<div class="section-header">⚕️ Health & Lifestyle</div>', unsafe_allow_html=True)

    sex    = st.radio("sex",    options=["female", "male"], horizontal=True)
    smoker = st.radio("smoker", options=["no", "yes"],      horizontal=True)

    st.markdown('<div class="section-header">🌍 Region</div>', unsafe_allow_html=True)

    region = st.selectbox(
        "region",
        options=["northeast", "northwest", "southeast", "southwest"],
        index=0,
    )

    st.markdown("---")
    st.button("🔮  Predict Charges", use_container_width=True)


# ─────────────────────────────────────────────
#  ENCODE FEATURES
#  Model expects: age, sex(0/1), bmi, children,
#  smoker(0/1), region_northwest, region_southeast, region_southwest
# ─────────────────────────────────────────────
sex_val    = 1 if sex    == "male" else 0
smoker_val = 1 if smoker == "yes"  else 0
reg_nw     = 1 if region == "northwest"  else 0
reg_se     = 1 if region == "southeast"  else 0
reg_sw     = 1 if region == "southwest"  else 0

features = np.array([[age, sex_val, bmi, children, smoker_val, reg_nw, reg_se, reg_sw]])


# ─────────────────────────────────────────────
#  PREDICTION (live — updates on every widget change)
# ─────────────────────────────────────────────
prediction = None
if model_loaded:
    prediction = model.predict(features)[0]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def risk_level(cost):
    if cost < 8000:    return "LOW CHARGES",    "risk-low",    "✅"
    elif cost < 20000: return "MEDIUM CHARGES", "risk-medium", "⚠️"
    else:              return "HIGH CHARGES",   "risk-high",   "🚨"

def bmi_category(b):
    if b < 18.5: return "Underweight"
    elif b < 25: return "Normal"
    elif b < 30: return "Overweight"
    else:        return "Obese"


# ─────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero-banner">
    <h1>🏥 MedCost AI</h1>
    <p>AI-powered Medical Charges Predictor — Random Forest Model</p>
</div>
""", unsafe_allow_html=True)


# ── Row 1: Quick stat cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">age</div>
        <div class="value">{age}</div>
        <div class="unit">years old</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">bmi</div>
        <div class="value">{bmi:.1f}</div>
        <div class="unit">{bmi_category(bmi)}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">children</div>
        <div class="value">{children}</div>
        <div class="unit">dependants</div>
    </div>""", unsafe_allow_html=True)

with col4:
    smoker_icon = "🚬" if smoker == "yes" else "🚭"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">smoker</div>
        <div class="value">{smoker_icon}</div>
        <div class="unit">{smoker}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Prediction + Profile & Feature importance
left, right = st.columns([1.2, 1], gap="large")

with left:
    if not model_loaded:
        st.error(f"⚠️ Model could not be loaded: {model_error}")
    elif prediction is not None:
        risk_label, risk_class, risk_icon = risk_level(prediction)
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Medical Charges</div>
            <div class="result-value">₹{prediction:,.0f}</div>
            <div class="result-sub">per year &nbsp;|&nbsp; ₹{prediction/12:,.0f} per month</div>
            <span class="risk-badge {risk_class}">{risk_icon} {risk_label}</span>
        </div>
        """, unsafe_allow_html=True)

        insight_extra = (
            "🚨 Smoking contributes ~68% of charge variation — the biggest single factor."
            if smoker == "yes"
            else "💡 Quitting smoking could significantly cut your medical charges."
            if age > 30 else ""
        )
        st.markdown(f"""
        <div class="info-box">
            💡 <b>Insight:</b> A {age}-year-old {sex} {'smoker' if smoker=='yes' else 'non-smoker'}
            with BMI {bmi:.1f} ({bmi_category(bmi)}) in the {region} region
            has estimated charges of <b>₹{prediction:,.0f}/year</b>. {insight_extra}
        </div>
        """, unsafe_allow_html=True)

with right:
    # Profile pills
    st.markdown('<div class="section-header">📋 Patient Profile</div>', unsafe_allow_html=True)
    pills_html = ""
    for lbl, val in [
        ("🧑 sex",      sex),
        ("📅 age",      f"{age} yrs"),
        ("⚖️ bmi",      f"{bmi:.1f} — {bmi_category(bmi)}"),
        ("👶 children", str(children)),
        ("🚬 smoker",   smoker),
        ("🌍 region",   region),
    ]:
        pills_html += f'<span class="profile-pill">{lbl}: {val}</span> '
    st.markdown(pills_html, unsafe_allow_html=True)

    # Feature importance bars
    st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)
    if model_loaded:
        feat_names  = ["age", "sex", "bmi", "children", "smoker",
                       "region_northwest", "region_southeast", "region_southwest"]
        importances = model.feature_importances_
        fi_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
                   .sort_values("importance", ascending=False))
        max_imp = fi_df["importance"].max()
        icons = {"smoker": "🚬", "bmi": "⚖️", "age": "📅", "children": "👶",
                 "sex": "🧑", "region_northwest": "🌍",
                 "region_southeast": "🌍", "region_southwest": "🌍"}

        for _, row in fi_df.iterrows():
            pct  = (row["importance"] / max_imp) * 100
            icon = icons.get(row["feature"], "•")
            disp = row["feature"].replace("_", " ").title()
            st.markdown(f"""
            <div class="feat-bar-wrap">
                <div class="feat-label">{icon} {disp} &nbsp;
                    <span style="color:#f59e0b">{row['importance']:.3f}</span>
                </div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{pct:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 3: What-If Scenario Comparison
st.markdown('<div class="section-header" style="font-size:1rem;margin-bottom:1rem;">🔬 What-If Scenario Comparison</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
scenarios = [
    ("Current Profile", [age, sex_val, bmi,  children, smoker_val, reg_nw, reg_se, reg_sw]),
    ("If smoker = yes", [age, sex_val, bmi,  children, 1,          reg_nw, reg_se, reg_sw]),
    ("If bmi = 35",     [age, sex_val, 35.0, children, smoker_val, reg_nw, reg_se, reg_sw]),
]
colours = ["#7c3aed", "#db2777", "#f59e0b"]

for col, (label, feat), colour in zip([c1, c2, c3], scenarios, colours):
    if model_loaded:
        val       = model.predict(np.array([feat]))[0]
        delta     = val - prediction if label != "Current Profile" else 0
        delta_str = f"+₹{delta:,.0f}" if delta > 0 else (f"-₹{abs(delta):,.0f}" if delta < 0 else "—")
        d_color   = "#fca5a5" if delta > 0 else ("#6ee7b7" if delta < 0 else "#a78bfa")
        with col:
            st.markdown(f"""
            <div class="metric-card"
                 style="border-color:{colour}40; background:linear-gradient(135deg,{colour}18,transparent);">
                <div class="label" style="color:{colour}">{label}</div>
                <div class="value">₹{val:,.0f}</div>
                <div class="unit" style="color:{d_color};font-weight:600;">{delta_str} vs current</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    MedCost AI &nbsp;•&nbsp; Powered by Random Forest Regressor &nbsp;•&nbsp; Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)