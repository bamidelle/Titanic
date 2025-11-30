import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
import joblib
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Titan Restoration", layout="wide")

# Inject global styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600&display=swap');
html, body, [class*="css"]  {
    font-family: 'Comfortaa', sans-serif;
    background: white;
}
.metric-card {
    background:#000;
    color:white;
    padding:16px;
    border-radius:12px;
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
    margin-bottom:12px;
}
.metric-title {
    font-size:14px;
    font-weight:600;
    color:white;
}
.metric-value {
    font-size:26px;
    font-weight:600;
    margin-top:6px;
}
.progress-bar {
  height:6px;
  border-radius:4px;
  margin-top:12px;
}
.priority-money {
  font-size:20px;
  font-weight:600;
  color:#22c55e; /* green */
}
.priority-time {
  font-size:15px;
  font-weight:600;
  color:#ef4444; /* red */
}
.submit-btn > button {
    width:100%;
    background:#ef4444;
    border:none;
    padding:14px;
    border-radius:8px;
    font-size:18px;
    font-weight:600;
    color:white;
}
.lead-chip {
  display:inline-block;
  padding:8px 14px;
  background:#000;
  color:#fff;
  border-radius:6px;
  font-size:14px;
  margin:4px 0;
}
.alert-panel {
  background:#000;
  padding:12px 18px;
  border-radius:8px;
  color:white;
  font-size:15px;
  font-weight:600;
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin:16px 0;
}
.close-btn {
  cursor:pointer;
  font-size:18px;
  color:white;
}
.spend-chart-container {
  margin-top:18px;
}
.sidebar-button {
  background:#000;
  color:white;
  padding:10px;
  border-radius:6px;
  text-align:center;
  margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- DATABASE ----------
engine = create_engine("sqlite:///titan_restoration.db", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# ---------- MODELS ----------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    phone = Column(String)
    email = Column(String)
    address = Column(String)
    source = Column(String)
    cost_to_acquire = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    inspection_date = Column(DateTime, nullable=True)
    estimate_value = Column(Float, default=0.0)
    status = Column(String, default="CAPTURED")
    converted = Column(Boolean, default=False)
    owner = Column(String, default="UNASSIGNED")
    score = Column(Integer, default=50)
    time_left = Column(Integer, default=48)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer)
    updated_by = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)
    old_status = Column(String)
    new_status = Column(String)
    note = Column(String)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    alerts_enabled = Column(Boolean, default=True)

Base.metadata.create_all(engine)

# Utility: fetch leads in date range
@st.cache_data(ttl=10)
def get_leads(start_date, end_date):
    s = SessionLocal()
    leads = s.query(Lead).filter(
        Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
        Lead.created_at <= datetime.combine(end_date, datetime.max.time())
    ).all()
    s.close()
    return leads

# Utility: silent internal ML train
MODEL_FILE = "internal_ml_model.pkl"
def silent_train_ml():
    s = SessionLocal()
    leads = s.query(Lead).all()
    s.close()
    if not leads:
        return None
    X = pd.DataFrame([{"spend":l.cost_to_acquire, "value":l.estimate_value, "source":l.source} for l in leads])
    y = [1 if l.converted else 0 for l in leads]

    pre = ColumnTransformer([
        ("num", StandardScaler(), ["spend","value"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["source"])
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=1000))
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

def load_ml_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return silent_train_ml()

# ---------- AUTH SECTION ----------
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = "Viewer"
    st.session_state.page = "pipeline"

if not st.session_state.user:
    u = st.sidebar.text_input("Username")
    if st.sidebar.button("Login"):
        s = SessionLocal()
        x = s.query(User).filter(User.username==u).first()
        if not x:
            x = User(username=u, full_name=u, role="Viewer")
            s.add(x)
            s.commit()
        st.session_state.user = x.username
        st.session_state.role = x.role
        s.close()
        st.rerun()
    else:
        st.stop()

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ---------- PAGE ROUTER ----------
page = st.session_state.page

# ---------- PIPELINE PAGE ----------
if page == "pipeline":
    st.markdown("## 🧾 Lead Pipeline")
    st.markdown("_Overview of lead pipeline performance_")

    st.markdown("### 📅 Select timeline")
    col_a, col_b, col_c = st.columns([1,1,3])
    start = col_a.date_input("Start", date.today())
    end = col_b.date_input("End", date.today())

    leads = get_leads(start, end)

    active_leads = sum(1 for l in leads if not l.converted)
    sla_success = random.randint(80, 100)
    qualification_rate = round(random.uniform(60, 95),1)
    conversion_rate = round(random.uniform(8, 30),1)
    inspection_booked = sum(1 for l in leads if l.inspection_date)
    estimate_sent = sum(1 for l in leads if l.status=="ESTIMATE_SENT")
    pipeline_val = sum(l.estimate_value for l in leads if not l.converted)
    overdue_leads = [l for l in leads if l.status=="OVERDUE"]

    data = [
        ("Active Leads", active_leads),
        ("SLA Success", f"{sla_success}%"),
        ("Qualification", f"{qualification_rate}%"),
        ("Conversion", f"{conversion_rate}%"),
        ("Inspections", inspection_booked),
        ("Estimates Sent", estimate_sent),
        ("Pipeline Value", f"${pipeline_val:,.2f}")
    ]

    df_priority = pd.DataFrame([{"Name":l.name, "Value":l.estimate_value, "Time":l.time_left} for l in overdue_leads])
    df_priority = df_priority.sort_values("Value",ascending=False).head(5)

    # KPI Cards - 2 rows
    rows = st.columns(4) + st.columns(3)
    for c,(title,val) in zip(rows,data):
        num_color = random.choice(["#3b82f6","#22d3ee","#f97316","#facc15","#c084fc","#34d399","#60a5fa"])
        pct = random.randint(30, 90)
        bar_color = random.choice(["cyan","green","orange","blue","purple","pink","yellow"])
        c.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value' style="color:{num_color}">{val}</div>
            <div class='progress-bar' style="background:{bar_color}; width:{pct}%"></div>
        </div>
        """, unsafe_allow_html=True)

    # Top 5 priority section
    st.markdown("### ⭐ TOP 5 PRIORITY LEADS")
    pcols = st.columns(5)
    for c,l in zip(pcols, df_priority.to_dict("records")):
        c.markdown(f"""
        <div class='metric-card'>
            <div style="color:{random.choice(["red","orange","blue","green","purple"])}; font-size:15px; font-weight:600;">{l['Name']}</div>
            <div class='priority-money'>${l['Value']:,.2f}</div>
            <div class='priority-time'>{l['Time']} hrs left</div>
        </div>
        """, unsafe_allow_html=True)

    # All leads expansion
    st.markdown("### 📚 All Leads")
    for l in leads:
        hist = []
        s = SessionLocal()
        h = s.query(LeadHistory).filter(LeadHistory.lead_id==l.id).all()
        for x in h:
            hist.append(f"{x.updated_by} changed {x.old_status} → {x.new_status} at {x.updated_at}")
        s.close()

        with st.expander(f"Lead #{l.id} — {l.name}"):
            st.text(f"Source: {l.source}")
            st.text(f"Spend: ${l.cost_to_acquire:.2f}")
            st.text(f"Value: ${l.estimate_value:.2f}")
            st.text(f"Owner: {l.owner}")
            st.text(f"Score: {l.score}")
            st.write("Audit trail:")
            for x in hist:
                st.text(x)

# ---------- ANALYTICS PAGE ----------
elif page == "analytics":
    st.markdown("## 📊 Analytics")
    leads = get_leads(date.today(),date.today())
    if leads:
        df = pd.DataFrame([{"Name":l.name,"Stage":l.status,"Owner":l.owner,"Spend":l.cost_to_acquire,"Value":l.estimate_value,"Time":l.time_left} for l in leads])
    else:
        df = pd.DataFrame()

    st.markdown("### Pipeline Stages (Donut)")
    if not df.empty:
        fig = px.pie(df, names="Stage", hole=0.6)
        st.plotly_chart(fig, use_container_width=False)

# ---------- CPA / ROI PAGE ----------
elif page == "cpa":
    st.markdown("## 💰 CPA & ROI")
    leads = get_leads(date.today(),date.today())
    spend = sum(l.cost_to_acquire for l in leads)
    won = sum(1 for l in leads if l.converted)
    cpa = spend/won if won else 0
    roi = sum(l.estimate_value for l in leads if l.converted) - spend
    roi_pct = (roi/spend*100) if spend else 0

    c1,c2 = st.columns(2)
    c1.markdown("<div class='metric-card'><div class='metric-title' style='color:red;'>Total Marketing Spend</div><div class='metric-value' style='color:white;'>${:,.2f}</div></div>".format(spend),unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><div class='metric-title' style='color:green;'>Conversions (Won)</div><div class='metric-value' style='color:white;'>{}</div></div>".format(won),unsafe_allow_html=True)

    st.markdown(f"<p style='font-size:20px;'><span style='color:red;'>🎯 CPA:</span> <span style='color:blue;'>${cpa:,.2f}</span></p>",unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><span style='color:orange;'>📈 ROI:</span> <span style='color:green;'>${roi:,.2f} ({roi_pct:,.1f}%)</span></p>",unsafe_allow_html=True)

    st.markdown("### 💹 Spend vs Conversions")
    s_cols = st.columns([1,3,1])
    fig = px.line(x=["Spend","Conversions"], y=[spend,won])
    s_cols[1].plotly_chart(fig, use_container_width=False)

# ---------- SETTINGS ----------
elif page == "settings":
    st.markdown("## ⚙ Settings")
    st.selectbox("Default Role",["Viewer","Estimator","Adjuster","Tech","Admin"])
    st.selectbox("Allowed Sources",["Website","Hotline","Facebook","Instagram","TikTok","Google Ads","Referral"])

# ---------- PROFILE ----------
elif page == "profile":
    st.markdown("## 👤 User Profile")
    st.text_input("Full Name", st.session_state.user)
    st.selectbox("Role",["Viewer","Estimator","Adjuster","Tech","Admin"])
    st.toggle("Alerts enabled", True)
