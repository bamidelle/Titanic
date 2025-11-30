import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ---------- DATABASE ----------
engine = create_engine("sqlite:///titan_restoration.db", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# ---------- TABLES ----------
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

# ---------- INTERNAL ML (Silent) ----------
MODEL_FILE = "internal_ml_model.pkl"
def internal_ml_train():
    s = SessionLocal()
    leads = s.query(Lead).all()
    s.close()

    if not leads:
        return None

    df = pd.DataFrame([{
        "name": l.name,
        "source": l.source,
        "spend": l.cost_to_acquire,
        "value": l.estimate_value,
        "converted": 1 if l.converted else 0
    } for l in leads])

    X = df.drop("converted", axis=1)
    y = df["converted"]

    numeric_cols = ["spend","value"]
    categorical_cols = ["source"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    model = make_pipeline(pre, LogisticRegression(max_iter=1000))

    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

@st.cache_resource
def load_internal_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    m = internal_ml_train()
    return m

# ---------- LOGIN ----------
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = "Viewer"
    st.session_state.page = "pipeline"

if not st.session_state.user:
    user = st.sidebar.text_input("Username")
    if st.sidebar.button("Login"):
        s = SessionLocal()
        u = s.query(User).filter(User.username==user).first()
        if not u:
            u = User(username=user, full_name="", role="Viewer")
            s.add(u)
            s.commit()
        st.session_state.user = u.username
        st.session_state.role = u.role
        st.session_state.page = "pipeline"
        s.close()
        st.rerun()
    else:
        st.info("👈 Login to continue")
        st.stop()

# Logout
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ---------- DATE FILTER ----------
st.markdown("### 📅 Lead Timeline")
start_date = st.date_input("Start Date", date.today())
end_date = st.date_input("End Date", date.today())

@st.cache_data(ttl=15)
def get_leads_by_date():
    s = SessionLocal()
    leads = s.query(Lead).filter(
        Lead.created_at >= datetime.combine(start_date,datetime.min.time()),
        Lead.created_at <= datetime.combine(end_date,datetime.max.time())
    ).all()
    s.close()
    return leads

# ---------- NAVIGATION ----------
MENU = {
    "📍 Pipeline": "pipeline",
    "📊 Analytics": "analytics",
    "💰 CPA/ROI": "cpa",
    "⚙ Settings": "settings",
    "👤 Profile": "profile"
}

st.sidebar.markdown("### Navigate")
for label, page in MENU.items():
    btn = f"<div class='sidebar-button'>{label}</div>"
    if st.sidebar.button(label, key=f"nav_{page}"):
        st.session_state.page = page
        st.rerun()

# ---------- PIPELINE ----------
if st.session_state.page == "pipeline":
    leads = get_leads_by_date()

    # KPI
    active = sum(1 for l in leads if not l.converted)
    sla_success = random.randint(72,100)
    qualification_rate = random.uniform(45,92)
    conversion_rate = random.uniform(8,34)
    inspection_booked = sum(1 for l in leads if l.inspection_date)
    estimate_sent = sum(1 for l in leads if l.status=="ESTIMATE_SENT")
    pipeline_val = sum(l.estimate_value for l in leads)
    overdue = [l for l in leads if l.status=="OVERDUE"]

    metrics = [
        ("ACTIVE LEADS", active),
        ("SLA SUCCESS", f"{sla_success}%"),
        ("QUALIFICATION RATE", f"{round(qualification_rate,1)}%"),
        ("CONVERSION RATE", f"{round(conversion_rate,1)}%"),
        ("INSPECTION BOOKED", inspection_booked),
        ("ESTIMATE SENT", estimate_sent),
        ("PIPELINE JOB VALUES", f"${pipeline_val:,.2f}")
    ]

    row1 = st.columns(4)
    row2 = st.columns(3)
    stage_colors = ["cyan","green","orange","blue","purple","pink","yellow"]

    for col,(title,val) in zip(row1+row2, metrics):
        color = random.choice(["#06b6d4","#22c55e","#f97316","#3b82f6","#8b5cf6","#ec4899","#facc15"])
        pct = random.randint(20,90)
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-title'>{title}</div>
          <div class='metric-value' style='color:{color};'>{val}</div>
          <div class='progress-bar' style='background:{random.choice(stage_colors)}; width:{pct}%'></div>
        </div>
        """, unsafe_allow_html=True)

    # SLA BELL
    st.markdown(f"<div class='alert-panel' id='alert_panel'><span class='metric-title'>🚨 SLA OVERDUE ({len(overdue)})</span><span class='close-btn' onclick=\"document.getElementById('alert_panel').style.display='none'\">✖</span></div>", unsafe_allow_html=True)

    if overdue:
        for l in overdue[:5]:
            st.markdown(f"<span class='lead-chip' style='background:black;color:white;'>Lead {l.id}: {l.name} | <span class='priority-time'>{l.time_left} hrs</span> | <span class='priority-money'>${l.estimate_value:,.2f}</span></span>", unsafe_allow_html=True)

    # Top 5 priority
    st.markdown("---")
    st.markdown("### TOP 5 PRIORITY LEADS")
    dfp = pd.DataFrame([{"Name":l.name,"Value":l.estimate_value,"Time":l.time_left} for l in leads]).sort_values("Value",ascending=False).head(5)

    pcols = st.columns(5)
    for c,(_,l) in zip(pcols, dfp.iterrows()):
        c.markdown(f"""
        <div class='metric-card'>
        <div class='metric-title'>{l["Name"]}</div>
        <span class='priority-money'>${l["Value"]:,.2f}</span><br>
        <span class='priority-time'>{l["Time"]} hrs left</span>
        </div>
        """, unsafe_allow_html=True)

    # All Leads expand/edit
    st.markdown("---")
    st.markdown("### ALL LEADS")
    for l in leads:
        with st.expander(f"Lead #{l.id} — {l.name}"):
            st.selectbox("Owner",["ADMIN","ESTIMATOR","ADJUSTER","TECH","UNASSIGNED"], key=f"own_{l.id}")
            st.selectbox("Stage",["CAPTURED","QUALIFIED","INSPECTED","ESTIMATE_SENT","AWARDED","OVERDUE"], key=f"stg_{l.id}")
            st.number_input("Spend ($)", l.cost_to_acquire, key=f"sp_{l.id}")
            st.number_input("Value ($)", l.estimate_value, key=f"val_{l.id}")

            if st.button("Save", key=f"bsv_{l.id}"):
                s2=SessionLocal()
                d=s2.query(Lead).filter(Lead.id==l.id).first()
                old=d.status
                d.owner=st.session_state[f"own_{l.id}"]
                d.status=st.session_state[f"stg_{l.id}"]
                d.cost_to_acquire=st.session_state[f"sp_{l.id}"] or 0
                d.estimate_value=st.session_state[f"val_{l.id}"] or 0
                d.converted=True if d.status=="AWARDED" else False
                s2.add(LeadHistory(lead_id=l.id, updated_by=st.session_state.user, old_status=old, new_status=d.status))
                s2.commit()
                s2.close()
                st.success("✅ Saved")

# ---------- ANALYTICS ----------
elif st.session_state.page == "analytics":
    leads = get_leads_by_date()
    df = pd.DataFrame([{ "Lead":l.name, "Stage":l.status, "Owner":l.owner, "Spend":l.cost_to_acquire, "Job Value":l.estimate_value, "Time Left":l.time_left } for l in leads])
    st.dataframe(df)

    fig = plt.figure()
    for s in df["Stage"].unique():
        stage_df = df[df["Stage"]==s]
        plt.plot(stage_df["Time Left"], stage_df["Job Value"], label=s)
    st.pyplot(fig)

# ---------- CPA/ROI ----------
elif st.session_state.page == "cpa":
    leads = get_leads_by_date()
    spend = sum(l.cost_to_acquire for l in leads)
    won = sum(1 for l in leads if l.converted)
    cpa = spend/won if won else 0
    roi = sum(l.estimate_value for l in leads) - spend
    roi_pct = (roi/spend*100) if spend else 0

    st.write("### 💰 Total Marketing Spend", f"${spend:,.2f}")
    st.write("### ✅ Conversions (Won)", f"{won}")
    st.markdown(f"<span style='font-size:18px; color:red;'>🎯 CPA: ${cpa:,.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size:18px; color:green;'>📈 ROI: ${roi:,.2f} ({roi_pct:,.1f}%)</span>", unsafe_allow_html=True)

    fig = plt.figure()
    plt.plot(["Spend","Won"], [spend,won])
    st.pyplot(fig)

# ---------- SETTINGS ----------
elif st.session_state.page == "settings":
    st.markdown("## ⚙ Settings")
    st.checkbox("Enable Alerts", True, key="set_alerts")
    st.selectbox("Default Role",["Viewer","Estimator","Adjuster","Tech","Admin"], key="def_role")

# ---------- PROFILE ----------
elif st.session_state.page == "profile":
    st.markdown("## 👤 User Profile")
    st.text_input("Full Name", st.session_state.user, key="pname")
    st.selectbox("Role",["Viewer","Estimator","Adjuster","Tech","Admin"], index=0, key="prole")
    st.checkbox("Receive Alerts", True, key="p_alert")

    if st.button("Save Profile"):
        s=SessionLocal()
        u=s.query(User).filter(User.username==st.session_state.user).first()
        u.full_name=st.session_state.pname
        u.role=st.session_state.prole
        u.alerts_enabled=st.session_state.p_alert
        s.commit()
        s.close()
        st.success("✅ Profile saved")
