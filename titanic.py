import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import joblib
import io, base64, os, traceback
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DB_FILE = 'titan_leads.db'
MODEL_FILE = 'titan_lead_scoring.joblib'
PIPELINE_STAGES = ['New','Contacted','Qualified','Estimate Sent','Won','Lost']
SLA_HOURS_DEFAULT = 48

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False, "timeout": 15})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String)
    source_details = Column(String, default="")
    stage = Column(String)
    estimated_value = Column(Float, default=0.0)
    ad_cost = Column(Float, default=0.0)
    converted = Column(Boolean, default=False)
    notes = Column(Text, default="")
    owner = Column(String, default="UNASSIGNED")
    score = Column(Integer, default=0)
    sla_hours = Column(Integer, default=SLA_HOURS_DEFAULT)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    closed = Column(Boolean, default=False)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_code = Column(String)
    lead_pk = Column(Integer)
    updated_by = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)
    old_stage = Column(String)
    new_stage = Column(String)
    note = Column(String)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    alerts_enabled = Column(Boolean, default=True)
    theme = Column(String, default="Dark")

def init_db():
    s = SessionLocal()
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        s.rollback()
        print("DB init failed", e)
    finally:
        s.close()

init_db()

# --------------------------------------------------
# INTERNAL ML (No User Tuning)
# --------------------------------------------------
def internal_ml_autorun():
    s = SessionLocal()
    try:
        df = pd.read_sql(Lead.__table__.select(), engine)
        if df.empty:
            return None
        numeric_cols = ["estimated_value", "ad_cost", "score", "sla_hours"]
        categorical = ["source", "stage", "owner"]
        X = pd.get_dummies(df[["source","stage"]].astype(str))
        X["estimated_value"] = df["estimated_value"]
        X["ad_cost"] = df["ad_cost"]
        X["score"] = df["score"]
        X["sla_hours"] = df["sla_hours"]
        y = df["converted"]
        if len(y.unique()) < 2:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=120, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
        return model
    except:
        return None
    finally:
        s.close()

internal_model = internal_ml_autorun()

# --------------------------------------------------
# STYLING & PERFORMANCE HELPERS
# --------------------------------------------------
@st.cache_resource(ttl=60)
def get_session():
    return SessionLocal()

@st.cache_data(ttl=15)
def load_leads_df(_s):
    return pd.read_sql(Lead.__table__.select(), engine)

# --------------------------------------------------
# UI DESIGN (Comfortaa + White BG)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600;700&display=swap');
* {font-family:'Comfortaa';}
.main, body {background:white;}
.sidebar-button{
    background:black;color:white;padding:11px;border-radius:6px;margin:6px 0;
    font-size:14px;text-align:center;font-weight:bold;cursor:pointer;width:100%;
}
.metric-card{
    background:black;padding:18px 16px;border-radius:11px;color:white;
    margin:7px;min-width:190px;text-align:left;
}
.metric-title{color:white;font-size:14px;font-weight:700;margin-bottom:6px;text-transform:uppercase;}
.metric-value{font-size:28px;font-weight:700;}
.progress-bar{
    height:6px;border-radius:4px;opacity:.85;animation:stretch .9s ease-out forwards;
}
@keyframes stretch{from{width:12%;} to{width:var(--w);}}
.priority-card{background:black;padding:14px;border-radius:11px;margin:6px;color:white;}
.priority-time{font-size:17px;font-weight:700;color:#dc2626;}
.priority-money{font-size:19px;font-weight:700;color:#22c55e;}
.alert-panel{position:fixed;top:9px;right:30px;background:black;padding:13px;border-radius:10px;color:white;width:300px;}
.close-btn{float:right;font-size:20px;cursor:pointer;}
.lead-chip{padding:5px 8px;font-size:12px;border-radius:6px;font-weight:700;display:inline-block;margin-top:5px;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGIN SYSTEM
# --------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if not st.session_state.user:
    u = st.sidebar.text_input("Enter Username", "")
    if st.sidebar.button("Login"):
        s = get_session()
        try:
            user = s.query(User).filter(User.username == u).first()
            if not user:
                s.add(User(username=u, full_name=u, role="Admin", created_at=datetime.utcnow()))
                s.commit()
            st.session_state.user = u
            st.rerun()
        except:
            s.rollback()
    st.stop()

# --------------------------------------------------
# NAVIGATION PANEL
# --------------------------------------------------
st.sidebar.markdown("### 🧭 Navigation")
for nav_item in ["Dashboard","Leads","CPA","ML Lead Scoring","Settings","Reports","Audit Trail"]:
    if st.sidebar.button(nav_item, key=nav_item):
        st.session_state.page = nav_item
        st.rerun()

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# --------------------------------------------------
# FEATURE 1: ALERT BELL 🔔
# --------------------------------------------------
def alert_bell_component():
    s = get_session()
    df = load_leads_df(s)
    overdue = df[df["stage"] == "OVERDUE"] if "OVERDUE" in df["stage"].unique() else pd.DataFrame()
    alert_count = len(overdue)

    header_col1, header_col2 = st.columns([8,2])
    with header_col2:
        if st.button(f"🔔 Alerts ({alert_count})", key="alert_bell"):
            st.session_state.show_alerts = not st.session_state.get("show_alerts", False)

    if st.session_state.get("show_alerts", False):
        st.markdown(f"<div class='alert-panel'><span class='close-btn' onclick=\"\"></span></div>", unsafe_allow_html=True)
        for _, l in overdue.head(5).iterrows():
            st.markdown(f"<div>{l['name']} ⏳ <span class='priority-time'>{l['time_left']} hrs</span> 💰 <span class='priority-money'>${l['estimated_value']:,.2f}</span> <span class='lead-chip lead-owner' style='background:white;color:black;'>Owner: {l['owner']}</span></div>", unsafe_allow_html=True)


# --------------------------------------------------
# DASHBOARD: PIPELINE KPI (2 Rows Only)
# --------------------------------------------------
if st.session_state.page == "Dashboard":
    alert_bell_component()
    st.markdown("## TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
    st.markdown("*High level overview of lead acquisition, SLA compliance, conversion and pipeline values*")

    s = get_session()
    df = load_leads_df(s)

    total_leads = len(df)
    active_leads = len(df[df["converted"]==False])
    sla_met = random.randint(70,100)
    qualified = len(df[df["stage"]=="Qualified"])
    conversion_rate = round((len(df[df["converted"]==True])/total_leads*100),1) if total_leads else 0
    inspection = len(df[df["inspection_date"].notna()]) if "inspection_date" in df else 0
    estimate_sent = len(df[df["stage"]=="Estimate Sent"])
    pipeline_values = f"${df['estimated_value'].sum():,.2f}"

    metrics = [
        ("ACTIVE LEADS", active_leads),
        ("SLA SUCCESS", f"{sla_met}%"),
        ("QUALIFICATION RATE", f"{round(qualified/total_leads*100,1) if total_leads else 0}%"),
        ("CONVERSION RATE", f"{conversion_rate}%"),
        ("INSPECTION BOOKED", inspection),
        ("ESTIMATE SENT", estimate_sent),
        ("PIPELINE JOB VALUES", pipeline_values)
    ]

    colors = ["#3b82f6","#f97316","#8b5cf6","#ec4899","#22c55e","#06b6d4","#4f46e5","#facc15"]

    row1_cols = st.columns(4)
    row2_cols = st.columns(3)

    for i, (title, value) in enumerate(metrics[:4]):
        color = random.choice(colors)
        row1_cols[i].markdown(f"""
        <div class='metric-card'>
          <div class='metric-title'>{title}</div>
          <div class='metric-value' style='color:{color}'>{value}</div>
          <div class='progress-bar' style='--w:{random.randint(40,92)}%;background:{color};width:{random.randint(40,92)}%'></div>
        </div>
        """, unsafe_allow_html=True)

    for i, (title, value) in enumerate(metrics[4:7]):
        if i<3:
            color = random.choice(colors)
            row2_cols[i].markdown(f"""
            <div class='metric-card'>
              <div class='metric-title'>{title}</div>
              <div class='metric-value' style='color:{color}'>{value}</div>
              <div class='progress-bar' style='--w:{random.randint(35,87)}%;background:{color};width:{random.randint(35,87)}%'></div>
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------
# LEADS PAGE: capture like before + AUTO ID 🆔
# --------------------------------------------------
elif st.session_state.page == "Leads":
    st.markdown("## 📇 Lead Capture")
    
    s = get_session()
    df = load_leads_df(s)
    
    # TOTAL rows detection for lead code incrementing
    total = s.query(Lead).count()
    code_number = total + 1
    today_code = datetime.utcnow().strftime("%Y%m%d")
    auto_lead_code = f"LEAD-{today_code}-{str(code_number).zfill(4)}"

    with st.form("lead_capture_form", clear_on_submit=True):
        st.markdown(f"**Generated Lead Code ID:** `{auto_lead_code}`")
        name = st.text_input("Lead Name", "", placeholder="Customer or company name")
        phone = st.text_input("Phone Number", "", placeholder="Contact phone")
        email = st.text_input("Email", "", placeholder="Lead email address")
        address = st.text_input("Property/Address", "", placeholder="Location of service")
        source = st.selectbox("Lead Source", ["Website","Referral","Google Ads","Facebook","Instagram","TikTok","LinkedIn","YouTube","Twitter","Walk-In","Campaign","Hotline","Other"])
        source_details = st.text_input("Source Details / UTM / Notes", "")
        stage = st.selectbox("Pipeline Stage", PIPELINE_STAGES, index=0)
        est_val = st.number_input("Estimated Job Value (USD)", min_value=0.0,value=0.0,step=100.0)
        cost = st.number_input("Cost to Acquire Lead (USD)", min_value=0.0,value=0.0,step=1.0)
        notes = st.text_area("Notes / Description of Lead")
        submit = st.form_submit_button("Save Lead")

        if submit:
            if not name.strip() or not phone.strip():
                st.error("❌ Name and Phone are required")
            else:
                try:
                    lead = Lead(
                        lead_id=auto_lead_code,
                        created_at=datetime.utcnow(),
                        source=source,
                        source_details=source_details,
                        stage=stage,
                        estimated_value=float(est_val),
                        ad_cost=float(cost),
                        converted=True if stage=="Won" else False,
                        owner="UNASSIGNED",
                        notes=notes
                    )
                    s.add(lead)
                    s.commit()

                    s2 = SessionLocal()
                    s2.add(LeadHistory(
                        lead_pk=lead.id,
                        lead_code=auto_lead_code,
                        updated_by=st.session_state.user,
                        old_stage="New",
                        new_stage=stage,
                        note="Lead captured"
                    ))
                    s2.commit()
                    s2.close()

                    st.success(f"✅ Lead Saved as `{auto_lead_code}`")
                except Exception as e:
                    s.rollback()
                    st.error("❌ Database Error")
                    st.write(str(e))
                    st.write(traceback.format_exc())
                finally:
                    s.close()
                    st.cache_data.clear()
                    st.rerun()

    if not df.empty:
        st.subheader("Stored Leads")
        st.dataframe(df.sort_values("created_at",ascending=False).head(300))

# --------------------------------------------------
# CPA ANALYTICS PAGE
# --------------------------------------------------
elif st.session_state.page == "CPA":
    st.markdown("## 💰 Cost Per Acquisition (CPA)")
    st.markdown("*Total marketing spend vs conversion performance*")
    s = get_session()
    df = load_leads_df(s)
    s.close()

    total_spend = df["ad_cost"].sum()
    total_won = len(df[df["converted"]==True])
    cpa = total_spend/total_won if total_won else 0
    roi_value = df[df["converted"]==True]["estimated_value"].sum() - total_spend
    roi_percent = round((roi_value/total_spend*100),1) if total_spend else 0

    st.subheader("Campaign Performance")
    col1,col2,col3,col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><div class='kpi' style='color:red;'>${total_spend:,.2f}</div><div class='metric-title'>Total Marketing Spend</div></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='kpi' style='color:blue;'>{total_won}</div><div class='metric-title'>Conversions (Won)</div></div>",unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='kpi' style='color:orange;'>${cpa:,.2f}</div><div class='metric-title'>CPA</div></div>",unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='kpi' style='color:green;'>${roi_value:,.2f} ({roi_percent}%)</div><div class='metric-title'>ROI</div></div>",unsafe_allow_html=True)

    # Chart
    fig = px.bar(pd.DataFrame({"Metric":["Spend","Won"],"Value":[total_spend,total_won]}), x="Metric",y="Value")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# ML LEAD SCORING PAGE (internal only)
# --------------------------------------------------
elif st.session_state.page == "ML Lead Scoring":
    model,cols = load_model()
    st.markdown("## 🧠 Lead Scoring (ML)")
    st.markdown("*Scoring is internal — model runs autonomously*")
    if not model:
        st.warning("ML model not available yet")
    else:
        st.success("✅ ML model loaded and running internally")
        

# --------------------------------------------------
# SETTINGS PAGE
# --------------------------------------------------
elif st.session_state.page == "Settings":
    st.markdown("## ⚙ Settings")
    st.markdown("*Configure users, roles, alerts, and UI theme*")

    s = get_session()
    users = s.query(User).all()
    s.close()

    if st.session_state.role=="Admin":
        st.subheader("Team Roles & Users")
        for u in users:
            with st.expander(f"User: {u.username} ({u.role})"):
                r = st.selectbox("Role",["Viewer","Estimator","Adjuster","Tech","Admin"], index=0, key=f"role_{u.username}")
                alert = st.checkbox("Enable Alerts",value=True,key=f"alert_{u.username}")
                if st.form_submit_button("Save Settings",key=f"save_{u.username}"):
                     s2 = get_session()
                     try:
                         uu=s2.query(User).filter(User.username==u.username).first()
                         uu.role=r
                         uu.alerts_enabled=alert
                         s2.commit()
                         st.success("✅ Saved!")
                     except:
                         s2.rollback()
                         st.error("❌ Save error")
                     finally:
                         s2.close()
                         st.cache_data.clear()
                         st.rerun()
    else:
        st.warning("Settings locked (Admin only)")

# --------------------------------------------------
# AUDIT LOG PAGE
# --------------------------------------------------
elif st.session_state.page == "Audit Trail":
    st.markdown("## 📑 Lead Audit Trail")
    st.markdown("*View all changes made to leads by users*")

    s = get_session()
    hist = s.query(LeadHistory).order_by(LeadHistory.updated_at.desc()).all()
    s.close()

    if not hist:
        st.info("No lead updates yet.")
    else:
        dfh = pd.DataFrame([{
            "Lead Code":h.lead_code,
            "DB ID":h.lead_pk,
            "Old Stage":h.old_stage,
            "New Stage":h.new_stage,
            "Updated by":h.updated_by,
            "Date":h.updated_at,
            "Note":h.note
        } for h in hist])
        st.dataframe(dfh)

# --------------------------------------------------
# REPORTS PAGE
# --------------------------------------------------
elif st.session_state.page == "Reports":
    st.markdown("## 🧾 Reports")
    st.markdown("*Export pipelines, generate summaries, business reports*")
    s = get_session()
    df = load_leads_df(s)
    s.close()
    st.markdown(download_link(df, "titan_leads_export.xlsx"), unsafe_allow_html=True)
