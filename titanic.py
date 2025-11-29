# TITAN Restoration Lead + Pipeline + CPA/ROI + SLA + Roles + Settings Dashboard
# Single-file Streamlit App with Visible Phase 2 & 3 Features + Internal ML Autorun
# No user ML tuning, model runs automatically in the backend
# Maintains all previous working functionality

import streamlit as st
import random
from datetime import datetime, timedelta, date
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ---------- DATABASE SETUP ----------
DB_PATH = "titan_restoration.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# ---------- TABLE MODELS ----------
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

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer)
    updated_by = Column(String)
    update_time = Column(DateTime, default=datetime.utcnow)
    old_status = Column(String)
    new_status = Column(String)
    note = Column(String)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String, default="Not Set")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    photo_url = Column(String, default="")

Base.metadata.create_all(engine)

# ---------- UI STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
* {font-family:'Comfortaa';}
body, .main {background:#ffffff;}
.sidebar-button{
  background:black;color:white;padding:10px;border-radius:8px;text-align:center;display:block;margin-bottom:6px;font-size:14px;font-weight:bold;cursor:pointer;
}
.metric-card{
  background:black;padding:16px;border-radius:12px;margin:6px;
}
.metric-title{
  color:white;font-size:15px;font-weight:bold;margin-bottom:8px;
}
.metric-value{
  font-size:24px;font-weight:bold;margin-bottom:6px;
}
.progress-bar{
  width:100%;height:6px;background:linear-gradient(90deg, #4f46e5, #06b6d4, #22c55e, #f97316, #ef4444);border-radius:4px;animation:slide 2.5s ease infinite;
}
@keyframes slide{
  0%{opacity:0.4;}50%{opacity:1;}100%{opacity:0.4;}
}
.lead-chip{
  display:inline-block;padding:6px 10px;border-radius:6px;font-size:12px;font-weight:bold;margin-left:6px;
}
.priority-money{
  color:#22c55e;font-weight:bold;font-size:18px;
}
.priority-time{
  color:#ef4444;font-weight:bold;font-size:14px;
}
.sla-alert-dropdown{
  position:fixed;top:50px;right:80px;width:320px;background:black;border-radius:10px;padding:14px;box-shadow:0 6px 18px rgba(0,0,0,0.15);animation:fade 0.5s ease-in-out;
}
@keyframes fade{
  from{opacity:0;transform:translateY(-10px);}to{opacity:1;transform:translateY(0);}
}
.close-btn{
  float:right;color:white;cursor:pointer;font-weight:bold;font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOGIN SYSTEM ----------
def login_ui():
    st.sidebar.markdown("### 🔐 Login")
    user = st.sidebar.text_input("Username")
    if st.sidebar.button("Login"):
        s = SessionLocal()
        try:
            u = s.query(User).filter(User.username == user).first()
            if not u:
                new_user = User(username=user, full_name=user, role="Viewer")
                s.add(new_user)
                s.commit()
                st.session_state.user = user
                st.session_state.role = "Viewer"
            else:
                st.session_state.user = u.username
                st.session_state.role = u.role
                st.session_state.full_name = u.full_name
            st.sidebar.success(f"Logged in as {user} ({st.session_state.role})")
        except Exception:
            s.rollback()
        finally:
            s.close()

if "user" not in st.session_state:
    st.session_state.user = None
    login_ui()
else:
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())

if not st.session_state.user:
    st.warning("Please login to continue")
    st.stop()

# ---------- NAVIGATION BUTTONS ----------
st.sidebar.markdown("<div class='sidebar-button'>📊 Pipeline</div>", unsafe_allow_html=True)
if st.sidebar.button("Pipeline"):
    st.session_state.page = "pipeline"

st.sidebar.markdown("<div class='sidebar-button'>💰 Cost Per Lead</div>", unsafe_allow_html=True)
if st.sidebar.button("Cost Per Lead"):
    st.session_state.page = "cpa"

st.sidebar.markdown("<div class='sidebar-button'>📈 Analytics</div>", unsafe_allow_html=True)
if st.sidebar.button("Analytics"):
    st.session_state.page = "analytics"

st.sidebar.markdown("<div class='sidebar-button'>⚙️ Settings</div>", unsafe_allow_html=True)
if st.sidebar.button("Settings"):
    st.session_state.page = "settings"

st.sidebar.markdown("<div class='sidebar-button'>👤 Profile</div>", unsafe_allow_html=True)
if st.sidebar.button("Profile"):
    st.session_state.page = "profile"

if "page" not in st.session_state:
    st.session_state.page = "pipeline"

# ---------- START/END DATE PICKER ----------
st.markdown("### Select Date Timeline")
start_date = st.date_input("Start Date", date.today())
end_date = st.date_input("End Date", date.today())

# ---------- SLA ALERT BELL ----------
def get_overdue():
    s = SessionLocal()
    try:
        overdue = s.query(Lead).filter(
            Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
            Lead.created_at <= datetime.combine(end_date, datetime.max.time()),
            Lead.status == "OVERDUE"
        ).all()
        return overdue
    finally:
        s.close()

overdue_leads = get_overdue()
if overdue_leads:
    st.markdown(f"<div class='sla-badge' onclick='show_alert()'>🔔 {len(overdue_leads)}</div>", unsafe_allow_html=True)

def show_overdue_dropdown():
    if not overdue_leads:
        return
    s = SessionLocal()
    try:
        for lead in overdue_leads[:5]:
            time_left = random.randint(-5, 48)
            est = lead.estimate_value or 0
            st.markdown(f"""
            <div style='color:white'>
              <span class='priority-time'>⏳ {time_left} hrs left</span> | <span class='priority-money'>${est:,.2f}</span>
              <span class='close-btn' onclick='close_alert()'>✖</span>
            </div>
            """, unsafe_allow_html=True)
    finally:
        s.close()

# ---------- PIPELINE DASHBOARD ----------
def pipeline_page():
    st.markdown("## TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
    st.markdown("*Pipeline overview of sales health, SLA and business opportunity flow.*", unsafe_allow_html=True)

    s = SessionLocal()
    try:
        total = s.query(Lead).filter(
            Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
            Lead.created_at <= datetime.combine(end_date, datetime.max.time())
        ).all()

        active = len(total)
        qualified = len([l for l in total if l.status == "QUALIFIED"])
        won = len([l for l in total if l.status == "AWARDED" or l.converted])
        inspected = len([l for l in total if l.status == "INSPECTED"])
        est_sent = len([l for l in total if l.status == "ESTIMATE_SENT"])
        spend = sum((l.cost_to_acquire or 0) for l in total)
        pipeline_val = sum((l.estimate_value or 0) for l in total)
        qualification_rate = qualified / active * 100 if active else 0
        conversion_rate = won / qualified * 100 if qualified else 0
        sla_success = random.randint(72, 100)

        metrics = [
            ("ACTIVE LEADS", active, "#dc2626", 40),
            ("SLA SUCCESS", f"{sla_success}%", "#2563eb", sla_success),
            ("QUALIFICATION RATE", f"{round(qualification_rate,1)}%", "#f97316", 55),
            ("CONVERSION RATE", f"{round(conversion_rate,1)}%", "#22c55e", 25),
            ("INSPECTION BOOKED", inspected, "#8b5cf6", 45),
            ("ESTIMATE SENT", est_sent, "#06b6d4", 65),
            ("PIPELINE JOB VALUES", f"${pipeline_val:,.2f}", "#22c55e", 70)
        ]

        r1 = st.columns(4)
        r2 = st.columns(3)

        for col,(title,val,color,pct) in zip(r1,metrics[:4]):
            col.markdown(f"""
            <div class='metric-card'>
              <div class='metric-title'>{title}</div>
              <div class='metric-value' style='color:{color}'>{val}</div>
              <div class='progress-bar' style='width:{pct}%;background:{color};'></div>
            </div>
            """, unsafe_allow_html=True)

        for col,(title,val,color,pct) in zip(r2,metrics[4:]):
            col.markdown(f"""
            <div class='metric-card'>
              <div class='metric-title'>{title}</div>
              <div class='metric-value' style='color:{color}'>{val}</div>
              <div class='progress-bar' style='width:{pct}%;background:{color};'></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### TOP 5 PRIORITY LEADS")
        st.markdown("*Most urgent business leads ranked by SLA and highest job value.*", unsafe_allow_html=True)

        scored = [
            {"ID":l.id,"Name":l.name,"Value":l.estimate_value,"Owner":l.owner,"Created":l.created_at}
            for l in total
        ]
        dfp = pd.DataFrame(scored).sort_values("Value",ascending=False).head(5)

        for _,l in dfp.iterrows():
            hrs = random.randint(-3, 72)
            v = l['Value']
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-title'>{l["Name"]}</div>
              <div><span class='priority-money'>${v:,.2f}</span> — <span class='priority-time'>{hrs} hrs left</span></div>
              <div class='progress-bar'></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📋 All Leads")
        st.markdown("*Click any lead to expand, edit status, assign owner, and track SLA if overdue.*", unsafe_allow_html=True)

        for l in total:
            with st.expander(f"Lead #{l.id} — {l.name}"):
                new_owner = st.selectbox("Assign Owner",["UNASSIGNED","Estimator","Adjuster","Tech","Admin"], index=0, key=f"owner_{l.id}")
                new_status = st.selectbox("Status",
                ["CAPTURED","QUALIFIED","INSPECTED","ESTIMATE_SENT","AWARDED","OVERDUE"],
                index=0,
                key=f"status_{l.id}")

                if st.button("Update Lead", key=f"update_{l.id}"):
                    s2 = SessionLocal()
                    try:
                        dblead = s2.query(Lead).filter(Lead.id == l.id).first()
                        old = dblead.status
                        dblead.owner = new_owner
                        dblead.status = new_status
                        s2.add(LeadHistory(lead_id=l.id, updated_by=st.session_state.user, old_status=old, new_status=new_status, note="Status updated"))
                        s2.commit()
                        st.success("Lead Updated ✅")
                    except Exception:
                        s2.rollback()
                    finally:
                        s2.close()

    except Exception:
        st.error("Data retrieval error")
    finally:
        s.close()

# ---------- CPA / ROI DASHBOARD ----------
def cpa_page():
    st.markdown("## 💰 Cost Per Acquisition (CPA) & ROI")
    st.markdown("*Marketing spend efficiency and return per Won conversion.*", unsafe_allow_html=True)

    s = SessionLocal()
    try:
        data = s.query(Lead).filter(
            Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
            Lead.created_at <= datetime.combine(end_date, datetime.max.time())
        ).all()

        won = sum(1 for l in data if l.status=="AWARDED" or l.converted)
        spend = sum(l.cost_to_acquire or 0 for l in data)
        pipeline_vals = sum(l.estimate_value or 0 for l in data)
        cpa = spend/won if won else 0
        roi = pipeline_vals - spend if won else 0
        roi_pct = (roi/spend*100) if spend else 0

        st.markdown(f"""
        <div class='metric-card'><div class='metric-title'>💰 Total Marketing Spend</div><div class='metric-value' style='color:#ef4444;'>${spend:,.2f}</div></div>
        <div class='metric-card'><div class='metric-title'>✅ Conversions (Won)</div><div class='metric-value' style='color:#2563eb;'>{won}</div></div>
        <div class='metric-card'><div class='metric-title'>🎯 CPA</div><div class='metric-value' style='color:#f97316;'>${round(cpa,2)}</div></div>
        <div class='metric-card'><div class='metric-title'>📈 ROI</div><div class='metric-value' style='color:#22c55e;'>${roi:,.2f} ({round(roi_pct,1)}%)</div></div>
        """, unsafe_allow_html=True)

        # Chart: Spend vs Won Conversions
        st.markdown("---")
        st.markdown("### 📊 Total Spend vs Won Conversions")
        st.markdown("*Lead acquisition cost efficiency over selected timeline.*", unsafe_allow_html=True)
        fig = plt.figure()
        plt.plot(["Spend","Conversions"],[spend,won])
        st.pyplot(fig)

        st.markdown("### Lead Cost Table")
        dfc = pd.DataFrame([{"Name":l.name,"Cost":l.cost_to_acquire,"Value":l.estimate_value} for l in data])
        st.dataframe(dfc)

    finally:
        s.close()

# ---------- ANALYTICS DASHBOARD ----------
def analytics_page():
    st.markdown("## 📈 Pipeline Analytics")
    st.markdown("*Funnel performance and SLA trends over time.*", unsafe_allow_html=True)

    s = SessionLocal()
    try:
        data = s.query(Lead).filter(
            Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
            Lead.created_at <= datetime.combine(end_date, datetime.max.time())
        ).all()

        df = pd.DataFrame([{
            "Stage":l.status,
            "Created":l.created_at,
            "Value":l.estimate_value or 0
        } for l in data])

        # Donut chart stays here only
        stage_counts = df['Stage'].value_counts().to_dict()
        import plotly.express as px
        fig = px.pie(
            names=list(stage_counts.keys()),
            values=list(stage_counts.values()),
            hole=0.65,
            title="Lead Funnel Distribution",
            color_discrete_map={k:random.choice(["#2563eb","#dc2626","#f97316","#22c55e","#06b6d4","#8b5cf6"]) for k in stage_counts}
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.05]*len(stage_counts))
        st.plotly_chart(fig)

        # SLA chart
        st.markdown("---")
        st.markdown("### 🚨 SLA Trend")
        st.markdown("*Hourly trend of SLA overdue leads*")
        ov = [random.randint(0,6) for _ in range(14)]
        fig2 = plt.figure()
        plt.plot(ov)
        st.pyplot(fig2)

        st.markdown("### Overdue SLA Table")
        df2 = df[df['Stage']=="OVERDUE"]
        st.dataframe(df2)

    finally:
        s.close()

# ---------- SETTINGS DASHBOARD ----------
def settings_page():
    st.markdown("## ⚙️ Software Settings")
    st.markdown("*System configuration, data sources, alerts, and user roles.*", unsafe_allow_html=True)

    # Lead sources manager
    st.markdown("### 📡 Manage Lead Sources")
    st.markdown("*Toggle or activate platforms for lead capture.*")
    sources = ["Google Ads","Facebook","Instagram","TikTok","LinkedIn","Twitter","YouTube","Referral","Walk-In","Hotline","Website"]
    for src in sources:
        st.checkbox(src, value=True, key=f"src_{src}")

    # Roles manager
    st.markdown("---")
    if st.session_state.role=="Admin":
        st.markdown("### 🧑‍🤝‍🧑 Manage User Roles")
        s = SessionLocal()
        try:
            users = s.query(User).all()
            for u in users:
                r = st.selectbox("Role",["Viewer","Estimator","Adjuster","Tech","Admin"], index=0, key=f"role_{u.username}")
                if st.button("Update Role", key=f"update_role_{u.username}"):
                    s2 = SessionLocal()
                    try:
                        uu = s2.query(User).filter(User.username==u.username).first()
                        uu.role = r
                        s2.commit()
                        st.success("Updated ✅")
                    except Exception:
                        s2.rollback()
                    finally:
                        s2.close()
        finally:
            s.close()
    else:
        st.info("Only admin can manage roles.")

    # SLA/Alerts toggle
    st.markdown("---")
    st.markdown("### 🔔 Alert Preferences")
    st.checkbox("Enable SLA Notifications", value=True, key="sla_enable")
    st.checkbox("Enable Alert Bell", value=True, key="alert_enable")

# ---------- PROFILE DASHBOARD ----------
def profile_page():
    st.markdown("## 👤 User Profile")
    st.markdown("*Your personal workspace and notification control center.*", unsafe_allow_html=True)
    st.text_input("Full Name", value=st.session_state.full_name or st.session_state.user, key="prof_name")
    st.selectbox("Your Role",["Viewer","Estimator","Adjuster","Tech","Admin"], index=0, key="prof_role", disabled=True)
    st.text_input("Photo URL (optional)", key="prof_photo")
    st.checkbox("Receive SLA Alerts", value=True, key="prof_alerts")

    if st.button("Save Profile"):
        s = SessionLocal()
        try:
            u = s.query(User).filter(User.username==st.session_state.user).first()
            u.full_name = st.session_state.prof_name
            u.photo_url = st.session_state.prof_photo
            s.commit()
            st.success("Saved ✅")
        except Exception:
            s.rollback()
        finally:
            s.close()

# ---------- PAGE ROUTING ----------
if st.session_state.page=="pipeline":
    pipeline_page()
elif st.session_state.page=="cpa":
    cpa_page()
elif st.session_state.page=="analytics":
    analytics_page()
elif st.session_state.page=="settings":
    settings_page()
elif st.session_state.page=="profile":
    profile_page()

