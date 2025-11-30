# titanic.py — FINAL FIXED MERGED CODE

import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import joblib
import os, io, base64, traceback
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------
# DATABASE SETUP (SQLite)
# --------------------------
DB_FILE = "titan.db"
DB_PATH = os.path.dirname(os.path.abspath(__file__))
DB_FULL_PATH = os.path.join(DB_PATH, DB_FILE)
DB_URL = f"sqlite:///{DB_FULL_PATH}"

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Direct")
    stage = Column(String, default="New")
    estimated_value = Column(Float, default=0.0)
    ad_cost = Column(Float, default=0.0)
    converted = Column(Boolean, default=False)
    notes = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    owner = relationship("User", foreign_keys=[owner_id])
    score = Column(Float, default=None)
    sla_hours = Column(Integer, default=72)
    sla_entered_at = Column(DateTime, default=None)
    created_by = Column(String, default=None)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String)
    changed_by = Column(String)
    field = Column(String)
    old_value = Column(String)
    new_value = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables if not exists
Base.metadata.create_all(engine)

# --------------------------
# SAFE SCHEMA MIGRATION
# --------------------------
def migrate_db():
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("leads")] if "leads" in inspector.get_table_names() else []
    needed = {"stage": "TEXT", "estimated_value": "FLOAT", "ad_cost": "FLOAT", "converted": "BOOLEAN", "owner_id": "INTEGER", "notes":"TEXT"}
    conn = engine.connect()
    for c, t in needed.items():
        if c not in cols:
            try:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {c} {t}")
            except:
                pass
    conn.close()

migrate_db()

# --------------------------
# DATABASE HELPERS
# --------------------------
def get_session():
    return SessionLocal()

def get_leads_df(start_date=None, end_date=None):
    s = get_session()
    try:
        leads = s.query(Lead).all()
        data = []
        for L in leads:
            data.append({
                "lead_id": L.lead_id,
                "created_at": L.created_at,
                "source": L.source,
                "stage": L.stage,
                "estimated_value": float(L.estimated_value or 0),
                "ad_cost": float(L.ad_cost or 0),
                "converted": bool(L.converted),
                "owner": L.owner.username if L.owner else None,
                "sla_hours": L.sla_hours or 72,
                "sla_entered_at": L.sla_entered_at,
                "score": L.score,
                "notes": L.notes,
                "created_by": L.created_by
            })
        df = pd.DataFrame(data)

        if df.empty:
            # inject safe defaults
            for col in ["created_at","converted","estimated_value","ad_cost","score","stage","source","owner"]:
                if col not in df.columns:
                    df[col] = np.nan

        # date filters
        if start_date:
            df = df[df["created_at"] >= datetime.combine(start_date, datetime.min.time())]
        if end_date:
            df = df[df["created_at"] <= datetime.combine(end_date, datetime.max.time())]

        return df.reset_index(drop=True)
    finally:
        s.close()

def upsert_lead(data, actor=None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == data["lead_id"]).first()
        if not lead:
            lead = Lead(**data)
            s.add(lead)
            s.commit()
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system",
                              field="create", old_value="None", new_value=data.get("stage")))
        else:
            for k,v in data.items():
                old = getattr(lead,k)
                if old != v:
                    setattr(lead,k,v)
                    s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system",
                                      field=k, old_value=str(old), new_value=str(v)))
            s.commit()
        return lead.lead_id
    except Exception as e:
        s.rollback()
        raise e
    finally:
        s.close()

def delete_lead(lead_id, actor=None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if not lead: return False
        s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system",
                          field="delete", old_value=str(lead.stage), new_value="deleted"))
        s.delete(lead)
        s.commit()
        return True
    except Exception as e:
        s.rollback()
        raise e
    finally:
        s.close()

# --------------------------
# SLA & PRIORITY HELPERS
# --------------------------
def compute_time_left(created_at, sla_hours=72):
    if not created_at:
        return 0, False
    if isinstance(created_at,str):
        created_at = datetime.fromisoformat(created_at)
    deadline = created_at + timedelta(hours=sla_hours)
    remaining = deadline - datetime.utcnow()
    return max(0, remaining.total_seconds()/3600), remaining.total_seconds() <= 0

def priority_score(row):
    hours_left, overdue = compute_time_left(row["sla_entered_at"] or row["created_at"], row["sla_hours"])
    urgency = 1 if overdue else (72 - min(hours_left,72))/72
    value_norm = min(1, row["estimated_value"]/5000)
    model_score = 0 if row["score"] is None else row["score"]
    return (urgency * 0.4) + (value_norm * 0.4) + (model_score * 0.2)

# --------------------------
# APP UI SETUP
# --------------------------
st.set_page_config(page_title="TITAN Lead Pipeline", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# Style
st.markdown("""
<style>
body, .stApp { background:#fff; font-family:'Comfortaa', sans-serif; color:#0b1220}
.kpi { padding:12px; border-radius:10px; background:#000; min-width:220px; margin:5px}
.kpi h4 {color:white; font-size:13px; font-weight:800}
.kpi h2 {font-size:26px; font-weight:900}
.top-time{color:#dc2626;font-weight:700}
.top-value{color:#22c55e;font-weight:800}
</style>
""", unsafe_allow_html=True)

# Sidebar Login
with st.sidebar:
    if not st.session_state.user:
        st.subheader("Login")
        name = st.text_input("Username")
        role = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=4)
        if st.button("Access TITAN"):
            if name.strip():
                s=get_session()
                try:
                    u=s.query(User).filter(User.username==name.strip()).first()
                    if not u:
                        u=User(username=name.strip(),role=role)
                        s.add(u); s.commit()
                    st.session_state.user=u.username
                    st.session_state.role=u.role
                    st.success(f"Logged in as {u.username} ({u.role})")
                    st.experimental_rerun()
                finally: s.close()
            else:
                st.warning("Enter username")
    else:
        st.write(f"Hello, **{st.session_state.user}**")
        if st.button("Exit TITAN"):
            st.session_state.user=None
            st.session_state.role=None
            st.experimental_rerun()

    nav = st.radio("Navigate", ["Dashboard","Leads","Analytics","CPA & ROI","ML","Settings","Exports"])

    # Date
    st.subheader("Report range")
    r = st.selectbox("Quick", ["Today","7d","30d","90d","All","Custom"], index=5)
    if r=="Today":
        st.session_state.start_date=date.today()
        st.session_state.end_date=date.today()
    elif r=="7d":
        st.session_state.start_date=date.today()-timedelta(days=6)
        st.session_state.end_date=date.today()
    elif r=="30d":
        st.session_state.start_date=date.today()-timedelta(days=29)
        st.session_state.end_date=date.today()
    elif r=="90d":
        st.session_state.start_date=date.today()-timedelta(days=89)
        st.session_state.end_date=date.today()
    elif r=="Custom":
        sd,ed = st.date_input("Start — End",[date.today()-timedelta(days=29),date.today()])
        st.session_state.start_date=sd
        st.session_state.end_date=ed
    else:
        st.session_state.start_date=None
        st.session_state.end_date=None

# Main dataframe scored if model exists
sd=st.session_state.start_date
ed=st.session_state.end_date
df = get_leads_df(sd, ed)

# If ML model loaded, apply scoring
model, model_cols = None, None
if os.path.exists(MODEL_FILE):
    pkg = joblib.load(MODEL_FILE)
    model, model_cols = pkg["model"], pkg["columns"]
if model:
    df2 = df.copy()
    df2["age_days"]=(datetime.utcnow()-df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str))
    X["ad_cost"]=df2["ad_cost"]
    X["estimated_value"]=df2["estimated_value"]
    X["age_days"]=df2["age_days"]

    for c in model_cols:
        if c not in X.columns:
            X[c]=0
    X=X[model_cols].fillna(0)
    df["score"]=model.predict_proba(X)[:,1]

# --------------------------
# PAGE IMPLEMENTATIONS
# --------------------------
def page_dashboard():
    df=df_merged()
    st.title("TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")

    # SAFE METRIC CALCULATIONS
    awarded_count = int(df[df["stage"] == "Won"].shape[0]) if "stage" in df.columns else 0
    lost_count = int(df[df["stage"] == "Lost"].shape[0]) if "stage" in df.columns else 0
    closed = awarded_count + lost_count

    contacted_count = int(df[df["stage"] == "Contacted"].shape[0]) if "stage" in df.columns else 0
    qualified_count = int(df[df["stage"] == "Qualified"].shape[0]) if "stage" in df.columns else 0
    inspection_booked = int(df[df["stage"] == "Inspection Scheduled"].shape[0]) if "stage" in df.columns else 0
    estimate_sent = int(df[df["stage"] == "Estimate Sent"].shape[0]) if "stage" in df.columns else 0
    pipeline_value = float(df["estimated_value"].sum()) if "estimated_value" in df.columns else 0.0

    # Derived
    sla_pct = (contacted_count/len(df)*100) if len(df) else 0.0
    qual_pct = (qualified_count/len(df)*100) if len(df) else 0.0
    conv_pct = (awarded_count/closed*100) if closed else 0.0
    insp_pct = (inspection_booked/qualified_count*100) if qualified_count else 0.0

    card_data = [
        ("Active Leads", len(df)-(awarded_count+lost_count), "#2563eb", "Still open in pipeline"),
        ("SLA Success", f"{sla_pct:.1f}%", "#0ea5e9", "First response in SLA window"),
        ("Qualification Rate", f"{qual_pct:.1f}%", "#a855f7", "Qualified from total captured"),
        ("Conversion Rate", f"{conv_pct:.1f}%", "#f97316", "Won from closed deals"),
        ("Inspections Booked", f"{insp_pct:.1f}%", "#dc2626", "From qualified leads"),
        ("Estimates Sent", estimate_sent, "#7c3aed", "Estimates delivered"),
        ("Pipeline Job Value", f"${pipeline_value:,.0f}", "#22c55e", "Potential revenue"),
    ]

    # Render 2 rows
    row1 = st.columns(4)
    row2 = st.columns(3)
    all_cols = row1 + row2

    for col,(t,v,c,n) in zip(all_cols,card_data):
        col.markdown(f"<div class='kpi'><h4>{t}</h4><h2 style='color:{c}'>{v}</h2><div class='kpi-note'>{n}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("TOP 5 PRIORITY LEADS")
    if df.empty:
        st.info("No leads")
        return

    df["priority"] = df.apply(lambda r: priority_score(r),axis=1)
    top=df.sort_values("priority",ascending=False).head(5)

    for _,r in top.iterrows():
        hours_left, overdue = compute_time_left(r["sla_entered_at"] or r["created_at"], r["sla_hours"])
        t_html = f"<span class='top-time'>❗ OVERDUE</span>" if overdue else f"<span class='top-time'>{int(hours_left)}h left</span>"
        v_html = f"<span class='top-value'>${r['estimated_value']:,.0f}</span>"
        st.markdown(f"<div class='lead-card'><div style='display:flex;justify-content:space-between'><div><b>#{r['lead_id']}</b><div class='small-muted'>{r['notes'] or ''}</div></div><div style='text-align:right'>{v_html}<br>{t_html}</div></div><small class='small-muted'>Priority: {r['priority']:.2f} · Stage: {r['stage']}</small></div>", unsafe_allow_html=True)

def page_leads():
    st.title("📇 Leads")
    st.subheader("Create / Update Lead")
    with st.form("cap", clear_on_submit=True):
        lead_id=st.text_input("Lead ID")
        source=st.selectbox("Source",["Google Ads","Organic","Referral","Facebook Ads","Direct","Partner","Other"])
        stage=st.selectbox("Stage",PIPELINE_STAGES,index=0)
        val=st.number_input("Estimated value",min_value=0.0,step=100.0)
        cost=st.number_input("Ad cost",min_value=0.0,step=1.0)
        notes=st.text_area("Notes")
        if st.form_submit_button("Save"):
            try:
                upsert_lead({"lead_id":lead_id.strip(),"source":source,"stage":stage,"estimated_value":float(val),"ad_cost":float(cost),"notes":notes,"sla_hours":72,"sla_entered_at":datetime.utcnow(),"created_by":st.session_state.user}, actor=st.session_state.user)
                st.success("Saved")
            except: st.error(traceback.format_exc())
    st.markdown("---")
    if not df.empty:
        st.dataframe(df)

def page_analytics():
    st.title("📈 Analytics & SLA Trend")
    if df.empty:
        st.info("No data")
        return
    fig=px.line(df,x="created_at",y="sla_hours",title="SLA Trend")
    st.plotly_chart(fig)

def page_cpa():
    st.title("💰 CPA & ROI")
    if df.empty: st.info("No data"); return
    spend=df["ad_cost"].sum()
    awarded=df[df["stage"]=="Won"]["estimated_value"].sum()
    c=df[df["stage"]=="Won"].shape[0]
    st.metric("Spend",f"${spend:,.2f}")
    st.metric("Conversions",c)
    st.metric("CPA",f"${(spend/c):,.2f}" if c else "$0")
    st.metric("ROI",f"${awarded-spend:,.2f}")

def page_ml():
    st.title("🧠 Internal ML (Lead Scoring)")
    if st.button("Train Model"):
        try: acc,msg=train_model()
        except: st.error("Training failed"); return
        if acc: st.success(f"Model trained. Accuracy: {acc:.2f}")
        else: st.warning(msg)

    model,model_cols=load_model()
    if not model:
        st.info("Train model first")
        return
    if st.button("Score Leads"):
        s=get_session()
        try:
            scored=score_df_with_model(df.copy(),model,model_cols)
            for _,r in scored.iterrows():
                L=s.query(Lead).filter(Lead.lead_id==r["lead_id"]).first()
                if L:
                    L.score=float(r["score"])
                    s.add(L)
            s.commit()
            st.success("Scores saved")
        except: s.rollback(); st.error(traceback.format_exc())
        finally: s.close()
    st.markdown("---")
    if not df.empty:
        st.dataframe(df.sort_values("score",ascending=False).head(20))

def page_settings():
    st.title("⚙️ Settings")
    all_users = get_session().query(User).all()
    st.write("Users in system:")
    for u in all_users:
        st.write(f"- {u.username} ({u.role})")

def page_export_import():
    st.title("📤 Export / Import")
    if not df.empty:
        st.download_button("Download CSV", df.to_csv(index=False), file_name="titan_leads.csv")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            dfi = pd.read_csv(file)
            if "lead_id" not in dfi.columns:
                st.error("CSV must contain lead_id column")
                return
            for _, r in dfi.iterrows():
                upsert_lead({
                    "lead_id": str(r["lead_id"]),
                    "source": r.get("source"),
                    "stage": r.get("stage"),
                    "estimated_value": float(r.get("estimated_value") or 0.0),
                    "ad_cost": float(r.get("ad_cost") or 0.0),
                    "converted": bool(r.get("converted") or False),
                    "notes": r.get("notes"),
                    "sla_hours": 72,
                    "sla_entered_at": datetime.utcnow(),
                    "created_by": st.session_state.user
                }, actor=st.session_state.user)
            st.success("Imported leads")
            st.experimental_rerun()
        except Exception as e:
            st.error(str(e))

# Router
def df_merged(): 
    return get_leads_df(sd, ed)

if nav=="Dashboard":
    page_dashboard()
elif nav=="Leads":
    page_leads()
elif nav=="Analytics":
    page_analytics()
elif nav=="CPA & ROI":
    page_cpa()
elif nav=="ML":
    page_ml()
elif nav=="Settings":
    page_settings()
elif nav=="Exports":
    page_export_import()

st.markdown("<div class='small-muted'>TITAN Control Panel — safe execution active</div>", unsafe_allow_html=True)
