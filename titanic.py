# ==========================================
# TITAN - RESTORATION CRM + PIPELINE + SLA + ANALYTICS (SINGLE FILE)
# ==========================================

# ---------------- CONFIG CONSTANTS (MUST BE FIRST) ----------------

DB_FILE = "titan_restoration.db"
MODEL_FILE = "titan_lead_scoring.joblib"
PIPELINE_STAGES = ["New", "Contacted", "Qualified", "Inspected", "Estimate Sent", "Won", "Lost", "Overdue"]
ROLES = ["Admin", "Estimator", "Adjuster", "Tech", "Viewer"]

# KPI colors for display (not CSS styling)
KPI_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f97316", "#dc2626", "#22c55e", "#facc15"]

# SLA threshold hours
DEFAULT_SLA_HOURS = 72

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import sqlite3
import os
import io, base64, traceback
import random
import plotly.express as px
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ---------------- APP SETUP ----------------

st.set_page_config(page_title="TITAN - Lead Management & Pipeline", layout="wide")

# ---------------- DATABASE SETUP ----------------

DB_PATH = os.path.dirname(os.path.abspath(__file__))
DB_FULL_PATH = os.path.join(DB_PATH, DB_FILE)
DB_URL = f"sqlite:///{DB_FULL_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# ---------------- SQLALCHEMY DATA MODELS ----------------

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
    ad_cost = Column(Float, default=0.0)  # Cost to acquire lead
    converted = Column(Boolean, default=False)
    notes = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    owner = relationship("User", foreign_keys=[owner_id])
    score = Column(Float, default=None)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=DEFAULT_SLA_HOURS)
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

# Create tables
Base.metadata.create_all(engine)

# ---------------- SAFE SCHEMA MIGRATION ----------------

def migrate_db():
    insp = inspect(engine)
    if "leads" not in insp.get_table_names():
        return
    existing = [c["name"] for c in insp.get_columns("leads")]
    needed = ["score", "sla_hours", "owner_id", "notes", "converted"]
    conn = engine.connect()
    for col in needed:
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {col} TEXT")
            except:
                pass
    conn.close()

migrate_db()

# ---------------- CRUD OPERATIONS ----------------

def upsert_lead(data, actor=None):
    s = SessionLocal()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == data["lead_id"]).first()
        if not lead:
            lead = Lead(**data)
            s.add(lead)
            s.commit()
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system", field="create", old_value="None", new_value=data["stage"]))
        else:
            for k,v in data.items():
                if getattr(lead,k) != v:
                    s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system", field=k, old_value=str(getattr(lead,k)), new_value=str(v)))
                    setattr(lead,k,v)
            s.commit()
        return True
    except Exception as e:
        s.rollback()
        raise e
    finally:
        s.close()

def delete_lead(lead_id, actor=None):
    s = SessionLocal()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if not lead: return False
        s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system", field="delete", old_value=str(lead.stage), new_value="deleted"))
        s.delete(lead)
        s.commit()
        return True
    except Exception as e:
        s.rollback(); raise e
    finally:
        s.close()

@st.cache_data(ttl=20)
def get_leads_df(start=None, end=None):
    s = SessionLocal()
    try:
        leads = s.query(Lead).all()
        data = []
        for L in leads:
            data.append({
                "id":L.id,
                "lead_id":L.lead_id,
                "created_at":L.created_at,
                "source":L.source,
                "stage":L.stage,
                "estimated_value":float(L.estimated_value or 0),
                "ad_cost":float(L.ad_cost or 0),
                "converted":bool(L.converted or False),
                "owner":L.owner.username if L.owner else None,
                "score":L.score,
                "sla_entered_at":L.sla_entered_at,
                "sla_hours":L.sla_hours,
                "notes":L.notes,
                "created_by":L.created_by
            })
        df = pd.DataFrame(data)
        if df.empty: return df

        if start: df = df[df["created_at"] >= datetime.combine(start, datetime.min.time())]
        if end: df = df[df["created_at"] <= datetime.combine(end, datetime.max.time())]
        return df.reset_index(drop=True)
    finally: s.close()

# ---------------- ALERTS (UI-SURFACE ONLY) ----------------

def check_sla_overdue():
    df = get_leads_df()
    overdue = []
    if df.empty: return overdue
    for _,r in df.iterrows():
        d=(r["created_at"]+timedelta(hours=r["sla_hours"]))-datetime.utcnow()
        if d.total_seconds() < 0:
            overdue.append({"lead_id":r["lead_id"],"time_left":"OVERDUE",
                            "value":r["estimated_value"],"stage":r["stage"]})
    return overdue

def show_alert_bell():
    overdue = check_sla_overdue()
    with st.container():
        st.markdown(f"""
        <div style='position:fixed;top:10px;right:20px;font-size:22px;cursor:pointer;'>
             🔔 <span style='background:red;color:white;border-radius:50%;padding:2px 8px;font-size:13px;'>{len(overdue)}</span>
        </div>
        """,unsafe_allow_html=True)
        if overdue:
            for L in overdue[:5]:
                st.markdown(f"""
                <div class='alert-panel'>
                    <span class='close-btn' onclick="this.parentElement.style.display='none'">✖</span>
                    <b>Lead #{L['lead_id']}</b><br>
                    Stage: {L['stage']}<br>
                    Value: <span class='top-value'>${L['value']:,.2f}</span><br>
                    ⏳ <span class='top-time'>{L['time_left']}</span>
                </div>
                """,unsafe_allow_html=True)

# ---------------- ML INTERNAL TRAINING ----------------

def train_model():
    df = pd.read_sql("SELECT * FROM leads", engine)
    if df.empty or len(df["converted"].unique()) == 1:
        return None, "Not enough data for training"
    X = pd.get_dummies(df[["source","stage"]])
    X["ad_cost"]=df["ad_cost"]; X["estimated_value"]=df["estimated_value"]
    y=df["converted"]
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,random_state=42)
    m=RandomForestClassifier(n_estimators=120,random_state=42)
    m.fit(X_train,y_train);p=m.predict(X_test)
    return {"model":m,"columns":X.columns.tolist()}, accuracy_score(y_test,p)

def load_model_bundle():
    if not os.path.exists(MODEL_FILE): return None,None
    try:
        b=joblib.load(MODEL_FILE)
        return b.get("model") or b, b.get("columns","")
    except:
        return None,None

def score_df_with_model(df,model,cols):
    if not model: return df
    X=pd.get_dummies(df[["source","stage"]])
    X["ad_cost"]=df.get("ad_cost",0); X["estimated_value"]=df.get("estimated_value",0)
    for c in cols:
        if c not in X.columns: X[c]=0
    return df.assign(score=model.predict_proba(X[cols])[:,1])

# ---------------- RENDER KPIs IN 2 ROWS ----------------

def show_pipeline_kpis():
    df=df_merged()
    if df.empty: return
    counts=df["stage"].value_counts().to_dict()
    vals=[counts.get(s,0) for s in PIPELINE_STAGES]
    row1=st.columns(4)
    row2=st.columns(4)
    bars=["#06b6d4","#8b5cf6","#3b82f6","#f97316","#dc2626","#22c55e","#facc15","#4f46e5"]
    for col,(stage,c) in zip(row1+row2,zip(PIPELINE_STAGES,vals)):
        bar=random.choice(bars);pct=random.randint(25,90)
        col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>{stage}</div>
                <div class='metric-value' style='color:{bar}'>{c}</div>
                <div class='progress-bar' style='--target:{pct}%;background:{bar};width:{pct}%'></div>
            </div>
        """, unsafe_allow_html=True)

# ---------------- MERGED DF LOADER ----------------

def df_merged(): 
    return get_leads_df(st.session_state.start_date, st.session_state.end_date)

# ---------------- MAIN ROUTER ----------------

nav = st.sidebar.radio("Go to", ["Dashboard","Lead Capture","Analytics","Settings","Export/Import","ML"])

def df_merged(): return get_leads_df(st.session_state.start_date, st.session_state.end_date)

df = df_merged()

if nav == "Dashboard":
    show_alert_bell()
    show_pipeline_kpis()
elif nav == "Lead Capture":
    page_leads()
elif nav == "Analytics":
    page_analytics()
elif nav == "Settings":
    page_settings()
elif nav == "Export/Import":
    page_export_import()
elif nav == "ML":
    page_ml()
