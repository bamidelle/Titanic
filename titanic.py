# titan_merged.py
"""
TITAN — merged full single-file Streamlit app
Features:
- Safe DB migration (adds missing columns/tables)
- Login + roles
- Lead capture/update/delete (SQLite via SQLAlchemy)
- KPI pipeline dashboard (2-row cards)
- Top-5 priority leads (time-left red, value green)
- Alerts bell (UI-only) with close
- Search & quick filters
- Audit trail (LeadHistory)
- CPA & ROI with date filters
- Analytics SLA trend (line chart)
- Import / Export
- Internal ML (train, score persistently) - optional, internal only
"""

import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import io
import base64
import traceback

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Configuration
# -----------------------------
DB_FILE = "titan_merged.db"
MODEL_FILE = "titan_lead_scoring.joblib"
PIPELINE_STAGES = ["New", "Contacted", "Qualified", "Inspection Scheduled", "Inspection Completed", "Estimate Sent", "Won", "Lost"]
SLA_HOURS_DEFAULT = 72  # SLA window (hours) used for "time left"
# Ensure DB path in current working dir (Streamlit Cloud forbids certain file ops)
DB_URL = f"sqlite:///{os.path.abspath(DB_FILE)}"

# SQLAlchemy engine/session
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# -----------------------------
# ORM Models
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")  # Admin, Estimator, Adjuster, Tech, Viewer
    created_at = Column(DateTime, default=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Website")
    source_details = Column(String, nullable=True)
    stage = Column(String, default="New")  # pipeline stage
    estimated_value = Column(Float, default=0.0)
    ad_cost = Column(Float, default=0.0)  # cost to acquire
    converted = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    owner = relationship("User", foreign_keys=[owner_id])
    sla_hours = Column(Integer, default=SLA_HOURS_DEFAULT)
    sla_entered_at = Column(DateTime, nullable=True)
    score = Column(Float, nullable=True)  # ML score (0-1)
    created_by = Column(String, nullable=True)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables if missing
Base.metadata.create_all(bind=engine)

# -----------------------------
# Safe DB migration (add missing columns if DB older)
# -----------------------------
def ensure_schema_migration():
    inspector = inspect(engine)
    with SessionLocal() as s:
        # leads table columns check and add if missing using raw SQL (SQLite ALTER TABLE ADD COLUMN)
        try:
            cols = [c["name"] for c in inspector.get_columns("leads")]
        except Exception:
            cols = []
        # desired columns
        desired = {
            "owner_id": "INTEGER",
            "sla_hours": "INTEGER",
            "sla_entered_at": "DATETIME",
            "score": "FLOAT",
            "created_by": "TEXT",
            "source_details": "TEXT"
        }
        conn = engine.connect()
        for col, typ in desired.items():
            if col not in cols:
                try:
                    conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {typ}")
                except Exception:
                    # harmless if fails (older DB) - ignore
                    pass
        conn.close()
ensure_schema_migration()

# -----------------------------
# Utility helpers
# -----------------------------
def get_session():
    return SessionLocal()

def load_leads_df(start_date=None, end_date=None):
    """
    Return leads as DataFrame filtered by optional start_date/end_date (date objects)
    """
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "lead_id": r.lead_id,
                "created_at": r.created_at,
                "source": r.source,
                "source_details": getattr(r, "source_details", None),
                "stage": r.stage,
                "estimated_value": float(r.estimated_value or 0.0),
                "ad_cost": float(r.ad_cost or 0.0),
                "converted": bool(r.converted),
                "notes": r.notes,
                "owner": r.owner.username if r.owner else None,
                "sla_hours": int(r.sla_hours or SLA_HOURS_DEFAULT),
                "sla_entered_at": r.sla_entered_at,
                "score": float(r.score) if r.score is not None else None,
                "created_by": r.created_by
            })
        df = pd.DataFrame(data)
        if df.empty:
            return df
        # filter by date if provided
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        df = df.reset_index(drop=True)
        return df
    finally:
        s.close()

def upsert_lead(row: dict, actor=None):
    """
    Create or update lead. row must include lead_id.
    """
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == row.get("lead_id")).first()
        if lead is None:
            lead = Lead(
                lead_id=row.get("lead_id"),
                created_at=row.get("created_at", datetime.utcnow()),
                source=row.get("source", "Website"),
                source_details=row.get("source_details"),
                stage=row.get("stage", "New"),
                estimated_value=float(row.get("estimated_value") or 0.0),
                ad_cost=float(row.get("ad_cost") or 0.0),
                converted=bool(row.get("converted") or False),
                notes=row.get("notes"),
                sla_hours=int(row.get("sla_hours") or SLA_HOURS_DEFAULT),
                sla_entered_at=row.get("sla_entered_at"),
                score=row.get("score"),
                created_by=actor or row.get("created_by")
            )
            # assign owner if provided username
            owner = row.get("owner")
            if owner:
                u = s.query(User).filter(User.username == owner).first()
                if u:
                    lead.owner = u
            s.add(lead)
            s.commit()
            s.refresh(lead)
            # history
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system", field="create", old_value=None, new_value=str(row.get("stage"))))
            s.commit()
            return lead.lead_id
        else:
            # update fields and write history for changes
            changed = []
            for key in ["source","source_details","stage","estimated_value","ad_cost","converted","notes","sla_hours","sla_entered_at","score"]:
                if key in row:
                    old = getattr(lead, key)
                    new = row.get(key)
                    # cast numeric
                    if key in ("estimated_value","ad_cost","score"):
                        try:
                            new_val = float(new) if new is not None else None
                        except Exception:
                            new_val = old
                    elif key in ("converted",):
                        new_val = bool(new)
                    elif key in ("sla_hours",):
                        new_val = int(new) if new is not None else old
                    else:
                        new_val = new
                    if new_val is not None and (old != new_val):
                        changed.append((key, old, new_val))
                        setattr(lead, key, new_val)
            # owner update
            if "owner" in row:
                new_owner = row.get("owner")
                old_owner = lead.owner.username if lead.owner else None
                if new_owner != old_owner:
                    u = s.query(User).filter(User.username == new_owner).first() if new_owner else None
                    lead.owner = u
                    changed.append(("owner", old_owner, new_owner))
            # created_by unchanged
            s.add(lead)
            # write history entries
            for (field, oldv, newv) in changed:
                s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system", field=field, old_value=str(oldv), new_value=str(newv)))
            s.commit()
            return lead.lead_id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def delete_lead(lead_id: str, actor=None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if not lead:
            return False
        s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor or "system", field="delete", old_value=str(lead.stage), new_value="deleted"))
        s.delete(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# -----------------------------
# ML helpers (internal only)
# -----------------------------
def train_model():
    """
    Train RandomForest on leads in DB to predict converted.
    Save model + feature columns to file.
    Returns (accuracy float) or (None, message)
    """
    df = load_leads_df()
    if df.empty:
        return None, "No data to train on"
    # need at least two classes
    if df["converted"].nunique() < 2:
        return None, "Need at least two target classes to train"
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    # one-hot source and stage
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    y = df2["converted"].astype(int)
    # align
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # persist model and columns
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    return acc, "trained"

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            obj = joblib.load(MODEL_FILE)
            return obj.get("model"), obj.get("columns")
        except Exception:
            return None, None
    return None, None

def score_df_with_model(df, model, cols):
    if model is None or df.empty:
        df["score"] = np.nan
        return df
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    # ensure columns
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols].fillna(0)
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        probs = model.predict(X)
    df["score"] = probs
    return df

# -----------------------------
# Priority & SLA helpers
# -----------------------------
def compute_time_left_hours(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or SLA_HOURS_DEFAULT))
        remain = deadline - datetime.utcnow()
        return max(remain.total_seconds()/3600.0, 0.0), (remain.total_seconds() <= 0)
    except Exception:
        return float("inf"), False

def compute_priority_score(row, weights=None):
    # row: dict or pd.Series with estimated_value, score, sla_hours, sla_entered_at, contacted flag optional
    if weights is None:
        weights = {"score_w": 0.6, "value_w": 0.3, "sla_w": 0.1, "value_baseline": 5000.0}
    try:
        score = float(row.get("score") or 0.0)
        val = float(row.get("estimated_value") or 0.0)
        vnorm = min(1.0, val / max(1.0, weights["value_baseline"]))
    except Exception:
        score, vnorm = 0.0, 0.0
    # SLA urgency: inverse of time left
    try:
        tleft_h, overdue = compute_time_left_hours(row.get("sla_entered_at") or row.get("created_at"), row.get("sla_hours"))
        sla_score = max(0.0, (72.0 - min(tleft_h,72.0)) / 72.0)
    except Exception:
        sla_score = 0.0
    total = score * weights["score_w"] + vnorm * weights["value_w"] + sla_score * weights["sla_w"]
    return max(0.0, min(1.0, total))

# -----------------------------
# UI helpers: CSS, download
# -----------------------------
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
body, .stApp { font-family: 'Comfortaa', sans-serif; background: #ffffff; color: #0b1220; }
.header { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.kpi-card { background:#000; border-radius:12px; padding:14px; color:white; min-width:200px; margin:6px; }
.kpi-title { color:white; font-weight:700; font-size:13px; margin-bottom:6px; }
.kpi-value { font-weight:900; font-size:24px; margin-bottom:8px; }
.kpi-note { color:rgba(255,255,255,0.9); font-size:12px }
.progress-bar { height:8px; border-radius:8px; margin-top:8px; transition:width .4s ease; }
.lead-card { border-radius:10px; padding:12px; border:1px solid #eee; margin-bottom:8px; background:#fff; }
.priority-time { color:#dc2626; font-weight:700; }
.priority-money { color:#22c55e; font-weight:800; }
.alert-bell { position: fixed; top: 18px; right: 18px; z-index: 9999; }
.alert-box { background:#111; color:white; padding:12px; border-radius:8px; width:320px; box-shadow: 0 10px 30px rgba(0,0,0,0.12); }
.small-muted { color:#6b7280; font-size:12px; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

def df_to_excel_download(df: pd.DataFrame, fname="leads.xlsx"):
    towrite = io.BytesIO()
    try:
        df.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        return f'<a href="{href}" download="{fname}">Download {fname}</a>'
    except Exception:
        csv = df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        return f'<a href="data:text/csv;base64,{b64}" download="{fname.replace(".xlsx",".csv")}">Download CSV</a>'

# -----------------------------
# App state defaults
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "alerts_open" not in st.session_state:
    st.session_state.alerts_open = False
if "weights" not in st.session_state:
    st.session_state.weights = {"score_w": 0.6, "value_w": 0.3, "sla_w": 0.1, "value_baseline": 5000.0}

# -----------------------------
# Login / Sidebar
# -----------------------------
st.set_page_config(page_title="TITAN — Lead Pipeline", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.markdown("## TITAN — Control")
    if st.session_state.user is None:
        st.markdown("### Login")
        username = st.text_input("Name (username)", value="")
        role_choice = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=4)
        if st.button("Login"):
            if not username.strip():
                st.warning("Enter a username")
            else:
                s = get_session()
                try:
                    u = s.query(User).filter(User.username == username.strip()).first()
                    if u is None:
                        u = User(username=username.strip(), full_name=username.strip(), role=role_choice)
                        s.add(u); s.commit()
                    st.session_state.user = u.username
                    st.session_state.role = u.role
                    st.success(f"Signed in as {st.session_state.user} ({st.session_state.role})")
                finally:
                    s.close()
    else:
        st.markdown(f"### Signed in as **{st.session_state.user}**")
        st.markdown(f"Role: **{st.session_state.role}**")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.role = None
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Dashboard","Leads / Capture","Analytics & SLA","CPA & ROI","ML (internal)","Settings","Exports"], index=0)
    st.markdown("---")
    st.markdown("### Date Range (for reports)")
    range_choice = st.selectbox("Quick ranges", ["Today","Last 7 days","Last 30 days","90 days","All","Custom"], index=5)
    if range_choice == "Today":
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
    elif range_choice == "Last 7 days":
        st.session_state.start_date = date.today() - timedelta(days=6)
        st.session_state.end_date = date.today()
    elif range_choice == "Last 30 days":
        st.session_state.start_date = date.today() - timedelta(days=29)
        st.session_state.end_date = date.today()
    elif range_choice == "90 days":
        st.session_state.start_date = date.today() - timedelta(days=89)
        st.session_state.end_date = date.today()
    elif range_choice == "All":
        st.session_state.start_date = None
        st.session_state.end_date = None
    else:
        sd, ed = st.date_input("Start — End", [date.today() - timedelta(days=29), date.today()])
        st.session_state.start_date = sd
        st.session_state.end_date = ed

# if not logged in, stop and ask to login
if st.session_state.user is None:
    st.info("Please login from the left panel to continue.")
    st.stop()

# -----------------------------
# Alerts bell UI
# -----------------------------
def compute_overdue_list(df):
    overdue = []
    for _, r in df.iterrows():
        tleft, is_over = compute_time_left_hours(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if is_over and r.get("stage") not in ("Won","Lost"):
            overdue.append(r)
    return overdue

# compute main filtered df for the date range
start_d = st.session_state.get("start_date", None)
end_d = st.session_state.get("end_date", None)
leads_df = load_leads_df(start_d, end_d)

# attempt to load model for scoring
model, model_cols = load_model()
if model is not None:
    leads_df = score_df_with_model(leads_df.copy(), model, model_cols)

overdue_list = compute_overdue_list(leads_df)

# bell UI (top right)
if len(overdue_list) > 0:
    # show small button to toggle alerts
    cols = st.columns([1, 10])
    with cols[0]:
        if st.button(f"🔔 {len(overdue_list)}"):
            st.session_state.alerts_open = not st.session_state.alerts_open
    with cols[1]:
        st.markdown("")
else:
    cols = st.columns([1, 10])
    with cols[0]:
        st.markdown("")

if st.session_state.alerts_open:
    # show alerts box top-right visually
    st.markdown("<div style='position:fixed; top:68px; right:18px; z-index:9999;'>", unsafe_allow_html=True)
    st.markdown("<div class='alert-box'><div style='display:flex; justify-content:space-between; align-items:center;'><div style='font-weight:800'>SLA Alerts</div><div><button id='close_alerts_btn'>✖</button></div></div><hr>", unsafe_allow_html=True)
    for r in overdue_list[:20]:
        tleft_h, _ = compute_time_left_hours(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        st.markdown(f"<div style='padding:6px 0;'><b>{r['lead_id']}</b> — <span style='color:#22c55e;'>${r['estimated_value']:,.0f}</span> — <span style='color:#dc2626;font-weight:700;'>{int(tleft_h)}h left / OVERDUE</span></div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# -----------------------------
# Page: Dashboard
# -----------------------------
def page_dashboard():
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("*High-level pipeline performance at a glance. Use filters + date range to drill into details.*")
    df = leads_df.copy()
    total_leads = len(df)
    qualified_leads = int(df[df["stage"] == "Qualified"].shape[0]) if not df.empty else 0
    sla_contacts = int(df[df["stage"] == "Contacted"].shape[0]) if not df.empty else 0
    awarded_count = int(df[df["stage"] == "Won"].shape[0]) if not df.empty else 0
    lost_count = int(df[df["stage"] == "Lost"].shape[0]) if not df.empty else 0
    inspection_scheduled_count = int(df[df["stage"] == "Inspection Scheduled"].shape[0]) if not df.empty else 0
    estimate_sent_count = int(df[df["stage"] == "Estimate Sent"].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)

    sla_success_pct = (sla_contacts / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_pct = (inspection_scheduled_count / qualified_leads * 100) if qualified_leads else 0.0

    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "#2563eb"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444"),
        ("Estimates Sent", f"{estimate_sent_count}", "#6d28d9"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "#22c55e")
    ]

    # render 2 rows (4 + 3)
    cols_row1 = st.columns(4)
    cols_row2 = st.columns(3)
    cols = cols_row1 + cols_row2
    for c, (title, value, color) in zip(cols, KPI_ITEMS):
        pct = min(100, max(10, int((hash(title) % 80) + 20)))  # deterministic-ish percent for progress bar
        c.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-title'>{title}</div>
              <div class='kpi-value' style='color:{color};'>{value}</div>
              <div class='progress-bar' style='background:{color}; width:{pct}%;'></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline stages (line chart removed from dashboard; show counts table)
    st.subheader("Lead Pipeline Stages")
    st.markdown("*Distribution across stages.*")
    if df.empty:
        st.info("No leads to show.")
    else:
        stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0).reset_index()
        stage_counts.columns = ["stage", "count"]
        st.table(stage_counts)

    st.markdown("---")

    # Top 5 priority leads
    st.subheader("TOP 5 PRIORITY LEADS")
    st.markdown("*Highest urgency leads by priority score (0–1). Address these first.*")
    if df.empty:
        st.info("No leads yet.")
    else:
        # compute priority score per lead
        df["priority_score"] = df.apply(lambda r: compute_priority_score(r), axis=1)
        pr_df = df.sort_values("priority_score", ascending=False).head(5).copy()
        for _, r in pr_df.iterrows():
            sla_left_h, overdue = compute_time_left_hours(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            time_html = f"<span class='priority-time'>{'❗ OVERDUE' if overdue else f'{int(sla_left_h)}h left'}</span>"
            money_html = f"<span class='priority-money'>${r.get('estimated_value',0):,.0f}</span>"
            st.markdown(f"""
                <div class='lead-card'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div><b>#{r['lead_id']} — {r['source']}</b><div class='small-muted'>{r.get('notes') or ''}</div></div>
                    <div style='text-align:right'>{money_html}<br>{time_html}</div>
                  </div>
                  <div style='margin-top:8px;'><small class='small-muted'>Priority score: {r['priority_score']:.2f} — Stage: {r['stage']}</small></div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("All Leads (expand a card to edit / change status)")
    st.markdown("*Expand a lead to edit details, change stage, assign owner, upload invoice, and add notes.*")

    # Search & quick filters
    qcol1, qcol2, qcol3 = st.columns([3,2,3])
    with qcol1:
        search_q = st.text_input("Search (lead_id, source, notes)")
    with qcol2:
        filter_source = st.selectbox("Filter source", options=["All"] + sorted(df["source"].dropna().unique().tolist()) if not df.empty else ["All"])
    with qcol3:
        filter_stage = st.selectbox("Filter stage", options=["All"] + PIPELINE_STAGES)

    df_view = df.copy()
    if search_q:
        sq = search_q.lower()
        df_view = df_view[df_view.apply(lambda r: sq in str(r.get("lead_id","")).lower() or sq in str(r.get("source","")).lower() or sq in str(r.get("notes","")).lower(), axis=1)]
    if filter_source and filter_source != "All":
        df_view = df_view[df_view["source"] == filter_source]
    if filter_stage and filter_stage != "All":
        df_view = df_view[df_view["stage"] == filter_stage]

    # show table and expand editor for each row
    if df_view.empty:
        st.info("No leads matching filters.")
    else:
        for _, row in df_view.sort_values("created_at", ascending=False).head(200).iterrows():
            with st.expander(f"#{row['lead_id']} — {row.get('source','—')} — {row.get('stage')}"):
                colL, colR = st.columns([3,1])
                with colL:
                    st.write(f"**Notes:** {row.get('notes') or '—'}")
                    st.write(f"**Created:** {row.get('created_at')}")
                    st.write(f"**Created by:** {row.get('created_by') or '—'}")
                with colR:
                    sla_left_h, overdue = compute_time_left_hours(row.get("sla_entered_at") or row.get("created_at"), row.get("sla_hours"))
                    if overdue:
                        st.markdown("<div class='priority-time'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='small-muted'>⏳ {int(sla_left_h)}h left</div>", unsafe_allow_html=True)

                # editor form
                new_stage = st.selectbox("Stage", PIPELINE_STAGES, index=PIPELINE_STAGES.index(row.get("stage")) if row.get("stage") in PIPELINE_STAGES else 0, key=f"stage_{row['lead_id']}")
                owner_list = [u.username for u in get_session().query(User).all()]
                new_owner = st.selectbox("Assign to (username)", options=[""] + owner_list, index=0 if not row.get("owner") else (owner_list.index(row["owner"])+1 if row["owner"] in owner_list else 0), key=f"owner_{row['lead_id']}")
                new_est = st.number_input("Estimate value (USD)", value=float(row.get("estimated_value") or 0.0), min_value=0.0, step=100.0, key=f"est_{row['lead_id']}")
                new_cost = st.number_input("Cost to acquire lead (USD)", value=float(row.get("ad_cost") or 0.0), min_value=0.0, step=1.0, key=f"cost_{row['lead_id']}")
                new_notes = st.text_area("Notes", value=row.get("notes") or "", key=f"notes_{row['lead_id']}")
                if st.button("Save changes", key=f"save_{row['lead_id']}"):
                    try:
                        upsert_lead({
                            "lead_id": row["lead_id"],
                            "stage": new_stage,
                            "owner": new_owner if new_owner else None,
                            "estimated_value": float(new_est or 0.0),
                            "ad_cost": float(new_cost or 0.0),
                            "notes": new_notes,
                            "sla_entered_at": row.get("sla_entered_at") or row.get("created_at"),
                            "created_by": row.get("created_by")
                        }, actor=st.session_state.user)
                        st.success("Lead updated")
                        # refresh dataframe
                        st.experimental_rerun()
                    except Exception as e:
                        st.error("Failed to save: " + str(e))
                        st.write(traceback.format_exc())

# -----------------------------
# Page: Leads / Capture
# -----------------------------
def page_leads_capture():
    st.header("📇 Lead Capture")
    with st.form("lead_form", clear_on_submit=True):
        lead_id = st.text_input("Lead ID (unique)", value=f"L{int(datetime.utcnow().timestamp())}")
        source = st.selectbox("Source", ["Google Ads","Organic","Referral","Facebook Ads","Direct","Partner","Other"])
        source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google...")
        stage = st.selectbox("Stage", PIPELINE_STAGES, index=0)
        est_val = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        ad_cost = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        notes = st.text_area("Notes")
        sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=SLA_HOURS_DEFAULT, step=1)
        submitted = st.form_submit_button("Create / Upsert Lead")
        if submitted:
            try:
                upsert_lead({
                    "lead_id": lead_id.strip(),
                    "created_at": datetime.utcnow(),
                    "source": source,
                    "source_details": source_details,
                    "stage": stage,
                    "estimated_value": float(est_val),
                    "ad_cost": float(ad_cost),
                    "notes": notes,
                    "sla_hours": int(sla_hours),
                    "sla_entered_at": datetime.utcnow(),
                    "created_by": st.session_state.user
                }, actor=st.session_state.user)
                st.success(f"Lead {lead_id} saved")
                st.experimental_rerun()
            except Exception as e:
                st.error("Failed to save lead: " + str(e))
                st.write(traceback.format_exc())

    st.markdown("---")
    st.subheader("Recent leads (most recent first)")
    df = load_leads_df(None, None)
    if df.empty:
        st.info("No leads yet.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(200))

# -----------------------------
# Page: Analytics & SLA
# -----------------------------
def page_analytics_sla():
    st.header("📈 Analytics — SLA Overdue Trend & Pipeline Stages")
    df = load_leads_df(start_d, end_d)
    if df.empty:
        st.info("No leads to analyze.")
        return

    # SLA Overdue timeseries: last 30 days
    st.markdown("### SLA / Overdue Leads (last 30 days)")
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(29, -1, -1)]
    counts = []
    for d in days:
        start_dt = datetime.combine(d, datetime.min.time())
        end_dt = datetime.combine(d, datetime.max.time())
        sub = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
        overdue_count = 0
        for _, r in sub.iterrows():
            tleft, overdue_flag = compute_time_left_hours(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            if overdue_flag and r.get("stage") not in ("Won","Lost"):
                overdue_count += 1
        counts.append(overdue_count)
    ts_df = pd.DataFrame({"date": days, "overdue": counts})
    fig = px.line(ts_df, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Pipeline Stage Counts (by selected date range)")
    stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0).reset_index()
    stage_counts.columns = ["stage","count"]
    fig2 = px.bar(stage_counts, x="stage", y="count", color="stage", title="Pipeline Stages")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("SLA Overdue Leads (current)")
    overdue_df = df[df.apply(lambda r: compute_time_left_hours(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))[1] and r.get("stage") not in ("Won","Lost"), axis=1)]
    if overdue_df.empty:
        st.info("No SLA overdue leads.")
    else:
        st.dataframe(overdue_df.sort_values("created_at"))

# -----------------------------
# Page: CPA & ROI
# -----------------------------
def page_cpa_roi():
    st.header("💰 CPA & ROI")
    df = load_leads_df(start_d, end_d)
    if df.empty:
        st.info("No leads available")
        return
    # mapping: user-provided spend per source not implemented; compute total ad_cost
    total_spend = float(df["ad_cost"].sum())
    awarded_df = df[df["stage"] == "Won"]
    conversions = int(awarded_df.shape[0])
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(awarded_df["estimated_value"].sum())
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Marketing Spend</div><div class='kpi-value' style='color:#2563eb'>${total_spend:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversions (Won)</div><div class='kpi-value' style='color:#06b6d4'>{conversions}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-title'>CPA</div><div class='kpi-value' style='color:#f97316'>${cpa:,.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-title'>ROI</div><div class='kpi-value' style='color:#22c55e'>${roi:,.2f} ({roi_pct:.1f}%)</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # chart: total spend vs conversions by source
    agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("stage", lambda s: (s=="Won").sum())).reset_index()
    if not agg.empty:
        fig = px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group", title="Total Spend vs Conversions by Source")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spend / conversion data to plot.")

# -----------------------------
# Page: ML internal (train + score)
# -----------------------------
def page_ml_internal():
    st.header("🧠 Internal ML (lead scoring — internal only)")
    st.markdown("Train a RandomForest on historic leads to predict conversion probability (saved locally).")
    if st.button("Train model now (internal)"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_model()
                if acc is None:
                    st.error(f"Training aborted: {msg}")
                else:
                    st.success(f"Model trained — approx accuracy: {acc:.3f}")
                    # reload model into memory
                    global model, model_cols
                    model, model_cols = load_model()
            except Exception as e:
                st.error("Training failed: " + str(e))
                st.write(traceback.format_exc())

    model, cols = load_model()
    if model is None:
        st.info("No trained model available. Train using the button above.")
    else:
        st.success("Model available (internal)")
        if st.button("Score leads now and persist scores to DB"):
            df = load_leads_df()
            scored = score_df_with_model(df.copy(), model, cols)
            # persist scores
            s = get_session()
            try:
                for _, r in scored.iterrows():
                    lead = s.query(Lead).filter(Lead.lead_id == r["lead_id"]).first()
                    if lead:
                        lead.score = float(r["score"])
                        s.add(lead)
                s.commit()
                st.success("Scores written to DB (score column).")
            except Exception as e:
                s.rollback()
                st.error("Failed to persist scores: " + str(e))
            finally:
                s.close()
        # preview top predicted leads
        df = load_leads_df()
        scored_preview = score_df_with_model(df.copy(), model, cols).sort_values("score", ascending=False).head(20)
        st.dataframe(scored_preview[["lead_id","source","stage","estimated_value","ad_cost","score"]])

# -----------------------------
# Page: Settings
# -----------------------------
def page_settings():
    st.header("⚙️ Settings")
    st.markdown("Application and user settings.")
    # show current user record
    s = get_session()
    try:
        u = s.query(User).filter(User.username == st.session_state.user).first()
        if u:
            st.markdown(f"**Username:** {u.username}")
            new_full = st.text_input("Full name", value=u.full_name or "")
            new_role = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=["Admin","Estimator","Adjuster","Tech","Viewer"].index(u.role))
            if st.button("Save profile"):
                u.full_name = new_full
                u.role = new_role
                s.add(u); s.commit()
                st.success("Profile saved")
    finally:
        s.close()
    st.markdown("---")
    st.subheader("Priority weight tuning (internal)")
    w_score = st.slider("Model score weight", 0.0, 1.0, st.session_state.weights["score_w"], 0.05)
    w_value = st.slider("Estimate value weight", 0.0, 1.0, st.session_state.weights["value_w"], 0.05)
    w_sla = st.slider("SLA urgency weight", 0.0, 1.0, st.session_state.weights["sla_w"], 0.05)
    baseline = st.number_input("Value baseline (for normalization)", value=st.session_state.weights["value_baseline"], min_value=100.0)
    if st.button("Save weights"):
        st.session_state.weights["score_w"] = w_score
        st.session_state.weights["value_w"] = w_value
        st.session_state.weights["sla_w"] = w_sla
        st.session_state.weights["value_baseline"] = baseline
        st.success("Weights saved")

# -----------------------------
# Page: Exports / Imports
# -----------------------------
def page_exports():
    st.header("📤 Exports & Imports")
    df = load_leads_df()
    if df.empty:
        st.info("No leads to export")
    else:
        if st.button("Download leads (Excel)"):
            html = df_to_excel_download(df, "leads_export.xlsx")
            st.markdown(html, unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Import leads (CSV or Excel). Required cols: lead_id, created_at, source, stage", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
            # minimal validation
            required = {"lead_id","created_at","source","stage"}
            if not required.issubset(set(df_in.columns)):
                st.error(f"Missing columns: {required - set(df_in.columns)}")
            else:
                # cast
                df_in["created_at"] = pd.to_datetime(df_in["created_at"])
                # upsert rows
                count = 0
                for _, r in df_in.iterrows():
                    try:
                        upsert_lead({
                            "lead_id": str(r["lead_id"]),
                            "created_at": r["created_at"],
                            "source": r.get("source"),
                            "stage": r.get("stage"),
                            "estimated_value": float(r.get("estimated_value") or 0.0),
                            "ad_cost": float(r.get("ad_cost") or 0.0),
                            "converted": int(r.get("converted") or 0),
                            "notes": r.get("notes"),
                            "score": r.get("score")
                        }, actor=st.session_state.user)
                        count += 1
                    except Exception:
                        continue
                st.success(f"Imported {count} rows (existing lead_id updated/ignored).")
        except Exception as e:
            st.error("Import failed: " + str(e))

# -----------------------------
# Router
# -----------------------------
start_d = st.session_state.get("start_date", None)
end_d = st.session_state.get("end_date", None)

if page == "Dashboard":
    page_dashboard()
elif page == "Leads / Capture":
    page_leads_capture()
elif page == "Analytics & SLA":
    page_analytics_sla()
elif page == "CPA & ROI":
    page_cpa_roi()
elif page == "ML (internal)":
    page_ml_internal()
elif page == "Settings":
    page_settings()
elif page == "Exports":
    page_exports()
else:
    st.info("Page not implemented")

# Footer & small helpful note
st.markdown("---")
st.markdown("<div class='small-muted'>TITAN — Single-file Streamlit · SQLite persistence · Internal ML (optional)</div>", unsafe_allow_html=True)
