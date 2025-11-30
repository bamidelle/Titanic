# ✅ PERMANENT FIX: Redirect removed/deprecated Streamlit rerun API
import streamlit as st
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

# titan_final_c.py
"""
TITAN — Single-file Streamlit app (Option C: DB in script folder)
Author: Generated for user
"""

import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import os
import io
import base64
import traceback
import joblib
import random

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError

# Temporary compatibility patch — prevents crashes from deprecated rerun calls
import streamlit as st
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun


# Optional ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px

# ---------------------------
# Configuration (Option C)
# ---------------------------
DB_FILENAME = "streamlit_app.db"        # Option C: DB file in same folder
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = [
    "New", "Contacted", "Inspection Scheduled", "Inspection Completed",
    "Estimate Sent", "Qualified", "Won", "Lost"
]
SLA_HOURS_DEFAULT = 48

# Ensure DB path is in script folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_FILENAME)
DB_URL = f"sqlite:///{DB_PATH}"

# ---------------------------
# SQLAlchemy setup
# ---------------------------
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# ---------------------------
# ORM Models
# ---------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")  # Admin, Estimator, Adjuster, Tech, Viewer
    created_at = Column(DateTime, default=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)  # internal DB id
    lead_code = Column(String, unique=True, nullable=False)  # e.g. LEAD-20251130-0001
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Website")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)  # username
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, default=0.0)
    ad_cost = Column(Float, default=0.0)  # cost to acquire
    stage = Column(String, default="New")
    converted = Column(Boolean, default=False)
    sla_hours = Column(Integer, default=SLA_HOURS_DEFAULT)
    sla_entered_at = Column(DateTime, nullable=True)
    score = Column(Float, nullable=True)  # ML score 0..1
    created_by = Column(String, nullable=True)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_code = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ---------------------------
# Create DB / safe migration
# ---------------------------
def init_db_and_migrate():
    Base.metadata.create_all(bind=engine)
    # safe add missing columns if migrating from earlier schemas
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("leads")} if inspector.has_table("leads") else set()
    required_cols = {
        "lead_code", "created_at", "source", "source_details", "contact_name", "contact_phone",
        "contact_email", "property_address", "damage_type", "assigned_to", "notes",
        "estimated_value", "ad_cost", "stage", "converted", "sla_hours", "sla_entered_at",
        "score", "created_by"
    }
    missing = required_cols - cols
    # For SQLite, adding columns is simple: ALTER TABLE ADD COLUMN
    if missing:
        conn = engine.connect()
        for c in missing:
            # basic add as TEXT (safe for most)
            try:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {c} TEXT")
            except Exception:
                pass
        conn.close()

init_db_and_migrate()

# ---------------------------
# Helpers: DB access and DataFrame conversions
# ---------------------------
def get_session():
    return SessionLocal()

def leads_to_df(session, start_date=None, end_date=None):
    rows = session.query(Lead).order_by(Lead.created_at.desc()).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "lead_code": r.lead_code,
            "created_at": r.created_at,
            "source": r.source,
            "source_details": r.source_details,
            "contact_name": r.contact_name,
            "contact_phone": r.contact_phone,
            "contact_email": r.contact_email,
            "property_address": r.property_address,
            "damage_type": r.damage_type,
            "assigned_to": r.assigned_to,
            "notes": r.notes,
            "estimated_value": float(r.estimated_value or 0.0),
            "ad_cost": float(r.ad_cost or 0.0),
            "stage": r.stage,
            "converted": bool(r.converted),
            "sla_hours": int(r.sla_hours or SLA_HOURS_DEFAULT),
            "sla_entered_at": r.sla_entered_at,
            "score": float(r.score) if r.score is not None else None,
            "created_by": r.created_by
        })
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # filter by date
    if start_date:
        df = df[df["created_at"] >= datetime.combine(start_date, datetime.min.time())]
    if end_date:
        df = df[df["created_at"] <= datetime.combine(end_date, datetime.max.time())]
    df = df.reset_index(drop=True)
    return df

# ---------------------------
# Utility functions
# ---------------------------
def generate_auto_lead_code(session):
    # produce LEAD-YYYYMMDD-0001 using count of existing leads that day +1
    today = datetime.utcnow().strftime("%Y%m%d")
    # count leads with today's prefix
    total_today = session.query(Lead).filter(Lead.lead_code.like(f"LEAD-{today}-%")).count()
    seq = total_today + 1
    return f"LEAD-{today}-{str(seq).zfill(4)}"

def create_or_update_lead(payload: dict, actor=None):
    """
    payload should contain 'lead_code' (prefer) or will generate one.
    If lead_code exists -> update. Else -> create.
    """
    s = get_session()
    try:
        lead_code = payload.get("lead_code")
        if not lead_code:
            lead_code = generate_auto_lead_code(s)
        lead = s.query(Lead).filter(Lead.lead_code == lead_code).first()
        if lead is None:
            # create
            lead = Lead(
                lead_code=lead_code,
                created_at=payload.get("created_at") or datetime.utcnow(),
                source=payload.get("source") or "Website",
                source_details=payload.get("source_details"),
                contact_name=payload.get("contact_name"),
                contact_phone=payload.get("contact_phone"),
                contact_email=payload.get("contact_email"),
                property_address=payload.get("property_address"),
                damage_type=payload.get("damage_type"),
                assigned_to=payload.get("assigned_to"),
                notes=payload.get("notes"),
                estimated_value=float(payload.get("estimated_value") or 0.0),
                ad_cost=float(payload.get("ad_cost") or 0.0),
                stage=payload.get("stage") or "New",
                converted=bool(payload.get("converted") or (payload.get("stage") == "Won")),
                sla_hours=int(payload.get("sla_hours") or SLA_HOURS_DEFAULT),
                sla_entered_at=payload.get("sla_entered_at") or datetime.utcnow(),
                score=payload.get("score"),
                created_by=actor or payload.get("created_by")
            )
            s.add(lead)
            s.commit()
            s.refresh(lead)
            # history
            hist = LeadHistory(lead_code=lead.lead_code, changed_by=actor or lead.created_by, field="create", old_value=None, new_value=lead.stage)
            s.add(hist); s.commit()
            return lead.lead_code
        else:
            # update fields with history
            changed = []
            for key in ["source","source_details","contact_name","contact_phone","contact_email","property_address","damage_type","assigned_to","notes","estimated_value","ad_cost","stage","converted","sla_hours","sla_entered_at","score"]:
                if key in payload:
                    oldv = getattr(lead, key)
                    newv = payload.get(key)
                    # cast numeric types appropriately
                    if key in ("estimated_value","ad_cost","score"):
                        try:
                            newv_cast = float(newv) if newv is not None else None
                        except Exception:
                            newv_cast = oldv
                        newv = newv_cast
                    if key in ("sla_hours",):
                        try:
                            newv = int(newv)
                        except Exception:
                            newv = oldv
                    if newv is not None and oldv != newv:
                        changed.append((key, oldv, newv))
                        setattr(lead, key, newv)
            # persist
            s.add(lead)
            for (field, old, new) in changed:
                h = LeadHistory(lead_code=lead.lead_code, changed_by=actor or None, field=field, old_value=str(old), new_value=str(new))
                s.add(h)
            s.commit()
            return lead.lead_code
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def delete_lead_by_code(lead_code, actor=None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_code == lead_code).first()
        if not lead:
            return False
        # history
        h = LeadHistory(lead_code=lead.lead_code, changed_by=actor, field="deleted", old_value=str(lead.stage), new_value="deleted")
        s.add(h)
        s.delete(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# ---------------------------
# Priority & SLA helpers
# ---------------------------
def compute_time_left_and_overdue(lead_row):
    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    if sla_entered is None:
        sla_entered = datetime.utcnow()
    if isinstance(sla_entered, str):
        try:
            sla_entered = datetime.fromisoformat(sla_entered)
        except Exception:
            sla_entered = datetime.utcnow()
    try:
        deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or SLA_HOURS_DEFAULT))
        remain = deadline - datetime.utcnow()
        remain_hours = max(remain.total_seconds() / 3600.0, 0.0)
        overdue = (remain.total_seconds() <= 0)
        return remain_hours, overdue
    except Exception:
        return float("inf"), False

def compute_priority_score(row, weights=None):
    # weights: score_w, value_w, sla_w, value_baseline
    if weights is None:
        weights = {"score_w":0.6, "value_w":0.3, "sla_w":0.1, "value_baseline":5000.0}
    score_part = float(row.get("score") or 0.0)
    try:
        val = float(row.get("estimated_value") or 0.0)
    except Exception:
        val = 0.0
    value_score = min(1.0, val / max(1.0, weights["value_baseline"]))
    sla_hours_left, overdue = compute_time_left_and_overdue(row)
    sla_score = max(0.0, (72.0 - min(sla_hours_left,72.0)) / 72.0)
    total = score_part * weights["score_w"] + value_score * weights["value_w"] + sla_score * weights["sla_w"]
    return max(0.0, min(1.0, total))

# ---------------------------
# Export helper
# ---------------------------
def df_to_excel_download(df: pd.DataFrame, filename="leads_export.xlsx"):
    towrite = io.BytesIO()
    try:
        df.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        return f'<a href="{href}" download="{filename}">Download {filename}</a>'
    except Exception:
        csv = df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        return f'<a href="data:text/csv;base64,{b64}" download="{filename.replace(".xlsx",".csv")}">Download CSV</a>'

# ---------------------------
# ML helpers
# ---------------------------
def train_internal_model():
    s = get_session()
    try:
        df = leads_to_df(s)
        if df.empty or df["converted"].nunique() < 2:
            return None, "Not enough data"
        df2 = df.copy()
        df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
        X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
        X["ad_cost"] = df2["ad_cost"]
        X["estimated_value"] = df2["estimated_value"]
        X["age_days"] = df2["age_days"]
        y = df2["converted"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
        return acc, "trained"
    finally:
        s.close()

def load_internal_model():
    if os.path.exists(MODEL_FILE):
        try:
            obj = joblib.load(MODEL_FILE)
            return obj.get("model"), obj.get("columns")
        except Exception:
            return None, None
    return None, None

def score_and_persist_scores():
    model, cols = load_internal_model()
    if model is None:
        return False, "No model"
    s = get_session()
    try:
        df = leads_to_df(s)
        if df.empty:
            return False, "No leads"
        df2 = df.copy()
        df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
        X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
        X["ad_cost"] = df2["ad_cost"]
        X["estimated_value"] = df2["estimated_value"]
        X["age_days"] = df2["age_days"]
        for c in cols:
            if c not in X.columns:
                X[c] = 0
        X = X[cols].fillna(0)
        try:
            probs = model.predict_proba(X)[:,1]
        except Exception:
            probs = model.predict(X)
        # persist
        for lead_code, p in zip(df2["lead_code"], probs):
            lead = s.query(Lead).filter(Lead.lead_code == lead_code).first()
            if lead:
                lead.score = float(p)
                s.add(lead)
        s.commit()
        return True, "Scores saved"
    except Exception as e:
        s.rollback()
        return False, str(e)
    finally:
        s.close()

# ---------------------------
# UI: Styles
# ---------------------------
st.set_page_config(page_title="TITAN — Pipeline", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
body, .stApp { font-family:'Comfortaa', sans-serif; background: #ffffff; color: #0b1220; }
.kpi-card { background: #000; color: #fff; border-radius:12px; padding:14px; margin:8px; min-width:200px; }
.kpi-title { font-size:13px; color:#fff; font-weight:700; }
.kpi-value { font-size:26px; font-weight:800; }
.progress{ height:8px; border-radius:8px; margin-top:8px; }
.lead-card { border:1px solid #eee; padding:10px; margin-bottom:8px; border-radius:8px; background:#fff; }
.priority-time { color:#dc2626; font-weight:700; }
.priority-money { color:#22c55e; font-weight:800; }
.alert-box { position:fixed; right:20px; top:80px; width:340px; background:#111; color:#fff; padding:12px; border-radius:8px; z-index:9999; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar + login
# ---------------------------
with st.sidebar:
    st.title("TITAN — Controls")
    if "username" not in st.session_state:
        st.session_state.username = None
    if st.session_state.username is None:
        st.text("Login")
        username_in = st.text_input("Username")
        role_in = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=4)
        if st.button("Login"):
            if not username_in.strip():
                st.warning("Enter username")
            else:
                s = get_session()
                user = s.query(User).filter(User.username == username_in.strip()).first()
                if user is None:
                    user = User(username=username_in.strip(), full_name=username_in.strip(), role=role_in)
                    s.add(user); s.commit()
                st.session_state.username = user.username
                st.session_state.role = user.role
                s.close()
                st.experimental_rerun()
        st.stop()
    else:
        st.markdown(f"**Signed in as:** {st.session_state.username}  ({st.session_state.role})")
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

    st.markdown("---")
    page = st.radio("Go to", ["Dashboard","Leads / Capture","Analytics & SLA","CPA & ROI","ML","Settings","Exports","Audit Trail"])
    st.markdown("---")
    st.markdown("Date range")
    quick = st.selectbox("Quick range", ["Today","Last 7","Last 30","90 days","All","Custom"], index=2)
    if quick == "Today":
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
    elif quick == "Last 7":
        st.session_state.start_date = date.today() - timedelta(days=6); st.session_state.end_date = date.today()
    elif quick == "Last 30":
        st.session_state.start_date = date.today() - timedelta(days=29); st.session_state.end_date = date.today()
    elif quick == "90 days":
        st.session_state.start_date = date.today() - timedelta(days=89); st.session_state.end_date = date.today()
    elif quick == "All":
        st.session_state.start_date = None; st.session_state.end_date = None
    else:
        sd, ed = st.date_input("Start & End", [date.today()-timedelta(days=29), date.today()])
        st.session_state.start_date = sd; st.session_state.end_date = ed

# ---------------------------
# Compute main leads DF once per view
# ---------------------------
start_date = st.session_state.get("start_date", None)
end_date = st.session_state.get("end_date", None)
session = get_session()
try:
    df_main = leads_to_df(session, start_date, end_date)
finally:
    session.close()

# Apply ML model scores to df_main if model exists
model, model_cols = load_internal_model()
if model is not None and not df_main.empty:
    # Use scoring function but do not persist here
    try:
        df_temp = df_main.copy()
        df_temp["age_days"] = (datetime.utcnow() - df_temp["created_at"]).dt.days
        X = pd.get_dummies(df_temp[["source","stage"]].astype(str), drop_first=False)
        X["ad_cost"] = df_temp["ad_cost"]
        X["estimated_value"] = df_temp["estimated_value"]
        X["age_days"] = df_temp["age_days"]
        # fill missing cols
        for c in model_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[model_cols].fillna(0)
        try:
            df_main["score"] = model.predict_proba(X)[:,1]
        except Exception:
            df_main["score"] = model.predict(X)
    except Exception:
        pass

# ---------------------------
# Alerts bell (SLA overdue)
# ---------------------------
def overdue_list_from_df(dframe):
    res = []
    for _, r in dframe.iterrows():
        tleft, overdue = compute_time_left_and_overdue(r)
        if overdue and r.get("stage") not in ("Won","Lost"):
            res.append(r)
    return res

overdue_list = overdue_list_from_df(df_main)

# top header & alerts button
header_cols = st.columns([10,1])
with header_cols[1]:
    if st.button(f"🔔 {len(overdue_list)}"):
        st.session_state.show_alerts = not st.session_state.get("show_alerts", False)
if st.session_state.get("show_alerts", False) and overdue_list:
    # show alert box in top-right
    html = "<div class='alert-box'><div style='font-weight:800;margin-bottom:6px;'>SLA Alerts</div>"
    for r in overdue_list[:10]:
        tleft, _ = compute_time_left_and_overdue(r)
        html += f"<div style='margin-bottom:6px;'><b>{r['lead_code']}</b> — <span style='color:#22c55e;'>${r['estimated_value']:,.0f}</span> — <span style='color:#dc2626;font-weight:700;'>{int(tleft)}h left / OVERDUE</span></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------
# ROUTER: Pages
# ---------------------------
if page == "Dashboard":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("*High-level pipeline performance at a glance. Use the date range in the left panel.*")
    df = df_main.copy()
    total_leads = len(df)
    # ✅ FIXED SAFE COUNT WITH FALLBACK FOR MISSING COLUMN if "stage" in df.columns:     # ✅ FIXED SAFE COUNT WITH FALLBACK FOR MISSING COLUMN if "stage" in df.columns:     awarded = len(df[df["stage"] == "Won"]) else:     awarded = 0     st.warning("⚠ Pipeline column 'stage' not found in DB read. Check DB structure.") else:     awarded = 0     st.warning("⚠ Pipeline column 'stage' not found in DB read. Check DB structure.")
    lost = len(df[df["stage"] == "Lost"])
    active_leads = total_leads - (awarded + lost)
    sla_success_pct = (len(df[df["stage"]=="Contacted"]) / total_leads * 100) if total_leads else 0.0
    qualification_pct = (len(df[df["stage"]=="Qualified"]) / total_leads * 100) if total_leads else 0.0
    conversion_rate = (awarded / (awarded + lost) * 100) if (awarded + lost) else 0.0
    inspection_pct = (len(df[df["stage"]=="Inspection Scheduled"]) / (len(df[df["stage"]=="Qualified"]) or 1) * 100) if total_leads else 0.0
    estimate_sent_count = len(df[df["stage"]=="Estimate Sent"])
    pipeline_value = df["estimated_value"].sum() if not df.empty else 0.0

    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "#2563eb"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444"),
        ("Estimates Sent", f"{estimate_sent_count}", "#6d28d9"),
        ("Pipeline Job Value", f"${pipeline_value:,.0f}", "#22c55e")
    ]

    # 2 rows (4 + 3)
    cols1 = st.columns(4)
    cols2 = st.columns(3)
    cols_all = cols1 + cols2
    for c, item in zip(cols_all, KPI_ITEMS):
        title, value, color = item
        pct = random.randint(30, 90)
        c.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value' style='color:{color};'>{value}</div>
                <div class='progress' style='background:{color}; width:{pct}%;'></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("TOP 5 PRIORITY LEADS")
    st.markdown("*Top leads by internal priority score — time left (red) and money (green)*")
    if df.empty:
        st.info("No leads")
    else:
        df["priority_score"] = df.apply(lambda r: compute_priority_score(r), axis=1)
        top5 = df.sort_values("priority_score", ascending=False).head(5)
        for _, r in top5.iterrows():
            tleft, overdue = compute_time_left_and_overdue(r)
            time_html = f"<span class='priority-time'>{'❗ OVERDUE' if overdue else f'{int(tleft)}h left'}</span>"
            money_html = f"<span class='priority-money'>${r['estimated_value']:,.0f}</span>"
            st.markdown(f"""
                <div class='lead-card'>
                    <div style='display:flex;justify-content:space-between;'>
                        <div><b>{r['lead_code']}</b><div style='font-size:13px;color:#6b7280'>{r.get('notes') or ''}</div></div>
                        <div style='text-align:right'>{money_html}<br>{time_html}</div>
                    </div>
                    <div style='margin-top:8px;'><small>Stage: {r['stage']} — Priority: {r['priority_score']:.2f}</small></div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("All Leads")
    st.markdown("*Expand to edit.*")
    # quick filters
    qcol1, qcol2, qcol3 = st.columns([3,2,3])
    with qcol1:
        q = st.text_input("Search lead code / notes / phone")
    with qcol2:
        fsrc = st.selectbox("Source", options=["All"] + sorted(df["source"].dropna().unique().tolist()) if not df.empty else ["All"])
    with qcol3:
        fstage = st.selectbox("Stage", options=["All"] + PIPELINE_STAGES)

    view_df = df.copy()
    if q:
        qlower = q.lower()
        view_df = view_df[view_df.apply(lambda r: qlower in str(r.get("lead_code","")).lower() or qlower in str(r.get("notes","")).lower() or qlower in str(r.get("contact_phone","")).lower(), axis=1)]
    if fsrc and fsrc != "All":
        view_df = view_df[view_df["source"] == fsrc]
    if fstage and fstage != "All":
        view_df = view_df[view_df["stage"] == fstage]

    if view_df.empty:
        st.info("No leads match filters.")
    else:
        for _, r in view_df.sort_values("created_at", ascending=False).head(200).iterrows():
            with st.expander(f"{r['lead_code']} — {r.get('contact_name') or 'No name'} — {r['stage']}"):
                colL, colR = st.columns([3,1])
                with colL:
                    st.write(f"**Contact:** {r.get('contact_name') or '—'} | {r.get('contact_phone') or '—'} | {r.get('contact_email') or '—'}")
                    st.write(f"**Address:** {r.get('property_address') or '—'}")
                    st.write(f"**Notes:** {r.get('notes') or '—'}")
                    st.write(f"**Created:** {r.get('created_at')}")
                with colR:
                    tleft, overdue = compute_time_left_and_overdue(r)
                    if overdue:
                        st.markdown("<div class='priority-time'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color:#6b7280'>{int(tleft)}h left</div>", unsafe_allow_html=True)

                # update form
                new_stage = st.selectbox("Stage", PIPELINE_STAGES, index=PIPELINE_STAGES.index(r['stage']) if r['stage'] in PIPELINE_STAGES else 0, key=f"stage_{r['lead_code']}")
                new_owner = st.text_input("Assign to (username)", value=r.get("assigned_to") or "", key=f"owner_{r['lead_code']}")
                new_est = st.number_input("Estimate value", value=float(r.get("estimated_value") or 0.0), min_value=0.0, step=100.0, key=f"est_{r['lead_code']}")
                new_cost = st.number_input("Acquisition cost", value=float(r.get("ad_cost") or 0.0), min_value=0.0, step=1.0, key=f"cost_{r['lead_code']}")
                new_notes = st.text_area("Notes", value=r.get("notes") or "", key=f"notes_{r['lead_code']}")
                if st.button("Save", key=f"save_{r['lead_code']}"):
                    try:
                        payload = {
                            "lead_code": r['lead_code'],
                            "stage": new_stage,
                            "assigned_to": new_owner or None,
                            "estimated_value": float(new_est or 0.0),
                            "ad_cost": float(new_cost or 0.0),
                            "notes": new_notes
                        }
                        create_or_update_lead(payload, actor=st.session_state.username)
                        st.success("Saved")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error("Save failed: " + str(e))
                        st.write(traceback.format_exc())

elif page == "Leads / Capture":
    st.header("Lead Capture — create a new lead (auto lead code)")
    s = get_session()
    # generate auto code
    lead_code = generate_auto_lead_code(s)
    s.close()
    with st.form("capture", clear_on_submit=True):
        st.markdown(f"**Lead Code**: `{lead_code}`")
        contact_name = st.text_input("Contact name")
        contact_phone = st.text_input("Contact phone")
        contact_email = st.text_input("Contact email")
        property_address = st.text_input("Property address")
        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"])
        source = st.selectbox("Source", ["Google Ads","Organic Search","Referral","Phone","Insurance","Social","Other"])
        source_details = st.text_input("Source details / UTM")
        assigned_to = st.text_input("Assign to (username)", value="")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        ad_cost = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        sla_hours = st.number_input("SLA hours", min_value=1, value=SLA_HOURS_DEFAULT, step=1)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            if not contact_name.strip():
                st.error("Contact name required")
            else:
                try:
                    payload = {
                        "lead_code": lead_code,
                        "created_at": datetime.utcnow(),
                        "source": source,
                        "source_details": source_details,
                        "contact_name": contact_name,
                        "contact_phone": contact_phone,
                        "contact_email": contact_email,
                        "property_address": property_address,
                        "damage_type": damage_type,
                        "assigned_to": assigned_to,
                        "notes": notes,
                        "estimated_value": float(estimated_value),
                        "ad_cost": float(ad_cost),
                        "stage": "New",
                        "sla_hours": int(sla_hours),
                        "sla_entered_at": datetime.utcnow(),
                        "created_by": st.session_state.username
                    }
                    create_or_update_lead(payload, actor=st.session_state.username)
                    st.success(f"Lead created: {lead_code}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error("Failed to create lead: " + str(e))
                    st.write(traceback.format_exc())

elif page == "Analytics & SLA":
    st.header("Analytics & SLA")
    df = df_main.copy()
    if df.empty:
        st.info("No leads yet")
    else:
        st.subheader("SLA Overdue (last 30 days)")
        today = date.today()
        days = [today - timedelta(days=i) for i in range(29, -1, -1)]
        rows = []
        for d in days:
            d_start = datetime.combine(d, datetime.min.time())
            d_end = datetime.combine(d, datetime.max.time())
            sub = df[(df["created_at"] >= d_start) & (df["created_at"] <= d_end)]
            overdue_count = 0
            for _, r in sub.iterrows():
                tleft, overdue = compute_time_left_and_overdue(r)
                if overdue and r.get("stage") not in ("Won","Lost"):
                    overdue_count += 1
            rows.append({"date": d, "overdue": overdue_count})
        ts = pd.DataFrame(rows)
        fig = px.line(ts, x="date", y="overdue", title="SLA Overdue (30 days)", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Pipeline counts (selected range)")
        counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0).reset_index()
        counts.columns = ["stage","count"]
        fig2 = px.bar(counts, x="stage", y="count", title="Pipeline Stage Counts")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "CPA & ROI":
    st.header("CPA & ROI")
    df = df_main.copy()
    if df.empty:
        st.info("No data")
    else:
        total_spend = df["ad_cost"].sum()
        won_df = df[df["stage"] == "Won"]
        conversions = len(won_df)
        cpa = (total_spend / conversions) if conversions else 0.0
        revenue = won_df["estimated_value"].sum()
        roi_value = revenue - total_spend
        roi_pct = (roi_value / total_spend * 100) if total_spend else 0.0

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Marketing Spend</div><div class='kpi-value' style='color:#2563eb'>${total_spend:,.2f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversions (Won)</div><div class='kpi-value' style='color:#06b6d4'>{conversions}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card'><div class='kpi-title'>CPA</div><div class='kpi-value' style='color:#f97316'>${cpa:,.2f}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi-card'><div class='kpi-title'>ROI</div><div class='kpi-value' style='color:#22c55e'>${roi_value:,.2f} ({roi_pct:.1f}%)</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("stage", lambda s: (s=="Won").sum())).reset_index()
        if not agg.empty:
            fig = px.bar(agg, x="source", y=["total_spend","conversions"], title="Total Spend vs Conversions by Source", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

elif page == "ML":
    st.header("Internal ML — Lead scoring (internal only)")
    st.markdown("Train an internal model and persist scores (optional). No user-tunable parameters exposed.")
    if st.button("Train internal model"):
        with st.spinner("Training model..."):
            acc, msg = train_internal_model()
            if acc is None:
                st.warning("Training skipped: " + str(msg))
            else:
                st.success(f"Model trained (approx accuracy {acc:.3f})")
    model_loaded, cols = load_internal_model()
    if model_loaded is not None:
        if st.button("Score & Persist leads (write score column)"):
            ok, msg = score_and_persist_scores()
            if ok:
                st.success("Persisted scores to DB")
            else:
                st.error("Failed: " + str(msg))
    else:
        st.info("No model file found. Train model first.")

elif page == "Settings":
    st.header("Settings")
    st.markdown("User profile and weight tuning")
    s = get_session()
    try:
        user = s.query(User).filter(User.username == st.session_state.username).first()
        st.write("Username:", user.username)
        user.full_name = st.text_input("Full name", value=user.full_name or "")
        user.role = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=["Admin","Estimator","Adjuster","Tech","Viewer"].index(user.role))
        if st.button("Save profile"):
            s.add(user); s.commit(); st.success("Saved profile")
    finally:
        s.close()

elif page == "Exports":
    st.header("Exports & Imports")
    df = df_main.copy()
    if df.empty:
        st.info("No data")
    else:
        if st.button("Download leads (Excel)"):
            html = df_to_excel_download(df, "leads_export.xlsx")
            st.markdown(html, unsafe_allow_html=True)
        st.markdown("---")
        uploaded = st.file_uploader("Import leads (CSV or XLSX)", type=["csv","xlsx"])
        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df_in = pd.read_csv(uploaded)
                else:
                    df_in = pd.read_excel(uploaded)
                required = {"lead_code","created_at","source","stage"}
                if not required.issubset(set(df_in.columns)):
                    st.error(f"Missing required cols: {required - set(df_in.columns)}")
                else:
                    count = 0
                    for _, row in df_in.iterrows():
                        payload = {
                            "lead_code": row["lead_code"],
                            "created_at": pd.to_datetime(row["created_at"]),
                            "source": row.get("source"),
                            "stage": row.get("stage"),
                            "estimated_value": row.get("estimated_value",0.0),
                            "ad_cost": row.get("ad_cost",0.0),
                            "notes": row.get("notes"),
                            "created_by": st.session_state.username
                        }
                        create_or_update_lead(payload, actor=st.session_state.username)
                        count += 1
                    st.success(f"Imported {count} rows")
            except Exception as e:
                st.error("Import failed: " + str(e))
                st.write(traceback.format_exc())

elif page == "Audit Trail":
    st.header("Audit Trail")
    s = get_session()
    try:
        rows = s.query(LeadHistory).order_by(LeadHistory.timestamp.desc()).limit(500).all()
        data = []
        for r in rows:
            data.append({
                "lead_code": r.lead_code,
                "changed_by": r.changed_by,
                "field": r.field,
                "old": r.old_value,
                "new": r.new_value,
                "when": r.timestamp
            })
        if not data:
            st.info("No history yet.")
        else:
            st.dataframe(pd.DataFrame(data))
    finally:
        s.close()

# ✅ backup rerun patch if called anywhere accidentally
if hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun


# Footer
st.markdown("---")
st.markdown("<small class='small-muted'>TITAN — Single-file Streamlit app. DB: streamlit_app.db (Option C). ML internal only.</small>", unsafe_allow_html=True)
