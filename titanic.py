# titan_backend.py
"""
TITAN Backend - Single-file Streamlit app
- No front-page login: this is an admin backend (Admin access by default)
- User & Role management available in Settings
- SQLite persistence via SQLAlchemy
- Internal ML training & scoring (no user tuning)
- Pipeline dashboard, Analytics, CPA/ROI, Exports/Imports, Alerts, SLA, Priority scoring, Audit trail
"""

import os
from datetime import datetime, timedelta, date
import io, base64, traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------
# CONFIG
# ----------------------
APP_TITLE = "ReCapture Pro"
DB_FILE = "titan_backend.db"   # stored in app working directory
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = [
    "New", "Contacted", "Inspection Scheduled", "Inspection Completed",
    "Estimate Sent", "Qualified", "Won", "Lost"
]
DEFAULT_SLA_HOURS = 72
COMFORTAA_IMPORT = "https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap"

# KPI colors (numbers)
KPI_COLORS = ["#2563eb", "#0ea5a4", "#a855f7", "#f97316", "#ef4444", "#6d28d9", "#22c55e"]

# ----------------------
# DB SETUP
# ----------------------
DB_PATH = os.path.join(os.getcwd(), DB_FILE)
ENGINE_URL = f"sqlite:///{DB_PATH}"

# SQLAlchemy engine and session factory
engine = create_engine(ENGINE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# ----------------------
# MODELS
# ----------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="Admin")  # Admin by default for backend
    created_at = Column(DateTime, default=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Other")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)  # username of owner
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, default=0.0)
    stage = Column(String, default="New")
    sla_hours = Column(Integer, default=DEFAULT_SLA_HOURS)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)
    ad_cost = Column(Float, default=0.0)  # cost to acquire
    converted = Column(Boolean, default=False)
    score = Column(Float, nullable=True)  # ML probability

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # ---------- BEGIN BLOCK A: NEW MODELS (Technician, InspectionAssignment, LocationPing) ----------
from sqlalchemy import DateTime as SA_DateTime

class Technician(Base):
    __tablename__ = "technicians"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)  # ideally matches User.username
    full_name = Column(String, default="")
    phone = Column(String, nullable=True)
    specialization = Column(String, nullable=True)  # e.g., 'Estimator','Adjuster','Tech'
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class InspectionAssignment(Base):
    __tablename__ = "inspection_assignments"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)          # lead_id from Lead.lead_id
    technician_username = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="assigned")      # assigned, enroute, onsite, completed, cancelled
    notes = Column(Text, nullable=True)

class LocationPing(Base):
    __tablename__ = "location_pings"
    id = Column(Integer, primary_key=True)
    tech_username = Column(String, nullable=False)
    lead_id = Column(String, nullable=True)          # optional - link to lead if assigned
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, nullable=True)          # optional accuracy (meters)
# ---------- END BLOCK A ----------


# Create tables if missing
from sqlalchemy import inspect

def safe_create_tables():
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            table.create(bind=engine)

safe_create_tables()


# Safe migration attempt (best-effort add missing columns)
def safe_migrate():
    try:
        inspector = inspect(engine)
        if "leads" in inspector.get_table_names():
            existing = [c['name'] for c in inspector.get_columns("leads")]
            desired = {
                "score": "FLOAT",
                "ad_cost": "FLOAT",
                "source_details": "TEXT",
                "contact_name": "TEXT",
                "assigned_to": "TEXT",
            }
            conn = engine.connect()
            for col, typ in desired.items():
                if col not in existing:
                    try:
                        conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {typ}")
                    except Exception:
                        pass
            conn.close()
    except Exception:
        pass

safe_migrate()
# ---------- BEGIN BLOCK B: SAFE MIGRATION / CREATE NEW TABLES ----------
def safe_migrate_new_tables():
    try:
        inspector = inspect(engine)
        # If tables don't exist in DB, create them via metadata.create_all()
        Base.metadata.create_all(bind=engine)
    except Exception:
        pass

# Run it once on startup (safe)
safe_migrate_new_tables()
# ---------- END BLOCK B ----------

# ----------------------
# HELPERS: DB ops
# ----------------------
def get_session():
    return SessionLocal()

def leads_to_df(start_date=None, end_date=None):
    """Load leads into a DataFrame. Filter by optional start_date/end_date (date objects)"""
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "lead_id": r.lead_id,
                "created_at": r.created_at,
                "source": r.source or "Other",
                "source_details": getattr(r, "source_details", None),
                "contact_name": getattr(r, "contact_name", None),
                "contact_phone": getattr(r, "contact_phone", None),
                "contact_email": getattr(r, "contact_email", None),
                "property_address": getattr(r, "property_address", None),
                "damage_type": getattr(r, "damage_type", None),
                "assigned_to": getattr(r, "assigned_to", None),
                "notes": r.notes,
                "estimated_value": float(r.estimated_value or 0.0),
                "stage": r.stage or "New",
                "sla_hours": int(r.sla_hours or DEFAULT_SLA_HOURS),
                "sla_entered_at": r.sla_entered_at or r.created_at,
                "contacted": bool(r.contacted),
                "inspection_scheduled": bool(r.inspection_scheduled),
                "inspection_scheduled_at": r.inspection_scheduled_at,
                "inspection_completed": bool(r.inspection_completed),
                "estimate_submitted": bool(r.estimate_submitted),
                "awarded_date": r.awarded_date,
                "lost_date": r.lost_date,
                "qualified": bool(r.qualified),
                "ad_cost": float(r.ad_cost or 0.0),
                "converted": bool(r.converted),
                "score": float(r.score) if r.score is not None else None
            })
        df = pd.DataFrame(data)
        if df.empty:
            # return empty with expected columns
            cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email",
                    "property_address","damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                    "contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted",
                    "awarded_date","lost_date","qualified","ad_cost","converted","score"]
            return pd.DataFrame(columns=cols)
        # apply date filters
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        return df.reset_index(drop=True)
    finally:
        s.close()
# ---------- BEGIN BLOCK C: DB HELPERS FOR TECHNICIANS / ASSIGNMENTS / PINGS ----------
def add_technician(username: str, full_name: str = "", phone: str = "", specialization: str = "Tech", active: bool = True):
    s = get_session()
    try:
        existing = s.query(Technician).filter(Technician.username == username).first()
        if existing:
            existing.full_name = full_name
            existing.phone = phone
            existing.specialization = specialization
            existing.active = active
            s.add(existing); s.commit()
            return existing.username
        t = Technician(username=username, full_name=full_name, phone=phone, specialization=specialization, active=active)
        s.add(t); s.commit()
        return t.username
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def get_technicians_df(active_only=True):
    s = get_session()
    try:
        q = s.query(Technician)
        if active_only:
            q = q.filter(Technician.active == True)
        rows = q.order_by(Technician.created_at.desc()).all()
        return pd.DataFrame([{"username": r.username, "full_name": r.full_name, "phone": r.phone, "specialization": r.specialization, "active": r.active} for r in rows])
    finally:
        s.close()

def create_inspection_assignment(lead_id: str, technician_username: str, notes: str = None):
    s = get_session()
    try:
        ia = InspectionAssignment(lead_id=lead_id, technician_username=technician_username, notes=notes)
        s.add(ia)
        # optional: also set lead.assigned_to to technician_username (non-destructive)
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if lead:
            lead.assigned_to = technician_username
            s.add(lead)
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by="system", field="assigned_to", old_value=str(getattr(lead, "assigned_to", "")), new_value=str(technician_username)))
        s.commit()
        return ia.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def get_assignments_for_lead(lead_id: str):
    s = get_session()
    try:
        rows = s.query(InspectionAssignment).filter(InspectionAssignment.lead_id == lead_id).order_by(InspectionAssignment.assigned_at.desc()).all()
        return rows
    finally:
        s.close()

def persist_location_ping(tech_username: str, latitude: float, longitude: float, lead_id: str = None, accuracy: float = None, timestamp: datetime = None):
    s = get_session()
    try:
        ping = LocationPing(tech_username=tech_username, latitude=float(latitude), longitude=float(longitude), lead_id=lead_id, accuracy=accuracy, timestamp=timestamp or datetime.utcnow())
        s.add(ping)
        s.commit()
        return ping.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
# ---------- END BLOCK C ----------


def upsert_lead_record(payload: dict, actor="admin"):
    """
    payload must include lead_id (string)
    other fields optional
    """
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == payload.get("lead_id")).first()
        if lead is None:
            # create
            lead = Lead(
                lead_id=payload.get("lead_id"),
                created_at=payload.get("created_at", datetime.utcnow()),
                source=payload.get("source"),
                source_details=payload.get("source_details"),
                contact_name=payload.get("contact_name"),
                contact_phone=payload.get("contact_phone"),
                contact_email=payload.get("contact_email"),
                property_address=payload.get("property_address"),
                damage_type=payload.get("damage_type"),
                assigned_to=payload.get("assigned_to"),
                notes=payload.get("notes"),
                estimated_value=float(payload.get("estimated_value") or 0.0),
                stage=payload.get("stage") or "New",
                sla_hours=int(payload.get("sla_hours") or DEFAULT_SLA_HOURS),
                sla_entered_at=payload.get("sla_entered_at") or datetime.utcnow(),
                ad_cost=float(payload.get("ad_cost") or 0.0),
                converted=bool(payload.get("converted") or False),
                score=payload.get("score")
            )
            s.add(lead)
            s.commit()
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field="create", old_value=None, new_value=str(lead.stage)))
            s.commit()
            return lead.lead_id
        else:
            # update fields and log changes
            changed = []
            for key in ["source","source_details","contact_name","contact_phone","contact_email","property_address",
                        "damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at","ad_cost","converted","score"]:
                if key in payload:
                    new = payload.get(key)
                    old = getattr(lead, key)
                    # normalize numeric conversions
                    if key in ("estimated_value","ad_cost"):
                        try:
                            new_val = float(new or 0.0)
                        except Exception:
                            new_val = old
                    elif key in ("sla_hours",):
                        try:
                            new_val = int(new or old)
                        except Exception:
                            new_val = old
                    elif key in ("converted",):
                        new_val = bool(new)
                    else:
                        new_val = new
                    if new_val is not None and old != new_val:
                        changed.append((key, old, new_val))
                        setattr(lead, key, new_val)
            # persist
            s.add(lead)
            for (f, old, new) in changed:
                s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field=f, old_value=str(old), new_value=str(new)))
            s.commit()
            return lead.lead_id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def delete_lead_record(lead_id: str, actor="admin"):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if not lead:
            return False
        s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field="delete", old_value=str(lead.stage), new_value="deleted"))
        s.delete(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def get_users_df():
    s = get_session()
    try:
        users = s.query(User).order_by(User.created_at.desc()).all()
        data = [{"id":u.id, "username":u.username, "full_name":u.full_name, "role":u.role, "created_at":u.created_at} for u in users]
        return pd.DataFrame(data)
    finally:
        s.close()

def add_user(username: str, full_name: str = "", role: str = "Admin"):
    s = get_session()
    try:
        existing = s.query(User).filter(User.username == username).first()
        if existing:
            existing.full_name = full_name
            existing.role = role
            s.add(existing); s.commit()
            return existing.username
        u = User(username=username, full_name=full_name, role=role)
        s.add(u); s.commit()
        return u.username
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# ----------------------
# ML - internal only
# ----------------------
def train_internal_model():
    df = leads_to_df()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labeled data to train"
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    y = df2["converted"].astype(int)
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    return acc, "trained"

def load_internal_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    try:
        obj = joblib.load(MODEL_FILE)
        return obj.get("model"), obj.get("columns")
    except Exception:
        return None, None

def score_dataframe(df, model, cols):
    if model is None or df.empty:
        df["score"] = np.nan
        return df
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
        df["score"] = model.predict_proba(X)[:,1]
    except Exception:
        df["score"] = model.predict(X)
    return df

# ----------------------
# Priority & SLA utilities
# ----------------------
def calculate_remaining_sla(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        remain = deadline - datetime.utcnow()
        return max(remain.total_seconds(), 0.0), (remain.total_seconds() <= 0)
    except Exception:
        return float("inf"), False

def compute_priority_for_row(row, weights=None):
    # row: Series/dict
    if weights is None:
        weights = {"score_w":0.6, "value_w":0.3, "sla_w":0.1, "value_baseline":5000.0}
    try:
        s = float(row.get("score") or 0.0)
    except Exception:
        s = 0.0
    try:
        val = float(row.get("estimated_value") or 0.0)
        vnorm = min(1.0, val / max(1.0, weights["value_baseline"]))
    except Exception:
        vnorm = 0.0
    try:
        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        if sla_entered is None:
            sla_score = 0.0
        else:
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            time_left_h = max((sla_entered + timedelta(hours=row.get("sla_hours") or DEFAULT_SLA_HOURS) - datetime.utcnow()).total_seconds()/3600.0, 0.0)
            sla_score = max(0.0, (72.0 - min(time_left_h,72.0)) / 72.0)
    except Exception:
        sla_score = 0.0
    total = s*weights["score_w"] + vnorm*weights["value_w"] + sla_score*weights["sla_w"]
    return max(0.0, min(1.0, total))

# ----------------------
# UI CSS and layout
# ----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"<link href='{COMFORTAA_IMPORT}' rel='stylesheet'>", unsafe_allow_html=True)

APP_CSS = """
<style>
body, .stApp { background: #ffffff; color: #0b1220; font-family: 'Comfortaa', sans-serif; }
.header { font-weight:800; font-size:20px; margin-bottom:6px; }
.kpi-grid { display:flex; gap:12px; flex-wrap:wrap; }
.kpi-card {
    background:#000;
    color:white;
    border-radius:12px;
    padding:12px;
    min-width:220px;
    box-shadow:0 8px 22px rgba(16,24,40,0.06);

    /* ✅ add spacing */
    margin-top: 12px;
    margin-bottom: 12px;
}

.kpi-title { font-size:12px; opacity:0.95; margin-bottom:6px; }
.kpi-number { font-size:22px; font-weight:900; margin-bottom:8px; }
.progress-bar { height:8px; border-radius:8px; background:#e6e6e6; overflow:hidden; }
.progress-fill { height:100%; border-radius:8px; transition:width .4s ease; }
.lead-card {
    background: #000000; /* ✅ Black */
    color: #ffffff; /* text stays white */
    border:1px solid #eef2ff;
    border-radius:10px;
    padding:12px;
    margin-bottom:8px;
}

.priority-time { color:#dc2626; font-weight:700; }
.priority-money { color:#22c55e; font-weight:800; }
.alert-bubble { background:#111; color:white; padding:10px; border-radius:10px; }
.small-muted { color:#F5F5F5; font-size:12px; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ----------------------
# Sidebar controls (Admin backend - no front login)
# ----------------------
with st.sidebar:
    st.header("TITAN Backend (Admin)")
    st.markdown("You are using the backend admin interface. User accounts and roles are managed in Settings.")
    page = st.radio(
    "Navigate", 
    [
        "Dashboard",
        "Lead Capture",
        "Pipeline Board",
        "Analytics",
        "CPA & ROI",
        "ML (internal)",
        "AI Recommendations",   # ← ADD THIS
        "Settings",
        "Exports"
    ], 
    index=0
)

    st.markdown("---")
    st.markdown("Date range for reports")
    quick = st.selectbox("Quick range", ["Today","Last 7 days","Last 30 days","90 days","All","Custom"], index=4)
    if quick == "Today":
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
    elif quick == "Last 7 days":
        st.session_state.start_date = date.today() - timedelta(days=6)
        st.session_state.end_date = date.today()
    elif quick == "Last 30 days":
        st.session_state.start_date = date.today() - timedelta(days=29)
        st.session_state.end_date = date.today()
    elif quick == "90 days":
        st.session_state.start_date = date.today() - timedelta(days=89)
        st.session_state.end_date = date.today()
    elif quick == "All":
        st.session_state.start_date = None
        st.session_state.end_date = None
    else:
        sd, ed = st.date_input("Start / End", [date.today() - timedelta(days=29), date.today()])
        st.session_state.start_date = sd
        st.session_state.end_date = ed

    st.markdown("---")
    st.markdown("Internal ML runs silently. Use ML page to train/score.")
    if st.button("Refresh data"):
        # clear caches and refresh
        try:
            st.rerun()
        except Exception:
            pass

# Utility: date filters
start_dt = st.session_state.get("start_date", None)
end_dt = st.session_state.get("end_date", None)

# Load leads
try:
    leads_df = leads_to_df(start_dt, end_dt)
except OperationalError as exc:
    st.error("Database error — ensure file is writable and accessible.")
    st.stop()

# Load model (if exists)
model, model_cols = load_internal_model()
if model is not None and not leads_df.empty:
    try:
        leads_df = score_dataframe(leads_df.copy(), model, model_cols)
    except Exception:
        # if scoring fails, continue without scores
        pass

# ----------------------
# Alerts bell (top-right)
# ----------------------
def alerts_ui():
    overdue = []
    for _, r in leads_df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(
            r.get("sla_entered_at") or r.get("created_at"),
            r.get("sla_hours")
        )
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue.append(r)

    if overdue:
        col1, col2 = st.columns([1, 10])

        # BELL BUTTON (needs unique key)
        with col1:
            if st.button(f"🔔 {len(overdue)}", key="alerts_button"):
                st.session_state.show_alerts = not st.session_state.get(
                    "show_alerts", False
                )

        with col2:
            st.markdown("")

        # EXPANDED ALERTS POPUP
        if st.session_state.get("show_alerts", False):
            with st.expander("SLA Alerts (click to close)", expanded=True):

                # list overdue leads
                for r in overdue:
                    st.markdown(
                        f"**{r['lead_id']}** — Stage: {r['stage']} — "
                        f"<span style='color:#22c55e;'>${r['estimated_value']:,.0f}</span> — "
                        f"<span style='color:#dc2626;'>OVERDUE</span>",
                        unsafe_allow_html=True
                    )

                # CLOSE BUTTON (needs unique key)
                if st.button("Close Alerts", key="alerts_close_button"):
                    st.session_state.show_alerts = False

    else:
        st.markdown("")


# ----------------------
# PAGES
# ----------------------

# Dashboard: KPI cards + Top 5 priority + All leads expanded editor
def page_dashboard():
    st.markdown("<div class='header'>TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR</div>", unsafe_allow_html=True)
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)
    alerts_ui()

    df = leads_df.copy()
    total_leads = len(df)
    qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
    sla_success_count = int(df[df["contacted"] == True].shape[0]) if not df.empty else 0
    awarded_count = int(df[df["stage"] == "Won"].shape[0]) if not df.empty else 0
    lost_count = int(df[df["stage"] == "Lost"].shape[0]) if not df.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_count = int(df[df["inspection_scheduled"] == True].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"] == True].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0

    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", KPI_COLORS[0], "Leads currently in pipeline"),
        ("SLA Success", f"{sla_success_pct:.1f}%", KPI_COLORS[1], "Leads contacted within SLA"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", KPI_COLORS[2], "Leads marked qualified"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", KPI_COLORS[3], "Won / Closed"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", KPI_COLORS[4], "Qualified → Scheduled"),
        ("Estimates Sent", f"{estimate_sent_count}", KPI_COLORS[5], "Estimates submitted"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", KPI_COLORS[6], "Total pipeline job value")
    ]

    # render 2 rows: first 4 then next 3
    r1 = st.columns(4)
    r2 = st.columns(3)
    cols = r1 + r2
    for col, (title, value, color, note) in zip(cols, KPI_ITEMS):
        pct = min(100, max(10, (hash(title) % 80) + 20))  # deterministic-ish
        col.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-title'>{title}</div>
              <div class='kpi-number' style='color:{color};'>{value}</div>
              <div class='progress-bar'><div class='progress-fill' style='width:{pct}%; background:{color};'></div></div>
              <div class='small-muted'>{note}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Lead Pipeline Stages")
    st.markdown("<em>Distribution of leads across pipeline stages.</em>", unsafe_allow_html=True)
    if df.empty:
        st.info("No leads yet. Create one in Lead Capture.")
    else:
        stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
        pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
        fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
#NEW IMPLEMENTATION FROM LINE 581 - 607 CHATGPpjx
    st.markdown("---")
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
    if df.empty:
        st.info("No priority leads to display.")
    else:
        df["priority_score"] = df.apply(lambda r: compute_priority_for_row(r), axis=1)
        pr_df = df.sort_values("priority_score", ascending=False).head(5)
        for _, r in pr_df.iterrows():
            sla_sec, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            hleft = int(sla_sec / 3600) if sla_sec not in (None, float("inf")) else 9999
            sla_html = f"<span class='priority-time'>❗ OVERDUE</span>" if overdue else f"<span class='small-muted'>⏳ {hleft}h left</span>"
            val_html = f"<span class='priority-money'>${r['estimated_value']:,.0f}</span>"
            st.markdown(f"""
                <div class='lead-card'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                      <div style='font-weight:800;'>#{r['lead_id']} — {r.get('contact_name') or 'No name'}</div>
                      <div class='small-muted'>{r.get('damage_type') or ''} • {r.get('source') or ''}</div>
                    </div>
                    <div style='text-align:right;'>
                      <div style='font-size:20px; font-weight:900; color:#111;'>{r['priority_score']:.2f}</div>
                      <div style='margin-top:8px;'>{val_html}<br>{sla_html}</div>
                    </div>
                  </div>
                </div>
            """, unsafe_allow_html=True)





    st.markdown("---")
    st.markdown("### 📋 All Leads (expand a card to edit / change status)")
    st.markdown("<em>Expand a lead to edit details, change status, assign owner, and create estimates.</em>", unsafe_allow_html=True)
    # Quick filters
    q1, q2, q3 = st.columns([3,2,3])
    with q1:
        search_q = st.text_input("Search (lead_id, contact name, address, notes)")
    with q2:
        filter_src = st.selectbox("Source filter", options=["All"] + sorted(df["source"].dropna().unique().tolist()) if not df.empty else ["All"])
    with q3:
        filter_stage = st.selectbox("Stage filter", options=["All"] + PIPELINE_STAGES)
    df_view = df.copy()
    if search_q:
        sq = search_q.lower()
        df_view = df_view[df_view.apply(lambda r: sq in str(r.get("lead_id","")).lower() or sq in str(r.get("contact_name","")).lower() or sq in str(r.get("property_address","")).lower() or sq in str(r.get("notes","")).lower(), axis=1)]
    if filter_src and filter_src != "All":
        df_view = df_view[df_view["source"] == filter_src]
    if filter_stage and filter_stage != "All":
        df_view = df_view[df_view["stage"] == filter_stage]

    if df_view.empty:
        st.info("No leads to show.")
    else:
        for _, lead in df_view.sort_values("created_at", ascending=False).head(200).iterrows():
            exp_key = f"exp_{lead['lead_id']}"
            with st.expander(f"#{lead['lead_id']} — {lead.get('contact_name') or 'No name'} — {lead.get('stage')}", expanded=False):
                left, right = st.columns([3,1])
                with left:
                    st.write(f"**Source:** {lead.get('source') or ''}  |  **Assigned:** {lead.get('assigned_to') or ''}")
                    st.write(f"**Address:** {lead.get('property_address') or ''}")
                    st.write(f"**Contact:** {lead.get('contact_name') or ''} / {lead.get('contact_phone') or ''} / {lead.get('contact_email') or ''}")
                    st.write(f"**Notes:** {lead.get('notes') or ''}")
                    st.write(f"**Created:** {lead.get('created_at')}")
                with right:
                    sla_sec, overdue = calculate_remaining_sla(lead.get("sla_entered_at") or lead.get("created_at"), lead.get("sla_hours"))
                    if overdue:
                        st.markdown("<div style='color:#dc2626;font-weight:700;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        hours = int(sla_sec // 3600)
                        mins = int((sla_sec % 3600) // 60)
                        st.markdown(f"<div class='small-muted'>⏳ {hours}h {mins}m left</div>", unsafe_allow_html=True)
                # update form
                c1, c2 = st.columns(2)
with st.form(f"update_{lead['lead_id']}", clear_on_submit=False):

    new_stage = st.selectbox(
        "Status",
        PIPELINE_STAGES,
        index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0,
        key=f"stage_{lead['lead_id']}"
    )

    new_assigned = st.text_input(
        "Assigned to (username)",
        value=lead.get("assigned_to") or "",
        key=f"assigned_{lead['lead_id']}"
    )

    new_est = st.number_input(
        "Estimated value (USD)",
        value=float(lead.get("estimated_value") or 0.0),
        min_value=0.0,
        step=100.0,
        key=f"estval_{lead['lead_id']}"
    )

    new_cost = st.number_input(
        "Cost to acquire lead (USD)",
        value=float(lead.get("ad_cost") or 0.0),
        min_value=0.0,
        step=1.0,
        key=f"cost_{lead['lead_id']}"
    )

    new_notes = st.text_area(
        "Notes",
        value=lead.get("notes") or "",
        key=f"notes_{lead['lead_id']}"
    )

    # 👍 CORRECT — submit button INSIDE the form
    submitted = st.form_submit_button(
        "Save changes",
        key=f"save_changes_{lead['lead_id']}"
    )

    if submitted:
        try:
            upsert_lead_record({
                "lead_id": lead["lead_id"],
                "stage": new_stage,
                "assigned_to": new_assigned or None,
                "estimated_value": new_est,
                "ad_cost": new_cost,
                "notes": new_notes
            }, actor="admin")

            st.success("Lead updated")
            st.experimental_rerun()

        except Exception as e:
            st.error("Failed to update lead: " + str(e))
            st.write(traceback.format_exc())


# ------------------------------------------------------------
# ---------- BLOCK E: TECHNICIAN ASSIGNMENT (OUTSIDE FORM) ---
# ------------------------------------------------------------

st.markdown("### Technician Assignment")

techs_df = get_technicians_df(active_only=True)
tech_options = [""] + (techs_df["username"].tolist() if not techs_df.empty else [])

selected_tech = st.selectbox(
    "Assign Technician (active)",
    options=tech_options,
    index=0,
    key=f"tech_select_{lead['lead_id']}"
)

assign_notes = st.text_area(
    "Assignment notes (optional)",
    value="",
    key=f"tech_notes_{lead['lead_id']}"
)

if st.button(
    f"Assign Technician to {lead['lead_id']}",
    key=f"assign_btn_{lead['lead_id']}"
):
    if not selected_tech:
        st.error("Select a technician")
    else:
        try:
            create_inspection_assignment(
                lead_id=lead["lead_id"],
                technician_username=selected_tech,
                notes=assign_notes
            )

            st.success(f"Assigned {selected_tech} to lead {lead['lead_id']}")

            upsert_lead_record({
                "lead_id": lead["lead_id"],
                "inspection_scheduled": True,
                "stage": "Inspection Scheduled"
            }, actor="admin")

            st.experimental_rerun()

        except Exception as e:
            st.error("Failed to assign: " + str(e))

# ------------------------------------------------------------
# NEXT PAGE STARTS HERE
# ------------------------------------------------------------

def page_lead_capture():
    st.markdown("<div class='header'>📇 Lead Capture</div>", unsafe_allow_html=True)
    st.markdown("<em>Create or upsert a lead. All inputs are saved for reporting and CPA calculations.</em>", unsafe_allow_html=True)
    with st.form("lead_capture_form", clear_on_submit=True):
        lead_id = st.text_input("Lead ID", value=f"L{int(datetime.utcnow().timestamp())}")
        source = st.selectbox("Lead Source", ["Google Ads","Organic Search","Referral","Phone","Insurance","Facebook","Instagram","LinkedIn","Other"])
        source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google...")
        contact_name = st.text_input("Contact name")
        contact_phone = st.text_input("Contact phone")
        contact_email = st.text_input("Contact email")
        property_address = st.text_input("Property address")
        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"])
        assigned_to = st.text_input("Assigned to (username)")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        ad_cost = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=DEFAULT_SLA_HOURS, step=1)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Create / Update Lead")
        if submitted:
            try:
                upsert_lead_record({
                    "lead_id": lead_id.strip(),
                    "created_at": datetime.utcnow(),
                    "source": source,
                    "source_details": source_details,
                    "contact_name": contact_name,
                    "contact_phone": contact_phone,
                    "contact_email": contact_email,
                    "property_address": property_address,
                    "damage_type": damage_type,
                    "assigned_to": assigned_to or None,
                    "estimated_value": float(estimated_value or 0.0),
                    "ad_cost": float(ad_cost or 0.0),
                    "sla_hours": int(sla_hours or DEFAULT_SLA_HOURS),
                    "sla_entered_at": datetime.utcnow(),
                    "notes": notes
                }, actor="admin")
                st.success(f"Lead {lead_id} saved.")
                st.rerun()
            except Exception as e:
                st.error("Failed to save lead: " + str(e))
                st.write(traceback.format_exc())

    st.markdown("---")
    st.subheader("Recent leads")
    df = leads_to_df(None, None)
    if df.empty:
        st.info("No leads yet.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# Pipeline Board (mostly similar to Dashboard but with card layout)
def page_pipeline_board():
    st.markdown("<div class='header'>PIPELINE BOARD — TOTAL LEAD PIPELINE</div>", unsafe_allow_html=True)
    st.markdown("<em>High-level pipeline board with KPI cards and priority list.</em>", unsafe_allow_html=True)
    alerts_ui()


# Analytics page (donut + SLA line + overdue table)
def page_analytics():
    st.markdown("<div class='header'>📈 Analytics & SLA</div>", unsafe_allow_html=True)
    st.markdown("<em>Donut of pipeline stages + SLA overdue chart and table</em>", unsafe_allow_html=True)
    df = leads_df.copy()
    if df.empty:
        st.info("No leads to analyze.")
        return
    # Donut: pipeline stages
    stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
    pie_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
    fig = px.pie(pie_df, names="stage", values="count", hole=0.45, color="stage")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    # SLA Overdue time series (last 30 days)
    st.subheader("SLA Overdue (last 30 days)")
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(29, -1, -1)]
    ts = []
    for d in days:
        start_dt = datetime.combine(d, datetime.min.time())
        end_dt = datetime.combine(d, datetime.max.time())
        sub = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
        overdue_cnt = 0
        for _, r in sub.iterrows():
            _, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            if overdue and r.get("stage") not in ("Won","Lost"):
                overdue_cnt += 1
        ts.append({"date": d, "overdue": overdue_cnt})
    ts_df = pd.DataFrame(ts)
    fig2 = px.line(ts_df, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Current Overdue Leads")
    overdue_rows = []
    for _, r in df.iterrows():
        _, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue and r.get("stage") not in ("Won","Lost"):
            overdue_rows.append({"lead_id": r.get("lead_id"), "stage": r.get("stage"), "value": r.get("estimated_value"), "assigned_to": r.get("assigned_to")})
    if overdue_rows:
        st.dataframe(pd.DataFrame(overdue_rows))
    else:
        st.info("No overdue leads currently.")

# CPA & ROI page
def page_cpa_roi():
    st.markdown("<div class='header'>💰 CPA & ROI</div>", unsafe_allow_html=True)
    st.markdown("<em>Total Marketing Spend vs Conversions and ROI calculations.</em>", unsafe_allow_html=True)
    df = leads_df.copy()
    if df.empty:
        st.info("No leads")
        return
    total_spend = float(df["ad_cost"].sum())
    won_df = df[df["stage"] == "Won"]
    conversions = len(won_df)
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(won_df["estimated_value"].sum())
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Marketing Spend</div><div class='kpi-number' style='color:{KPI_COLORS[0]}'>${total_spend:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversions (Won)</div><div class='kpi-number' style='color:{KPI_COLORS[1]}'>{conversions}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-title'>CPA</div><div class='kpi-number' style='color:{KPI_COLORS[3]}'>${cpa:,.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-title'>ROI</div><div class='kpi-number' style='color:{KPI_COLORS[6]}'>${roi:,.2f} ({roi_pct:.1f}%)</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    # chart: spend vs conversions by source
    agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("stage", lambda s: (s=="Won").sum())).reset_index()
    if not agg.empty:
        fig = px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group", title="Total Spend vs Conversions by Source")
        st.plotly_chart(fig, use_container_width=True)

# ML internal page
def page_ml_internal():
    st.markdown("<div class='header'>🧠 Internal ML — Lead Scoring</div>", unsafe_allow_html=True)
    st.markdown("<em>Model runs internally and writes score back to leads. No user tuning exposed.</em>", unsafe_allow_html=True)
    if st.button("Train model (internal)"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_internal_model()
                if acc is None:
                    st.error(f"Training aborted: {msg}")
                else:
                    st.success(f"Model trained (accuracy approx): {acc:.3f}")
            except Exception as e:
                st.error("Training failed: " + str(e))
                st.write(traceback.format_exc())
    model, cols = load_internal_model()
    if model:
        st.success("Model available (internal)")
        if st.button("Score all leads and persist scores"):
            df = leads_to_df()
            scored = score_dataframe(df.copy(), model, cols)
            s = get_session()
            try:
                for _, r in scored.iterrows():
                    lead = s.query(Lead).filter(Lead.lead_id == r["lead_id"]).first()
                    if lead:
                        lead.score = float(r["score"])
                        s.add(lead)
                s.commit()
                st.success("Scores persisted to DB")
            except Exception as e:
                s.rollback()
                st.error("Failed to persist scores: " + str(e))
            finally:
                s.close()
        if st.checkbox("Preview top scored leads"):
            df = leads_to_df()
            scored = score_dataframe(df.copy(), model, cols).sort_values("score", ascending=False).head(20)
            st.dataframe(scored[["lead_id","source","stage","estimated_value","ad_cost","score"]])

# ---------- BEGIN BLOCK G: AI RECOMMENDATIONS PAGE ----------
def page_ai_recommendations():
    st.markdown("<div class='header'>🤖 AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<em>Auto-suggested actions based on pipeline state, SLAs, and technician assignments.</em>", unsafe_allow_html=True)
    df = leads_to_df(start_dt, end_dt)
    if df.empty:
        st.info("No leads to analyze.")
        return

    # Top overdue leads
    overdue_list = []
    for _, r in df.iterrows():
        rem_s, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue and r.get("stage") not in ("Won","Lost"):
            overdue_list.append({"lead_id": r["lead_id"], "stage": r["stage"], "assigned_to": r.get("assigned_to"), "value": r.get("estimated_value")})
    st.subheader("Top Overdue Leads")
    if overdue_list:
        st.table(pd.DataFrame(overdue_list).head(10))
    else:
        st.info("No overdue leads.")

    # Bottleneck stages
    st.subheader("Pipeline Bottlenecks")
    stage_counts = df["stage"].value_counts().reset_index().rename(columns={"index":"stage","stage":"count"})
    st.table(stage_counts.head(10))

    # Technicians workload
    st.subheader("Technician Workload (Assigned Inspections)")
    s = get_session()
    try:
        q = s.query(InspectionAssignment.technician_username, func.count(InspectionAssignment.id).label("assignments")).group_by(InspectionAssignment.technician_username).order_by("assignments desc")
        # SQLAlchemy func import
    finally:
        s.close()

    # Simple action suggestions
    st.subheader("Suggested Actions")
    suggestions = []
    # Suggest reassign if inspection scheduled but no tech assigned
    unassigned = df[df["inspection_scheduled"] == True]
    unassigned = unassigned[unassigned["assigned_to"].isnull() | (unassigned["assigned_to"] == "")]
    for _, r in unassigned.iterrows():
        suggestions.append(f"Lead {r['lead_id']} is scheduled for inspection but has no assigned technician — assign ASAP.")
    # Suggest follow-up on high-value leads not contacted
    high_uncontacted = df[(df["estimated_value"] >= 5000) & (df["contacted"] == False)]
    for _, r in high_uncontacted.iterrows():
        suggestions.append(f"High-value lead {r['lead_id']} (${r['estimated_value']:,.0f}) not contacted — call within SLA.")
    # show up to 10 suggestions
    if suggestions:
        for sgt in suggestions[:10]:
            st.markdown(f"- {sgt}")
    else:
        st.markdown("No immediate suggestions. Pipeline looks healthy.")

# ---------- END BLOCK G ----------


# Settings page: user & role management, weights (priority), audit trail
def page_settings():
    st.markdown("<div class='header'>⚙️ Settings & User Management</div>", unsafe_allow_html=True)
    st.markdown("<em>Add team users, set roles for role-based integration later.</em>", unsafe_allow_html=True)
    st.subheader("Users")
    users_df = get_users_df()
    with st.form("add_user_form"):
        uname = st.text_input("Username (unique)")
        fname = st.text_input("Full name")
        role = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=0)
        if st.form_submit_button("Add / Update User"):
            if not uname:
                st.error("Username required")
            else:
                add_user(uname.strip(), full_name=fname.strip(), role=role)
                st.success("User saved")
                st.rerun()
    if not users_df.empty:
        st.dataframe(users_df)
    st.markdown("---")
    # ---------- BEGIN BLOCK D: SETTINGS UI - TECHNICIANS MANAGEMENT ----------
    st.markdown("---")
    st.subheader("Technicians (Field Users)")
    tech_df = get_technicians_df(active_only=False)
    with st.form("add_technician_form"):
        t_uname = st.text_input("Technician username (unique)")
        t_name = st.text_input("Full name")
        t_phone = st.text_input("Phone")
        t_role_sel = st.selectbox("Specialization", ["Tech", "Estimator", "Adjuster", "Driver"], index=0)
        t_active = st.checkbox("Active", value=True)
        if st.form_submit_button("Add / Update Technician"):
            if not t_uname:
                st.error("Technician username required")
            else:
                try:
                    add_technician(t_uname.strip(), full_name=t_name.strip(), phone=t_phone.strip(), specialization=t_role_sel, active=t_active)
                    st.success("Technician saved")
                    st.experimental_rerun()
                except Exception as e:
                    st.error("Failed to save technician: " + str(e))
    if tech_df is not None and not tech_df.empty:
        st.dataframe(tech_df)
    else:
        st.info("No technicians yet.")
# ---------- END BLOCK D ----------

    st.subheader("Priority weight tuning (internal)")
    wscore = st.slider("Model score weight", 0.0, 1.0, 0.6, 0.05)
    wvalue = st.slider("Estimate value weight", 0.0, 1.0, 0.3, 0.05)
    wsla = st.slider("SLA urgency weight", 0.0, 1.0, 0.1, 0.05)
    baseline = st.number_input("Value baseline (for normalization)", value=5000.0)
    if st.button("Save weights"):
        st.session_state.weights = {"score_w": wscore, "value_w": wvalue, "sla_w": wsla, "value_baseline": baseline}
        st.success("Weights updated (in session)")

    st.markdown("---")
    st.subheader("Audit Trail")
    s = get_session()
    try:
        hist = s.query(LeadHistory).order_by(LeadHistory.timestamp.desc()).limit(200).all()
        if hist:
            hist_df = pd.DataFrame([{"lead_id":h.lead_id,"changed_by":h.changed_by,"field":h.field,"old":h.old_value,"new":h.new_value,"timestamp":h.timestamp} for h in hist])
            st.dataframe(hist_df)
        else:
            st.info("No audit entries yet.")
    finally:
        s.close()

# Exports page
def page_exports():
    st.markdown("<div class='header'>📤 Exports & Imports</div>", unsafe_allow_html=True)
    st.markdown("<em>Export leads, import CSV/XLSX. Imported rows upsert by lead_id.</em>", unsafe_allow_html=True)
    df = leads_to_df(None, None)
    if not df.empty:
        towrite = io.BytesIO()
        df.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        st.markdown(f'<a href="{href}" download="leads_export.xlsx">Download leads_export.xlsx</a>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload leads (CSV/XLSX) for import/upsert", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
            if "lead_id" not in df_in.columns:
                st.error("File must include a lead_id column")
            else:
                count = 0
                for _, r in df_in.iterrows():
                    try:
                        upsert_lead_record({
                            "lead_id": str(r["lead_id"]),
                            "created_at": pd.to_datetime(r.get("created_at")) if r.get("created_at") is not None else datetime.utcnow(),
                            "source": r.get("source"),
                            "contact_name": r.get("contact_name"),
                            "contact_phone": r.get("contact_phone"),
                            "contact_email": r.get("contact_email"),
                            "property_address": r.get("property_address"),
                            "damage_type": r.get("damage_type"),
                            "assigned_to": r.get("assigned_to"),
                            "notes": r.get("notes"),
                            "estimated_value": float(r.get("estimated_value") or 0.0),
                            "ad_cost": float(r.get("ad_cost") or 0.0),
                            "stage": r.get("stage") or "New",
                            "converted": bool(r.get("converted") or False)
                        }, actor="admin")
                        count += 1
                    except Exception:
                        continue
                st.success(f"Imported/Upserted {count} rows.")
        except Exception as e:
            st.error("Failed to import: " + str(e))

# ---------- BEGIN BLOCK F: FLASK API FOR LOCATION PINGS (optional but ready) ----------
try:
    from flask import Flask, request, jsonify
    import threading
    flask_app = Flask("recapture_pro_api")

    @flask_app.route("/api/ping_location", methods=["POST"])
    def api_ping_location():
        try:
            payload = request.get_json(force=True)
            tech = payload.get("tech_username") or payload.get("username")
            lat = payload.get("latitude") or payload.get("lat")
            lon = payload.get("longitude") or payload.get("lon")
            lead_id = payload.get("lead_id")
            accuracy = payload.get("accuracy")
            ts = payload.get("timestamp")
            ts_parsed = None
            if ts:
                try:
                    ts_parsed = datetime.fromisoformat(ts)
                except Exception:
                    ts_parsed = None
            if not tech or lat is None or lon is None:
                return jsonify({"error":"missing fields (tech_username, latitude, longitude)"}), 400
            pid = persist_location_ping(tech_username=str(tech), latitude=float(lat), longitude=float(lon), lead_id=lead_id, accuracy=accuracy, timestamp=ts_parsed)
            return jsonify({"ok": True, "ping_id": pid}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def run_flask():
        try:
            # choose port 5001 to avoid Streamlit port conflicts
            flask_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
        except Exception:
            pass

    # start flask in background daemon thread (only if not already started)
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
except Exception:
    # if Flask isn't available (not installed) the API simply won't start — harmless
    pass
# ---------- END BLOCK F ----------


# ----------------------
# Router (main)
# ----------------------
if page == "Dashboard":
    page_dashboard()
elif page == "Lead Capture":
    page_lead_capture()
elif page == "Pipeline Board":
    page_pipeline_board()
elif page == "Analytics":
    page_analytics()
elif page == "CPA & ROI":
    page_cpa_roi()
elif page == "AI Recommendations":
    page_ai_recommendations()
elif page == "ML (internal)":
    page_ml_internal()
elif page == "Settings":
    page_settings()
elif page == "Exports":
    page_exports()
else:
    st.info("Page not implemented yet.")

# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>ReCapture Pro. SQLite persistence. Integrated Field Tracking (upgrade) enabled.</div>", unsafe_allow_html=True)




