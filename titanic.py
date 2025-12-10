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

# ---------- BEGIN BLOCK TASKS: DB MODEL ----------
class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    task_type = Column(String, nullable=False)          # e.g., "Follow-up", "Assign Technician", "SLA Alert"
    lead_id = Column(String, nullable=True)             # optional link to lead.lead_id
    description = Column(Text, nullable=False)
    priority = Column(String, default="Medium")        # Low/Medium/High/Urgent
    status = Column(String, default="open")            # open, assigned, in_progress, completed, cancelled
    assigned_to = Column(String, nullable=True)        # username of assignee (technician or user)
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
# ---------- END BLOCK TASKS ----------

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

# Force creation of technician-related tables on first launch
Base.metadata.create_all(bind=engine, tables=[
    Technician.__table__,
    InspectionAssignment.__table__,
    LocationPing.__table__
])
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
        
        # ALWAYS return these columns, even if empty
        df = pd.DataFrame([
            {
                "username": r.username,
                "full_name": r.full_name or "",
                "phone": r.phone or "",
                "specialization": r.specialization or "Tech",
                "active": r.active
            } for r in rows
        ])
        
        # If completely empty, return empty DF with correct columns
        if df.empty:
            df = pd.DataFrame(columns=["username", "full_name", "phone", "specialization", "active"])
        
        return df
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

def get_technician_usernames(active_only=True):
    df = get_technicians_df(active_only=active_only)
    return sorted(df["username"].dropna().unique().tolist())
# ======================================================================
# 🔵 TECHNICIAN LOCATION & MAP SYSTEM (COMPLETE + CORRECT)
# ======================================================================

import pydeck as pdk
from datetime import datetime

# ======================================================================
# 1) GET LATEST LOCATION PING PER TECHNICIAN
# ======================================================================
def get_latest_tech_locations():
    """
    Returns: list of {tech_username, latitude, longitude, timestamp, accuracy, lead_id}
    """
    s = get_session()
    try:
        rows = (
            s.query(LocationPing)
            .order_by(LocationPing.tech_username, LocationPing.timestamp.desc())
            .all()
        )

        latest = {}

        for r in rows:
            uname = getattr(r, "tech_username", None)
            if not uname:
                continue

            # first seen per tech = latest ping because of ordering
            if uname not in latest:
                latest[uname] = {
                    "tech_username": uname,
                    "latitude": float(r.latitude),
                    "longitude": float(r.longitude),
                    "timestamp": r.timestamp,
                    "accuracy": float(r.accuracy) if r.accuracy else None,
                    "lead_id": getattr(r, "lead_id", None),
                }

        return list(latest.values())

    finally:
        s.close()


# ======================================================================
# 2) RENDER INTERACTIVE MAP (TECH LOCATIONS + ARC LINES)
# ======================================================================
def render_tech_map(zoom=11, show_lines=False):
    tech_locs = get_latest_tech_locations()

    if not tech_locs:
        st.info("No technician location data yet.")
        return

    df = pd.DataFrame(tech_locs)

    # Compute center (average lat/lon)
    center = (df["latitude"].mean(), df["longitude"].mean())

    # Base layer: technician markers
    scatter = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude", "latitude"],
        get_radius=80,
        get_color=[255, 0, 0],
        pickable=True,
    )

    tooltip = {
        "html": (
            "<b>Technician:</b> {tech_username}<br/>"
            "<b>Time:</b> {timestamp}<br/>"
            "<b>Lead:</b> {lead_id}<br/>"
            "<b>Accuracy:</b> {accuracy}"
        ),
        "style": {"color": "white"},
    }

    layers = [scatter]

    # OPTIONAL LINES (tech → lead)
    if show_lines:
        s = get_session()
        try:
            arcs = []
            for r in tech_locs:
                lid = r.get("lead_id")
                if not lid:
                    continue

                # latest ping for that lead
                latest_lead_ping = (
                    s.query(LocationPing)
                    .filter(LocationPing.lead_id == lid)
                    .order_by(LocationPing.timestamp.desc())
                    .first()
                )

                if latest_lead_ping:
                    arcs.append({
                        "source_lon": float(r["longitude"]),
                        "source_lat": float(r["latitude"]),
                        "target_lon": float(latest_lead_ping.longitude),
                        "target_lat": float(latest_lead_ping.latitude),
                    })

            if arcs:
                arc_df = pd.DataFrame(arcs)
                arc_layer = pdk.Layer(
                    "ArcLayer",
                    arc_df,
                    get_source_position=["source_lon", "source_lat"],
                    get_target_position=["target_lon", "target_lat"],
                    get_width=4,
                    get_tilt=15,
                    pickable=False,
                    get_source_color=[0, 128, 200],
                    get_target_color=[200, 0, 80],
                )
                layers.append(arc_layer)

        finally:
            s.close()

    # VIEW STATE
    view_state = pdk.ViewState(
        latitude=float(center[0]),
        longitude=float(center[1]),
        zoom=zoom,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=layers,
        tooltip=tooltip,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
    )

    st.pydeck_chart(deck, use_container_width=True)


# ======================================================================
# 3) TECHNICIAN MAP PAGE (UI PAGE)
# ======================================================================
def page_technician_map():
    st.markdown("<div class='header'>🗺️ Technician Map Tracking</div>", unsafe_allow_html=True)
    st.markdown("<em>Live technician GPS locations.</em>", unsafe_allow_html=True)

    # auto refresh option
    auto = st.checkbox("Enable auto-refresh", value=False)
    if auto:
        interval = st.number_input("Refresh interval (sec)", min_value=5, max_value=300, value=20)
        st.markdown(
            f'<meta http-equiv="refresh" content="{interval}">', 
            unsafe_allow_html=True
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        show_lines = st.checkbox("Show tech → lead lines", value=False)
    with col2:
        zoom = st.slider("Zoom", min_value=3, max_value=18, value=11)

    # Render map
   # controls on page_technician_map
filter_choice = st.selectbox("Filter technician", options=["All"] + get_technician_usernames(active_only=True), index=0)
show_paths = st.checkbox("Show path history", value=False)
show_heatmap = st.checkbox("Show heatmap (last 24h)", value=False)
show_assigned = st.checkbox("Show assigned lead markers", value=False)

render_tech_map(
    zoom=st.slider("Zoom", 3, 18, 11),
    show_lines=st.checkbox("Show lines to assigned lead", value=False),
    filter_tech=None if filter_choice == "All" else filter_choice,
    show_paths=show_paths,
    show_heatmap=show_heatmap,
    show_assigned_leads=show_assigned
)


# ---------- END BLOCK MAP HELPERS ----------
# =========================
# TECH MAP: advanced features
# =========================
import pydeck as pdk
from datetime import timedelta

# --- Helper: tech status map (derive a status per tech) ---
def get_tech_status_map():
    """
    Returns dict: { username: status }
    Status logic (first applicable):
      - 'onsite' if any assignment for tech has status 'onsite'
      - 'enroute' if any assignment has 'enroute'
      - 'assigned' if any assignment is 'assigned'
      - 'idle' otherwise
    """
    s = get_session()
    try:
        # load active technicians
        techs = s.query(Technician).all()
        status_map = {t.username: "idle" for t in techs if getattr(t, "username", None)}
        # check assignments
        rows = s.query(InspectionAssignment).filter(InspectionAssignment.technician_username != None).all()
        for r in rows:
            uname = r.technician_username
            stt = (r.status or "").lower()
            if not uname:
                continue
            # priority: onsite > enroute > assigned > any other
            prev = status_map.get(uname, "idle")
            if "onsite" in stt:
                status_map[uname] = "onsite"
            elif prev != "onsite" and "enroute" in stt:
                status_map[uname] = "enroute"
            elif prev not in ("onsite","enroute") and "assigned" in stt:
                status_map[uname] = "assigned"
            else:
                status_map.setdefault(uname, prev)
        return status_map
    finally:
        s.close()

# --- Helper: last N pings per technician (for path history) ---
def get_tech_paths(limit_per_tech: int = 50, since_minutes: int = 240):
    """
    Returns dict { tech_username: [ {lat,lon,timestamp} ... ] }
    Limits results to last N pings per tech or pings since `since_minutes`.
    """
    s = get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        q = s.query(LocationPing).filter(LocationPing.timestamp >= cutoff).order_by(LocationPing.tech_username, LocationPing.timestamp.asc())
        rows = q.all()
        paths = {}
        for r in rows:
            uname = getattr(r, "tech_username", None)
            if not uname:
                continue
            paths.setdefault(uname, []).append({
                "lat": float(r.latitude),
                "lon": float(r.longitude),
                "ts": r.timestamp
            })
        # enforce per-tech limit (keep recent)
        for k, v in list(paths.items()):
            if len(v) > limit_per_tech:
                paths[k] = v[-limit_per_tech:]
        return paths
    finally:
        s.close()

# --- Updated render_tech_map with new options ---
def render_tech_map(zoom=11,
                    show_lines=False,
                    filter_tech: str | None = None,
                    show_paths: bool = False,
                    path_points_limit: int = 100,
                    show_heatmap: bool = False,
                    heatmap_radius_pixels: int = 40,
                    show_assigned_leads: bool = False):
    """
    Enhanced map renderer:
      - color coding by tech status
      - optional lines to assigned lead pings
      - optional path history lines
      - optional heatmap
      - filter by single technician
    """
    tech_locs = get_latest_tech_locations()
    if not tech_locs:
        st.info("No technician location pings available.")
        return

    # Apply technician filter if requested
    if filter_tech:
        tech_locs = [t for t in tech_locs if t["tech_username"] == filter_tech]
        if not tech_locs:
            st.info(f"No location pings for {filter_tech}.")
            return

    df = pd.DataFrame(tech_locs)
    # derive center (if single tech, center on them)
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

    # get tech statuses
    status_map = get_tech_status_map()

    # color mapping by status
    COLOR_BY_STATUS = {
        "idle": [120, 120, 120],       # gray
        "assigned": [0, 122, 255],     # blue
        "enroute": [255, 165, 0],      # orange
        "onsite": [34, 197, 94],       # green
        # fallback
        "unknown": [200, 30, 60]
    }

    # attach color column based on status_map
    def _color_for_row(r):
        uname = r.get("tech_username")
        stt = status_map.get(uname, "unknown")
        return COLOR_BY_STATUS.get(stt, COLOR_BY_STATUS["unknown"])

    df["fill_color"] = df.apply(lambda r: _color_for_row(r), axis=1)
    df["radius"] = df.apply(lambda r: 60 if status_map.get(r["tech_username"], "") in ("enroute","onsite") else 35, axis=1)
    df["label"] = df.apply(lambda r: f"{r['tech_username']} ({status_map.get(r['tech_username'],'unknown')})", axis=1)

    layers = []

    # Scatter layer for technicians (color-coded)
    scatter = pdk.Layer(
        "ScatterplotLayer",
        df,
        pickable=True,
        get_position=["longitude", "latitude"],
        get_fill_color="fill_color",
        get_radius="radius",
        radius_scale=1,
        radius_min_pixels=6,
        radius_max_pixels=120,
        auto_highlight=True,
    )
    layers.append(scatter)

    # Optional: show path history as LineLayer per tech
    if show_paths:
        paths = get_tech_paths(limit_per_tech=path_points_limit, since_minutes=720)
        arc_data = []
        for tech, pts in paths.items():
            if filter_tech and tech != filter_tech:
                continue
            if len(pts) < 2:
                continue
            # build sequential line segments for this tech
            for i in range(len(pts)-1):
                a = pts[i]
                b = pts[i+1]
                arc_data.append({
                    "source_lon": float(a["lon"]),
                    "source_lat": float(a["lat"]),
                    "target_lon": float(b["lon"]),
                    "target_lat": float(b["lat"]),
                    "tech": tech,
                })
        if arc_data:
            arc_df = pd.DataFrame(arc_data)
            arc_layer = pdk.Layer(
                "LineLayer",
                arc_df,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_width=3,
                get_color=[60, 120, 180],
                pickable=False,
            )
            layers.append(arc_layer)

    # Optional: heatmap (all pings)
    if show_heatmap:
        # collect recent pings (last 24h)
        s = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            pings = s.query(LocationPing).filter(LocationPing.timestamp >= cutoff).all()
            if pings:
                heat = pd.DataFrame([{"lat": float(p.latitude), "lon": float(p.longitude)} for p in pings])
                heat["position"] = heat.apply(lambda r: [r["lon"], r["lat"]], axis=1)
                heat_layer = pdk.Layer(
                    "HeatmapLayer",
                    heat,
                    get_position="position",
                    aggregation=pdk.types.String("MEAN"),
                    radiusPixels=heatmap_radius_pixels,
                )
                layers.append(heat_layer)
        finally:
            s.close()

    # Optional: draw lines from tech to lead latest ping
    if show_lines or show_assigned_leads:
        s = get_session()
        try:
            lead_markers = []
            arc_data = []
            for r in tech_locs:
                lid = r.get("lead_id")
                if not lid:
                    continue
                lead_ping = s.query(LocationPing).filter(LocationPing.lead_id == lid).order_by(LocationPing.timestamp.desc()).first()
                if not lead_ping:
                    continue
                # lead marker
                lead_markers.append({
                    "lead_id": lid,
                    "latitude": float(lead_ping.latitude),
                    "longitude": float(lead_ping.longitude)
                })
                # arc from tech to lead
                if show_lines:
                    arc_data.append({
                        "source_lon": float(r["longitude"]),
                        "source_lat": float(r["latitude"]),
                        "target_lon": float(lead_ping.longitude),
                        "target_lat": float(lead_ping.latitude),
                        "tech": r["tech_username"],
                        "lead_id": lid
                    })
            if lead_markers and show_assigned_leads:
                lead_df = pd.DataFrame(lead_markers)
                lead_layer = pdk.Layer(
                    "IconLayer",
                    lead_df,
                    get_icon="get_icon",
                    get_position=["longitude", "latitude"],
                    size_scale=15,
                    pickable=True,
                    icon_mapping={
                        "marker": {"x": 0, "y": 0, "width": 128, "height": 128, "anchorY": 128, "anchorX": 64}
                    }
                )
                # build a column with get_icon info
                lead_df["get_icon"] = {
                    "url": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png",
                    "width": 128, "height": 128, "anchorY": 128, "anchorX": 64
                }
                layers.append(lead_layer)
            if arc_data:
                arc_df = pd.DataFrame(arc_data)
                arc_layer = pdk.Layer(
                    "ArcLayer",
                    arc_df,
                    get_source_position=["source_lon", "source_lat"],
                    get_target_position=["target_lon", "target_lat"],
                    get_width=3,
                    get_tilt=15,
                    get_source_color=[0, 128, 200],
                    get_target_color=[200, 0, 80],
                )
                layers.append(arc_layer)
        finally:
            s.close()

    # view state and deck
    view_state = pdk.ViewState(latitude=float(center_lat), longitude=float(center_lon), zoom=zoom, pitch=0)
    tooltip = {"html": "<b>{label}</b><br/>Tech: {tech_username}<br/>Last ping: {timestamp}", "style": {"color": "white"}}

    deck = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=layers, tooltip=tooltip)

    st.pydeck_chart(deck, use_container_width=True)

# ---------- BEGIN BLOCK C.1: TECH / ASSIGNMENT HELPERS ----------
def get_assignments_for_technician(technician_username: str, only_open: bool = True):
    """
    Returns list of assignments with related lead fields for a technician.
    Each item: { id, lead_id, assigned_at, status, notes, lead: {contact_name, property_address, stage, estimated_value} }
    """
    s = get_session()
    try:
        q = s.query(InspectionAssignment).filter(InspectionAssignment.technician_username == technician_username)
        if only_open:
            q = q.filter(InspectionAssignment.status != "completed")
        rows = q.order_by(InspectionAssignment.assigned_at.desc()).all()
        results = []
        for r in rows:
            lead = s.query(Lead).filter(Lead.lead_id == r.lead_id).first() if getattr(r, "lead_id", None) else None
            results.append({
                "id": r.id,
                "lead_id": r.lead_id,
                "assigned_at": r.assigned_at.isoformat() if r.assigned_at else None,
                "status": r.status,
                "notes": r.notes,
                "lead": {
                    "contact_name": getattr(lead, "contact_name", None) if lead else None,
                    "property_address": getattr(lead, "property_address", None) if lead else None,
                    "stage": getattr(lead, "stage", None) if lead else None,
                    "estimated_value": float(getattr(lead, "estimated_value", 0) or 0) if lead else 0
                }
            })
        return results
    finally:
        s.close()

def update_assignment_status(assignment_id: int, status: str = None, note: str = None, mark_lead_inspection_completed: bool = False):
    """
    Update assignment status and optional note. Optionally also update the associated lead (inspection_completed).
    Returns True/False.
    """
    s = get_session()
    try:
        ia = s.query(InspectionAssignment).filter(InspectionAssignment.id == int(assignment_id)).first()
        if not ia:
            return False
        if status:
            ia.status = status
        if note is not None:
            # append note to existing notes
            existing = ia.notes or ""
            ts = datetime.utcnow().isoformat()
            ia.notes = existing + ("\n" if existing else "") + f"{ts} - {note}"
        s.add(ia)
        # update lead if requested
        if mark_lead_inspection_completed and ia.lead_id:
            lead = s.query(Lead).filter(Lead.lead_id == ia.lead_id).first()
            if lead:
                lead.inspection_completed = True
                s.add(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
# ---------- END BLOCK C.1 ----------

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

# ---------- BEGIN BLOCK TASK HELPERS ----------
def create_task(task_type: str, description: str, lead_id: str = None, priority: str = "Medium",
                assigned_to: str = None, created_by: str = "system"):
    s = get_session()
    try:
        t = Task(task_type=task_type, description=description, lead_id=lead_id,
                 priority=priority, assigned_to=assigned_to, created_by=created_by)
        s.add(t)
        s.commit()
        return t.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def get_tasks_df(status_filter: list = None):
    s = get_session()
    try:
        q = s.query(Task).order_by(Task.created_at.desc())
        if status_filter:
            q = q.filter(Task.status.in_(status_filter))
        rows = q.all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "task_type": r.task_type,
                "lead_id": r.lead_id,
                "description": r.description,
                "priority": r.priority,
                "status": r.status,
                "assigned_to": r.assigned_to,
                "created_by": r.created_by,
                "created_at": r.created_at,
                "updated_at": r.updated_at
            })
        return pd.DataFrame(data)
    finally:
        s.close()

def update_task_status(task_id: int, status: str, assigned_to: str = None):
    s = get_session()
    try:
        t = s.query(Task).filter(Task.id == int(task_id)).first()
        if not t:
            return False
        t.status = status
        if assigned_to is not None:
            t.assigned_to = assigned_to
        s.add(t); s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
# ---------- END BLOCK TASK HELPERS ----------
# ---------- BEGIN BLOCK TASK CREATOR ----------
def create_task_from_suggestion(suggestion_text: str, lead_id: str = None, priority: str = "High", assign_to: str = None, created_by: str = "ai"):
    # Creates a Task and returns ID
    try:
        tid = create_task(task_type="AI Suggestion", description=suggestion_text, lead_id=lead_id, priority=priority, assigned_to=assign_to, created_by=created_by)
        return tid
    except Exception as e:
        raise
# ---------- END BLOCK TASK CREATOR ----------

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
# ----------------------------------------
# STEP 4: STREAMLIT API ENDPOINT FOR GPS
# ----------------------------------------

def streamlit_api_handler():
    params = st.query_params

    if "api" not in params:
        return  # not an API request

    if params["api"] == "ping_location":
        try:
            tech = params.get("tech")
            lat = params.get("lat")
            lon = params.get("lon")
            lead_id = params.get("lead_id", None)
            acc = params.get("acc", None)

            if not tech or not lat or not lon:
                st.json({"error": "Missing fields"})
                st.stop()

            pid = persist_location_ping(
                tech_username=str(tech),
                latitude=float(lat),
                longitude=float(lon),
                lead_id=lead_id,
                accuracy=float(acc) if acc else None
            )

            st.json({"ok": True, "ping_id": pid})
            st.stop()

        except Exception as e:
            st.json({"error": str(e)})
            st.stop()

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

streamlit_api_handler()
# ----------------------
with st.sidebar:
    st.header("ReCapture Pro (Admin)")
    st.markdown("You are using the backend admin interface. User accounts and roles are managed in Settings.")

    page = st.radio(
        "Menu",
        [
        "Dashboard",
        "Lead Capture",
        "Pipeline Board",
        "Analytics",
        "CPA & ROI",
        "AI Recommendations",
        "ML (internal)",
        "Technician Mobile",
        "Technician Map Tracking",   # << add this
        "Tasks",
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


def page_dashboard():
    st.markdown("<div class='header'>TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR</div>", unsafe_allow_html=True)
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)
    alerts_ui()

    df = leads_df.copy()

    # KPI calculations
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

    # KPI Cards (7 animated)
    r1 = st.columns(4)
    r2 = st.columns(3)
    cols = r1 + r2

    for col, (title, value, color, note) in zip(cols, KPI_ITEMS):
        pct = min(100, max(10, (hash(title) % 80) + 20))
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
        fig = px.pie(pie_df, names="status", values="count", hole=0.45)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

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
st.markdown("---")
st.markdown("### 📋 All Leads (expand a card to edit / change status)")  
st.markdown("---")

st.markdown("### Live Technician Map")
st.markdown("<em>Latest location for each technician (click Auto-refresh to update automatically).</em>", unsafe_allow_html=True)

# small inline control
c1, c2 = st.columns([3,1])
with c1:
    inline_show_lines = st.checkbox("Show lines to assigned lead", key="dash_map_lines", value=False)
with c2:
    inline_refresh = st.checkbox("Auto-refresh inline map (meta refresh)", key="dash_map_auto", value=False)

if inline_refresh:
    # default 20 sec
    st.markdown('<meta http-equiv="refresh" content="20">', unsafe_allow_html=True)

# ---------------------------
# Technician Filters
# ---------------------------
filter_choice = st.selectbox(
    "Filter technician",
    options=["All"] + get_technician_usernames(active_only=True),
    index=0
)

show_paths = st.checkbox("Show path history", value=False)
show_heatmap = st.checkbox("Show heatmap (last 24h)", value=False)
show_assigned = st.checkbox("Show assigned lead markers", value=False)

render_tech_map(
    zoom=st.slider("Zoom", 3, 18, 11),
    show_lines=st.checkbox("Show lines to assigned lead", value=False),
    filter_tech=None if filter_choice == "All" else filter_choice,
    show_paths=show_paths,
    show_heatmap=show_heatmap,
    show_assigned_leads=show_assigned
)

# ---------------------------
# Leads List Section
# ---------------------------
st.markdown("### 📋 All Leads (expand a card to edit / change status)")

# Filter bar
q1, q2, q3 = st.columns([3,2,3])
with q1:
    search_q = st.text_input("Search (lead_id, contact name, address, notes)")
with q2:
    filter_src = st.selectbox(
        "Source filter",
        options=["All"] + sorted(df["source"].dropna().unique().tolist()) if not df.empty else ["All"]
    )
with q3:
    filter_stage = st.selectbox("Stage filter", options=["All"] + PIPELINE_STAGES)

df_view = df.copy()

# --- Search filter ---
if search_q:
    sq = search_q.lower()
    df_view = df_view[df_view.apply(
        lambda r: (
            sq in str(r.get("lead_id", "")).lower() or
            sq in str(r.get("contact_name", "")).lower() or
            sq in str(r.get("property_address", "")).lower() or
            sq in str(r.get("notes", "")).lower()
        ),
        axis=1
    )]

# --- Source filter ---
if filter_src != "All":
    df_view = df_view[df_view["source"] == filter_src]

# --- Stage filter ---
if filter_stage != "All":
    df_view = df_view[df_view["stage"] == filter_stage]

# --- No leads case ---
if df_view.empty:
    st.info("No leads to show.")
    st.stop()

# -----------------------------
# Leads List
# -----------------------------
    for _, lead in df_view.sort_values("created_at", ascending=False).head(200).iterrows():

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

            # -------------------------
            # UPDATE FORM
            # -------------------------
            with st.form(f"update_{lead['lead_id']}", clear_on_submit=False):

                new_stage = st.selectbox("Status", PIPELINE_STAGES,
                                         index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0)

                new_assigned = st.text_input("Assigned to (username)", value=lead.get("assigned_to") or "")
                new_est = st.number_input("Estimated value (USD)", value=float(lead.get("estimated_value") or 0.0), min_value=0.0, step=100.0)
                new_cost = st.number_input("Cost to acquire lead (USD)", value=float(lead.get("ad_cost") or 0.0), min_value=0.0, step=1.0)
                new_notes = st.text_area("Notes", value=lead.get("notes") or "")

                submitted = st.form_submit_button("Save changes")
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
                        st.rerun()

                    except Exception as e:
                        st.error("Failed to update lead: " + str(e))
                        st.write(traceback.format_exc())

            # -------------------------
            # TECHNICIAN ASSIGNMENT (BLOCK E)
            # -------------------------
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

            if st.button(f"Assign Technician to {lead['lead_id']}", key=f"assign_btn_{lead['lead_id']}"):
                if not selected_tech:
                    st.error("Select a technician")
                else:
                    try:
                        create_inspection_assignment(
                            lead_id=lead["lead_id"],
                            technician_username=selected_tech,
                            notes=assign_notes
                        )

                        upsert_lead_record({
                            "lead_id": lead["lead_id"],
                            "inspection_scheduled": True,
                            "stage": "Inspection Scheduled"
                        }, actor="admin")

                        st.success(f"Assigned {selected_tech} to lead {lead['lead_id']}")
                        st.rerun()

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
    """
    Advanced AI Recommendations page (Option B).
    - Safe: validates data & prevents duplicate-column crashes
    - Uses internal model if available: load_internal_model(), score_dataframe()
    - Uses DB for technician workload: InspectionAssignment
    """
    import plotly.express as px
    from sqlalchemy import func
    from datetime import datetime

    st.markdown("<div class='header'>🤖 AI Recommendations — Advanced</div>", unsafe_allow_html=True)
    st.markdown("<em>Predictive insights, pipeline bottlenecks, technician workload, and recommended actions.</em>", unsafe_allow_html=True)

    # 0) Load and validate data
    try:
        df = leads_to_df()
    except Exception as e:
        st.error("Failed to load leads: " + str(e))
        return

    if df is None or df.empty:
        st.info("No lead data available. Add leads first or check DB.")
        return

    # Ensure required columns exist (provide helpful error if missing)
    required = ["lead_id", "stage", "estimated_value", "created_at"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column for AI analysis: {col}")
            return

    # Normalize types lightly
    df = df.copy()
    try:
        df["estimated_value"] = pd.to_numeric(df["estimated_value"].fillna(0))
    except Exception:
        pass

    # ---------- 1) Top Overdue Leads ----------
    st.subheader("Top Overdue Leads")
    overdue_rows = []
    for _, r in df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue_rows.append({
                "lead_id": r["lead_id"],
                "stage": r.get("stage"),
                "assigned_to": r.get("assigned_to") or "",
                "estimated_value": float(r.get("estimated_value") or 0.0),
                "overdue_seconds": rem_s or 0
            })
    if overdue_rows:
        over_df = pd.DataFrame(overdue_rows).sort_values(["estimated_value","overdue_seconds"], ascending=[False,True])
        # keep safe columns and unique names
        over_df = over_df.rename(columns={"lead_id":"Lead ID","stage":"Stage","assigned_to":"Assigned To","estimated_value":"Est. Value","overdue_seconds":"Overdue Seconds"})
        st.table(over_df[["Lead ID","Stage","Assigned To","Est. Value"]].head(20))
        csv = over_df.to_csv(index=False)
        st.download_button("Download overdue leads (CSV)", data=csv, file_name="overdue_leads.csv", mime="text/csv")
    else:
        st.info("No overdue leads detected.")

    st.markdown("---")

    # ---------- 2) Pipeline Bottlenecks ----------
    st.subheader("Pipeline Bottlenecks")
    try:
        stages = PIPELINE_STAGES
    except Exception:
        stages = df["stage"].unique().tolist()

    # Use value_counts -> reset_index -> unique names (prevents pyarrow error)
    stage_counts = df["stage"].value_counts().reindex(stages, fill_value=0)
    stage_df = stage_counts.reset_index()
    stage_df.columns = ["Stage","Count"]
    st.table(stage_df.head(20))

    # horizontal bar chart
    try:
        fig = px.bar(stage_df, x="Count", y="Stage", orientation="h", height=320, title="Leads by Stage")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write(stage_df)

    st.markdown("---")

    # ---------- 3) Technician Workload (from InspectionAssignment) ----------
    st.subheader("Technician Workload (Assigned Inspections)")
    try:
        s = get_session()
        try:
            # count assignments per tech
            rows = s.query(InspectionAssignment.technician_username, func.count(InspectionAssignment.id)).group_by(InspectionAssignment.technician_username).all()
            if rows:
                tw = pd.DataFrame(rows, columns=["Technician","Assigned Inspections"]).sort_values("Assigned Inspections", ascending=False)
                st.table(tw)
            else:
                st.info("No inspection assignments found.")
        finally:
            s.close()
    except Exception as e:
        st.error("Failed to load assignments: " + str(e))

    st.markdown("---")

    # ---------- 4) Internal ML model scoring (optional) ----------
    st.subheader("Model Scoring & Predictions")
    model = None
    model_cols = None
    try:
        # try to load internal model if you have one
        model, model_cols = load_internal_model()
    except Exception:
        model = None
        model_cols = None

    if model:
        st.success("Internal model loaded — scoring leads.")
        try:
            df_score_input = df.copy()
            # Score only required cols if available; score_dataframe should handle missing cols defensively
            scored = score_dataframe(df_score_input, model, model_cols)
            # ensure unique column names
            if "score" in scored.columns:
                scored = scored.rename(columns={c:c if scored.columns.tolist().count(c)==1 else f"{c}_{i}" for i,c in enumerate(scored.columns)})
                top_wins = scored.sort_values("score", ascending=False).head(10)
                top_risk = scored.sort_values("score", ascending=True).head(10)
                st.markdown("**Top predicted wins**")
                st.table(top_wins[["lead_id","stage","estimated_value","score"]].head(10))
                st.markdown("**Top at-risk leads**")
                st.table(top_risk[["lead_id","stage","estimated_value","score"]].head(10))
                # export
                st.download_button("Download scored leads (CSV)", data=scored.to_csv(index=False), file_name="scored_leads.csv", mime="text/csv")
            else:
                st.info("Model did not return 'score' column.")
        except Exception as e:
            st.error("Model scoring failed: " + str(e))
    else:
        st.info("No internal model available. You can train one from ML internal page.")

    st.markdown("---")

    suggestions = []
    suggestion_rows = []  # keep structured suggestions: dicts

    # generate suggestions as before (you already had this)
    if not df.empty:
        # overdue leads
        for _, r in df.iterrows():
            rem_s, overdue_flag = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            if overdue_flag and r.get("stage") not in ("Won","Lost"):
                suggestions.append(f"Lead {r['lead_id']} is OVERDUE in stage {r.get('stage')}. Contact owner or reassign.")
                suggestion_rows.append({"lead_id": r["lead_id"], "text": f"Lead {r['lead_id']} is OVERDUE in stage {r.get('stage')}. Contact owner or reassign.", "priority": "High"})
        # scheduled but unassigned
        scheduled = df[(df.get("inspection_scheduled") == True)]
        scheduled_unassigned = scheduled[(scheduled["assigned_to"].isnull()) | (scheduled["assigned_to"] == "")]
        for _, r in scheduled_unassigned.iterrows():
            txt = f"Lead {r['lead_id']} scheduled for inspection but unassigned — assign a technician."
            suggestions.append(txt)
            suggestion_rows.append({"lead_id": r["lead_id"], "text": txt, "priority": "High"})
        # high-value not contacted
        high_uncontacted = df[(df["estimated_value"] >= 5000) & (df.get("contacted") != True)]
        for _, r in high_uncontacted.iterrows():
            txt = f"High-value lead {r['lead_id']} (${int(r['estimated_value']):,}) not contacted — prioritize within SLA."
            suggestions.append(txt)
            suggestion_rows.append({"lead_id": r["lead_id"], "text": txt, "priority": "High"})
        # bottleneck suggestion
        bottleneck = stage_df.sort_values("Count", ascending=False).iloc[0] if not stage_df.empty else None
        if bottleneck is not None and bottleneck["Count"] > max(5, len(df) * 0.2):
            txt = f"Stage '{bottleneck['Stage']}' is a bottleneck with {int(bottleneck['Count'])} leads — review process at this stage."
            suggestions.append(txt)
            suggestion_rows.append({"lead_id": None, "text": txt, "priority": "Medium"})

        

    if not suggestion_rows:
        st.markdown("No immediate suggestions — pipeline looks healthy.")
    else:
        # render each suggestion with a Create Task button and optional assign-to
        techs_df = get_technicians_df(active_only=True)
        tech_options = [""] + (techs_df["username"].tolist() if not techs_df.empty else [])
        for i, srow in enumerate(suggestion_rows):
            st.markdown(f"- {srow['text']}")
            cols = st.columns([2,1,1])
            with cols[0]:
                # small input to override priority
                p = st.selectbox(f"Priority (suggestion {i})", options=["Low","Medium","High","Urgent"], index=["Low","Medium","High","Urgent"].index(srow.get("priority","Medium")), key=f"sugg_priority_{i}")
            with cols[1]:
                assignee = st.selectbox(f"Assign to (optional) (suggestion {i})", options=tech_options, index=0, key=f"sugg_assign_{i}")
            with cols[2]:
                if st.button("Create Task", key=f"sugg_create_{i}"):
                    try:
                        tid = create_task_from_suggestion(suggestion_text=srow["text"], lead_id=srow.get("lead_id"), priority=p, assign_to=assignee, created_by="ai")
                        st.success(f"Task {tid} created")
                    except Exception as e:
                        st.error("Failed to create task: " + str(e))



    # ---------- 6) Quick exports for ops ----------
    st.subheader("Exports")
    try:
        if 'over_df' in locals() and not over_df.empty:
            st.download_button("Download overdue leads (CSV)", data=over_df.to_csv(index=False), file_name="overdue_leads.csv", mime="text/csv")
        # full leads export
        st.download_button("Download full leads CSV", data=df.to_csv(index=False), file_name="all_leads.csv", mime="text/csv")
    except Exception:
        pass

    st.markdown("---")
    # End of page


    # End of page
def page_tasks():
    st.markdown("<div class='header'>🗂️ Tasks</div>", unsafe_allow_html=True)
    st.markdown("<em>Operational task queue (auto-created by AI or manual).</em>", unsafe_allow_html=True)

    # Filters
    f1, f2 = st.columns([2,1])
    with f1:
        status_filter = st.selectbox("Status", options=["All","open","assigned","in_progress","completed","cancelled"], index=0)
    with f2:
        priority_filter = st.selectbox("Priority", options=["All","Urgent","High","Medium","Low"], index=0)

    statuses = None if status_filter == "All" else [status_filter]
    tasks_df = get_tasks_df(status_filter=statuses)
    if tasks_df is None or tasks_df.empty:
        st.info("No tasks found.")
        return

    # apply priority filter
    if priority_filter != "All":
        tasks_df = tasks_df[tasks_df["priority"] == priority_filter]

    # show table
    st.dataframe(tasks_df[["id","task_type","lead_id","priority","status","assigned_to","created_at"]].sort_values("created_at", ascending=False))

    # Action: select a task to update
    st.markdown("### Update task")
    sel = st.number_input("Task ID", min_value=0, value=0, step=1, key="task_sel_id")
    if sel:
        row = tasks_df[tasks_df["id"] == int(sel)]
        if row.empty:
            st.warning("Task not found.")
        else:
            row = row.iloc[0]
            st.write("**Description**")
            st.write(row["description"])
            new_status = st.selectbox("New status", options=["open","assigned","in_progress","completed","cancelled"], index=["open","assigned","in_progress","completed","cancelled"].index(row["status"]))
            new_assignee = st.text_input("Assign to (username)", value=row["assigned_to"] or "")
            if st.button("Save task update", key=f"save_task_{sel}"):
                try:
                    update_task_status(task_id=sel, status=new_status, assigned_to=new_assignee or None)
                    st.success("Task updated")
                    st.rerun()
                except Exception as e:
                    st.error("Failed to update task: " + str(e))

def page_technician_mobile():
    st.markdown("<div class='header'>📱 Technician Mobile Preview</div>", unsafe_allow_html=True)
    st.markdown("<em>Preview of the mobile shell for technicians (for admins only).</em>", unsafe_allow_html=True)

    uname = st.text_input("Technician username (preview)", value="")
    if not uname:
        st.info("Enter a technician username to preview assigned inspections.")
        return

    try:
        rows = get_assignments_for_technician(uname, only_open=True)
    except Exception as e:
        st.error("Failed to load assignments: " + str(e))
        return

    if not rows:
        st.info("No assignments for " + uname)
        return

    for it in rows:
        lead = it.get("lead") or {}

        with st.container():
            st.markdown(f"**Lead:** {it.get('lead_id') or ''} • {lead.get('contact_name') or ''}")
            st.markdown(f"{lead.get('property_address') or ''}")
            st.markdown(f"Stage: **{lead.get('stage') or ''}** • Value: ${int(lead.get('estimated_value') or 0):,}")
            st.markdown(f"Status: **{it.get('status')}**")

            new_status = st.selectbox(
                "Set status",
                ["","enroute","onsite","completed"],
                key=f"mstat_{it['id']}"
            )

            note = st.text_area("Note (optional)", key=f"mnote_{it['id']}")

            if st.button("Save", key=f"msave_{it['id']}"):
                try:
                    ok = update_assignment_status(
                        assignment_id=it['id'],
                        status=new_status or None,
                        note=note or None,
                        mark_lead_inspection_completed=(new_status == "completed")
                    )
                    if ok:
                        st.success("Updated")
                        st.rerun()
                    else:
                        st.error("Update failed")
                except Exception as e:
                    st.error("Failed to update: " + str(e))

        # ---- GPS BUTTON (correct indentation) ----
        if st.button("Send GPS Ping", key=f"gps_{it['id']}"):
            st.markdown(f"""
            <script>
            navigator.geolocation.getCurrentPosition(function(pos) {{
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                const url = window.location.origin +
                    "?api=ping_location&tech={uname}&lat=" + lat + "&lon=" + lon;

                fetch(url).then(r => r.json()).then(data => {{
                    alert("GPS Updated: " + JSON.stringify(data));
                }});
            }});
            </script>
            """, unsafe_allow_html=True)



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
                    st.rerun()
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
                # -------------------------------
    # Technician mobile endpoints
    # -------------------------------
    @flask_app.route("/tech/assignments", methods=["GET"])
    def api_tech_assignments():
        """
        Query params:
        - username: technician username (required)
        - open_only: "1" or "0" (optional, default 1)
        """
        try:
            username = request.args.get("username")
            open_only = request.args.get("open_only", "1") != "0"
            if not username:
                return jsonify({"error":"username required"}), 400
            rows = get_assignments_for_technician(username, only_open=open_only)
            return jsonify({"ok": True, "assignments": rows}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/tech/update_assignment", methods=["POST"])
    def api_update_assignment():
        """
        POST JSON:
        {
          "assignment_id": 123,
          "status": "enroute" | "onsite" | "completed",
          "note": "optional",
          "mark_lead_completed": true|false
        }
        """
        try:
            payload = request.get_json(force=True)
            aid = payload.get("assignment_id")
            if not aid:
                return jsonify({"error":"assignment_id required"}), 400
            status = payload.get("status")
            note = payload.get("note")
            mark_lead = bool(payload.get("mark_lead_completed", False))
            ok = update_assignment_status(assignment_id=aid, status=status, note=note, mark_lead_inspection_completed=mark_lead)
            return jsonify({"ok": bool(ok)}), 200 if ok else 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Minimal mobile HTML shell (calls /tech/assignments and /tech/update_assignment)
    
    @flask_app.route("/tech/mobile", methods=["GET"])
    def tech_mobile_shell():
        html = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Tech Mobile - ReCapture Pro</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;padding:12px;background:#f7f7f9;}
    .card{background:#fff;padding:12px;margin-bottom:12px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
    .btn{display:inline-block;padding:8px 10px;border-radius:6px;border:none;margin:4px 2px;font-size:14px}
    .btn-primary{background:#0ea5e9;color:#fff}
    .btn-ghost{background:#eef2ff;color:#111}
    input,select,textarea{width:100%;padding:8px;margin:6px 0;border-radius:6px;border:1px solid #ddd}
    header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
  </style>
</head>
<body>
<header>
  <h3>ReCapture Pro — Tech</h3>
  <div>
    <input id="username" placeholder="Technician username" />
    <button onclick="loadAssignments()" class="btn btn-primary">Load</button>
  </div>
</header>
<div id="assignments"></div>

<script>
async function loadAssignments(){
  const u = document.getElementById('username').value;
  if(!u){ alert('enter username'); return; }
  document.getElementById('assignments').innerHTML = '<div class="card">Loading...</div>';
  try{
    const res = await fetch('/tech/assignments?username=' + encodeURIComponent(u));
    const j = await res.json();
    if(!j.ok){ document.getElementById('assignments').innerHTML = '<div class="card">Error loading</div>'; return; }
    renderAssignments(j.assignments, u);
  }catch(e){
    document.getElementById('assignments').innerHTML = '<div class="card">Network error</div>';
  }
}

function renderAssignments(items, username){
  if(!items || items.length===0){
    document.getElementById('assignments').innerHTML = '<div class="card">No assignments</div>';
    return;
  }
  const container = document.getElementById('assignments');
  container.innerHTML = '';
  items.forEach(it => {
    const lead = it.lead || {};
    const html = document.createElement('div');
    html.className = 'card';
    html.innerHTML = `
      <div><strong>Lead: ${it.lead_id || '—'}</strong> • <small>${lead.contact_name || ''}</small></div>
      <div style="margin-top:6px">${lead.property_address || ''}</div>
      <div style="margin-top:6px">Stage: <strong>${lead.stage || ''}</strong> — Value: $${(lead.estimated_value||0).toLocaleString()}</div>
      <div style="margin-top:8px">Status: <em id="status_${it.id}">${it.status}</em></div>
      <div style="margin-top:8px">
        <select id="select_${it.id}">
          <option value="">-- set status --</option>
          <option value="enroute">Enroute</option>
          <option value="onsite">Onsite</option>
          <option value="completed">Completed</option>
        </select>
      </div>
      <div style="margin-top:8px">
        <textarea id="note_${it.id}" placeholder="Add a note (optional)"></textarea>
      </div>
      <div style="margin-top:8px">
        <button class="btn btn-primary" onclick="updateAssignment(${it.id}, '${username}')">Save</button>
      </div>
    `;
    container.appendChild(html);
  });
}

async function updateAssignment(aid, username){
  const sel = document.getElementById('select_' + aid);
  const note = document.getElementById('note_' + aid).value;
  const status = sel ? sel.value : '';
  if(!status){ alert('select a status'); return; }
  try{
    const res = await fetch('/tech/update_assignment', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ assignment_id: aid, status: status, note: note, mark_lead_completed: status === 'completed' })
    });
    const j = await res.json();
    if(j.ok){
      document.getElementById('status_' + aid).innerText = status;
      alert('Updated');
    } else {
      alert('Failed: ' + (j.error || 'unknown'));
    }
  }catch(e){
    alert('Network error');
  }
}
</script>
</body>
</html>
"""
        return html, 200, {"Content-Type": "text/html"}


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
elif page == "Tasks":
    page_tasks()       # ✅ newly added
elif page == "Technician Mobile":
    page_technician_mobile()
elif page == "Technician Map Tracking":
    page_technician_map()
elif page == "Settings":
    page_settings()
elif page == "Exports":
    page_exports()
else:
    st.info("Page not implemented yet.")


# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>ReCapture Pro. SQLite persistence. Integrated Field Tracking (upgrade) enabled.</div>", unsafe_allow_html=True)






