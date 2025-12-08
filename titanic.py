# recapture_pro_final.py
"""
ReCapture Pro — Single-file Streamlit backend (rebuilt)
Features:
- Leads (simple model)
- LeadHistory audit
- Technicians model + management UI
- InspectionAssignment model + assignment UI in pipeline
- LocationPing model + Flask API endpoint (/api/ping_location)
- Safe migrations (won't break existing DB)
- AI Recommendations page (heuristic-based)
- No duplicate Streamlit element IDs; no st.button inside st.form
- Minimal, clean, extendable
"""

import os
import threading
import traceback
from datetime import datetime, timedelta, date
import io
import json

import streamlit as st
import pandas as pd
import numpy as np

# DB / SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError

# Try import Flask for the GPS API; optional
try:
    from flask import Flask, request, jsonify
    flask_available = True
except Exception:
    flask_available = False

# -----------------------
# Config
# -----------------------
APP_TITLE = "ReCapture Pro — Backend Admin"
DB_FILE = os.environ.get("RECAPTURE_DB", "recapture_pro.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

st.set_page_config(page_title="ReCapture Pro", layout="wide")

# -----------------------
# Engine / Base / Session
# -----------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# -----------------------
# MODELS
# -----------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, default=0.0)
    stage = Column(String, default="New")
    sla_hours = Column(Integer, default=48)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    ad_cost = Column(Float, default=0.0)
    score = Column(Float, nullable=True)
    converted = Column(Boolean, default=False)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Technician(Base):
    __tablename__ = "technicians"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    phone = Column(String, nullable=True)
    specialization = Column(String, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class InspectionAssignment(Base):
    __tablename__ = "inspection_assignments"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    technician_username = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="assigned")  # assigned, enroute, onsite, completed, cancelled
    notes = Column(Text, nullable=True)

class LocationPing(Base):
    __tablename__ = "location_pings"
    id = Column(Integer, primary_key=True)
    tech_username = Column(String, nullable=False)
    lead_id = Column(String, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, nullable=True)

# -----------------------
# SAFE MIGRATION / CREATE TABLES
# -----------------------
def safe_create_tables():
    inspector = inspect(engine)
    existing = inspector.get_table_names()
    # Create tables that don't exist
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            table.create(bind=engine)

# Run safe creation at import
try:
    safe_create_tables()
except OperationalError:
    # If DB file locked or otherwise problematic, show warning in UI later
    pass

# -----------------------
# DB Helpers
# -----------------------
def get_session():
    return SessionLocal()

def add_lead(lead_id: str, **kwargs):
    s = get_session()
    try:
        existing = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if existing:
            return existing.lead_id
        l = Lead(lead_id=lead_id, **kwargs)
        s.add(l)
        s.commit()
        return l.lead_id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def upsert_lead_record(payload: dict, actor="admin"):
    s = get_session()
    try:
        lid = payload.get("lead_id")
        if not lid:
            raise ValueError("lead_id required")
        lead = s.query(Lead).filter(Lead.lead_id == lid).first()
        if not lead:
            # create minimal lead
            lead = Lead(lead_id=lid, created_at=datetime.utcnow())
        # record old values and update
        for k, v in payload.items():
            if k == "lead_id":
                continue
            old = getattr(lead, k, None)
            setattr(lead, k, v)
            if old != v:
                s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field=k, old_value=str(old), new_value=str(v)))
        s.add(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def leads_to_df(start_date=None, end_date=None):
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
                "contact_name": r.contact_name,
                "contact_phone": r.contact_phone,
                "contact_email": r.contact_email,
                "property_address": r.property_address,
                "damage_type": r.damage_type,
                "assigned_to": r.assigned_to,
                "notes": r.notes,
                "estimated_value": float(r.estimated_value or 0.0),
                "stage": r.stage or "New",
                "sla_hours": int(r.sla_hours or 48),
                "sla_entered_at": r.sla_entered_at or r.created_at,
                "contacted": bool(r.contacted),
                "inspection_scheduled": bool(r.inspection_scheduled),
                "inspection_scheduled_at": r.inspection_scheduled_at,
                "inspection_completed": bool(r.inspection_completed),
                "estimate_submitted": bool(False),
                "awarded_date": None,
                "lost_date": None,
                "qualified": bool(False),
                "ad_cost": float(r.ad_cost or 0.0),
                "converted": bool(r.converted),
                "score": float(r.score) if r.score is not None else None
            })
        df = pd.DataFrame(data)
        if df.empty:
            cols = ["id","lead_id","created_at","source","contact_name","contact_phone","contact_email",
                    "property_address","damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                    "contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted",
                    "awarded_date","lost_date","qualified","ad_cost","converted","score"]
            return pd.DataFrame(columns=cols)
        # optional filters
        if start_date:
            df = df[df["created_at"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["created_at"] <= pd.to_datetime(end_date)]
        return df.reset_index(drop=True)
    finally:
        s.close()

# Technician helpers
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
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if lead:
            # don't override assigned_to if set manually; set for convenience
            lead.assigned_to = technician_username
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by="system", field="assigned_to", old_value=str(getattr(lead, "assigned_to", "")), new_value=str(technician_username)))
            s.add(lead)
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

# -----------------------
# Utilities
# -----------------------
def calculate_remaining_sla(start_dt, sla_hours):
    if start_dt is None:
        return None, False
    try:
        if isinstance(start_dt, str):
            start_dt = pd.to_datetime(start_dt)
        end_dt = pd.to_datetime(start_dt) + pd.Timedelta(hours=int(sla_hours))
        remaining = (end_dt - pd.Timestamp.utcnow()).total_seconds()
        return remaining, remaining < 0
    except Exception:
        return None, False

# -----------------------
# Minimal sample data (if none)
# -----------------------
def bootstrap_sample_data():
    s = get_session()
    try:
        if s.query(Lead).count() == 0:
            now = datetime.utcnow()
            sample = [
                {"lead_id":"L1001","source":"Website","contact_name":"Alice","estimated_value":1500,"stage":"New"},
                {"lead_id":"L1002","source":"Google","contact_name":"Bob","estimated_value":8500,"stage":"Inspection Scheduled"},
                {"lead_id":"L1003","source":"Referral","contact_name":"Charlie","estimated_value":3000,"stage":"New"},
            ]
            for it in sample:
                l = Lead(lead_id=it["lead_id"], source=it["source"], contact_name=it["contact_name"], estimated_value=it["estimated_value"], stage=it["stage"], created_at=now)
                s.add(l)
            s.commit()
    except Exception:
        s.rollback()
    finally:
        s.close()

bootstrap_sample_data()

# -----------------------
# Flask API for location pings (optional)
# -----------------------
if flask_available:
    try:
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
                flask_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
            except Exception:
                pass

        t = threading.Thread(target=run_flask, daemon=True)
        t.start()
    except Exception:
        pass

# -----------------------
# STREAMLIT PAGES
# -----------------------
st.markdown(f"<h1 style='font-family:Inter'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.sidebar.title("Navigate")
PAGES = ["Dashboard", "Lead Capture", "Pipeline Board", "AI Recommendations", "Settings", "Exports"]
page = st.sidebar.radio("Go to", PAGES, index=0)

# Alerts helper uses a global leads_df var during dashboard rendering; we'll define per page
DEFAULT_SLA_HOURS = 48
PIPELINE_STAGES = ["New","Contacted","Inspection Scheduled","Inspection","Estimate Sent","Won","Lost"]

# ---- Dashboard Page ----
def alerts_ui(leads_df):
    overdue = []
    for _, r in leads_df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue_flag and r.get("stage") not in ("Won","Lost"):
            overdue.append(r)
    if overdue:
        col1, col2 = st.columns([1, 10])
        with col1:
            # unique key prevents duplicates across pages
            if st.button(f"🔔 {len(overdue)}", key="alerts_button"):
                st.session_state.show_alerts = not st.session_state.get("show_alerts", False)
        with col2:
            st.markdown("")
        if st.session_state.get("show_alerts", False):
            with st.expander("SLA Alerts (click to close)", expanded=True):
                for r in overdue:
                    st.markdown(f"**{r['lead_id']}** — Stage: {r['stage']} — <span style='color:#22c55e;'>${r['estimated_value']:,.0f}</span> — <span style='color:#dc2626;'>OVERDUE</span>", unsafe_allow_html=True)
                if st.button("Close Alerts", key="alerts_close_button"):
                    st.session_state.show_alerts = False

def page_dashboard():
    st.subheader("Dashboard")
    leads_df = leads_to_df()
    alerts_ui(leads_df)
    st.write("Quick pipeline snapshot")
    snap = leads_df["stage"].value_counts().rename_axis("stage").reset_index(name="count") if not leads_df.empty else pd.DataFrame(columns=["stage","count"])
    st.table(snap)

# ---- Lead Capture ----
def page_lead_capture():
    st.markdown("<div class='header'>📇 Lead Capture</div>", unsafe_allow_html=True)
    st.info("Quickly create a lead to test the pipeline.")
    with st.form("lead_capture_form"):
        lid = st.text_input("Lead ID", value=f"L{int(datetime.utcnow().timestamp())%100000}")
        name = st.text_input("Contact name")
        est = st.number_input("Estimated value", value=0.0, min_value=0.0, step=100.0)
        source = st.text_input("Source", value="Website")
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            try:
                add_lead(lid, source=source, contact_name=name, estimated_value=est)
                st.success(f"Lead {lid} created")
            except Exception as e:
                st.error("Failed to create lead: " + str(e))

# ---- Pipeline Board ----
def page_pipeline_board():
    st.markdown("<div class='header'>📋 Pipeline Board</div>", unsafe_allow_html=True)
    df = leads_to_df()
    if df.empty:
        st.info("No leads in pipeline.")
        return

    # iterate leads
    for idx, lead in df.iterrows():
        with st.container():
            st.markdown(f"### Lead: {lead['lead_id']} — {lead['stage']}")

            # update form (per-lead)
            with st.form(f"update_{lead['lead_id']}", clear_on_submit=False):
                new_stage = st.selectbox(
                    "Status",
                    PIPELINE_STAGES,
                    index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0,
                    key=f"stage_select_{lead['lead_id']}"
                )

                new_assigned = st.text_input(
                    "Assigned to (username)",
                    value=lead.get("assigned_to") or "",
                    key=f"assigned_to_{lead['lead_id']}"
                )

                new_est = st.number_input(
                    "Estimated value (USD)",
                    value=float(lead.get("estimated_value") or 0.0),
                    min_value=0.0,
                    step=100.0,
                    key=f"est_val_{lead['lead_id']}"
                )

                new_cost = st.number_input(
                    "Cost to acquire lead (USD)",
                    value=float(lead.get("ad_cost") or 0.0),
                    min_value=0.0,
                    step=1.0,
                    key=f"ad_cost_{lead['lead_id']}"
                )

                new_notes = st.text_area(
                    "Notes",
                    value=lead.get("notes") or "",
                    key=f"notes_{lead['lead_id']}"
                )

                # form submit
                submitted = st.form_submit_button("Save changes", key=f"save_changes_{lead['lead_id']}")
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

            # ---------- BEGIN BLOCK E: TECHNICIAN ASSIGNMENT ----------
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
                        st.success(f"Assigned {selected_tech} to lead {lead['lead_id']}")
                        upsert_lead_record({
                            "lead_id": lead["lead_id"],
                            "inspection_scheduled": True,
                            "stage": "Inspection Scheduled"
                        }, actor="admin")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error("Failed to assign: " + str(e))
            # ---------- END BLOCK E ----------

# ---- AI Recommendations ----
def page_ai_recommendations():
    st.markdown("<div class='header'>🤖 AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<em>Simple heuristic recommendations based on pipeline state.</em>", unsafe_allow_html=True)
    df = leads_to_df()
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

    # Pipeline bottlenecks
    st.subheader("Pipeline Bottlenecks")
    stage_counts = df["stage"].value_counts().reset_index().rename(columns={"index":"stage","stage":"count"}) if not df.empty else pd.DataFrame()
    st.table(stage_counts.head(10))

    # Technician workload
    st.subheader("Technician Workload (Assigned Inspections)")
    s = get_session()
    try:
        rows = s.query(InspectionAssignment.technician_username).all()
        # simple count
        assign_counts = pd.DataFrame([{"technician": r[0]} for r in rows])
        if not assign_counts.empty:
            wc = assign_counts["technician"].value_counts().reset_index().rename(columns={"index":"technician", "technician":"assignments"})
            st.table(wc)
        else:
            st.info("No assignments yet.")
    finally:
        s.close()

    # Suggestions
    st.subheader("Suggested Actions")
    suggestions = []
    unassigned = df[df["inspection_scheduled"] == True]
    unassigned = unassigned[unassigned["assigned_to"].isnull() | (unassigned["assigned_to"] == "")]
    for _, r in unassigned.iterrows():
        suggestions.append(f"Lead {r['lead_id']} is scheduled for inspection but has no assigned technician — assign ASAP.")
    high_uncontacted = df[(df["estimated_value"] >= 5000) & (df["contacted"] == False)]
    for _, r in high_uncontacted.iterrows():
        suggestions.append(f"High-value lead {r['lead_id']} (${r['estimated_value']:,.0f}) not contacted — call within SLA.")
    if suggestions:
        for sgt in suggestions[:10]:
            st.markdown(f"- {sgt}")
    else:
        st.markdown("No immediate suggestions. Pipeline looks healthy.")

# ---- Settings ----
def page_settings():
    st.markdown("<div class='header'>⚙️ Settings & User Management</div>", unsafe_allow_html=True)
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

# ---- Exports (basic) ----
def page_exports():
    st.markdown("<div class='header'>📤 Exports</div>", unsafe_allow_html=True)
    df = leads_to_df()
    if df.empty:
        st.info("No data to export.")
        return
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="leads.csv", mime="text/csv")

# -----------------------
# Router
# -----------------------
if page == "Dashboard":
    page_dashboard()
elif page == "Lead Capture":
    page_lead_capture()
elif page == "Pipeline Board":
    page_pipeline_board()
elif page == "AI Recommendations":
    page_ai_recommendations()
elif page == "Settings":
    page_settings()
elif page == "Exports":
    page_exports()
else:
    st.info("Page not implemented yet.")

# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>ReCapture Pro — SQLite persistence. Integrated Field Tracking (upgrade) enabled.</div>", unsafe_allow_html=True)
