# project_x_stable.py
# Stable single-file Streamlit app for Restoration Pipeline (robust, defensive)
# Features: login, pipeline KPIs, top-5 priority, SLA alerts, leads CRUD, CPA/ROI, settings, users, export
# Run: streamlit run project_x_stable.py

import os
from datetime import datetime, timedelta, date, time
import traceback

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# plotly optional
try:
    import plotly.express as px
except Exception:
    px = None

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base

# ----- Config -----
DB_FILE = os.path.join(os.getcwd(), "project_x_stable.db")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----- Database setup -----
# expire_on_commit=False prevents detached instances
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


# ----- Models -----
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String, default="Website")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, nullable=True)
    status = Column(String, default="New")
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    awarded_date = Column(DateTime, nullable=True)
    awarded_comment = Column(Text, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    lost_comment = Column(Text, nullable=True)
    qualified = Column(Boolean, default=False)
    cost_to_acquire = Column(Float, default=0.0)
    owner = Column(String, default=None)
    priority_score = Column(Float, default=0.0)


class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)


class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer)
    updated_by = Column(String)
    update_time = Column(DateTime, default=datetime.utcnow)
    old_status = Column(String)
    new_status = Column(String)
    note = Column(Text)


Base.metadata.create_all(engine)


# ----- Utilities -----
def get_session():
    return SessionLocal()


def row_to_dict(lead_row):
    # convert SQLAlchemy object to plain dict safely
    return {
        "id": lead_row.id,
        "source": lead_row.source,
        "source_details": lead_row.source_details,
        "contact_name": lead_row.contact_name,
        "contact_phone": lead_row.contact_phone,
        "contact_email": lead_row.contact_email,
        "property_address": lead_row.property_address,
        "damage_type": lead_row.damage_type,
        "assigned_to": lead_row.assigned_to,
        "notes": lead_row.notes,
        "estimated_value": float(lead_row.estimated_value or 0.0),
        "status": lead_row.status,
        "created_at": lead_row.created_at,
        "sla_hours": int(lead_row.sla_hours or 24),
        "sla_entered_at": lead_row.sla_entered_at or lead_row.created_at,
        "contacted": bool(lead_row.contacted),
        "inspection_scheduled": bool(lead_row.inspection_scheduled),
        "inspection_scheduled_at": lead_row.inspection_scheduled_at,
        "inspection_completed": bool(lead_row.inspection_completed),
        "estimate_submitted": bool(lead_row.estimate_submitted),
        "awarded_date": lead_row.awarded_date,
        "awarded_invoice": lead_row.awarded_invoice,
        "lost_date": lead_row.lost_date,
        "qualified": bool(lead_row.qualified),
        "cost_to_acquire": float(lead_row.cost_to_acquire or 0.0),
        "owner": lead_row.owner,
        "priority_score": float(lead_row.priority_score or 0.0)
    }


def fetch_leads(start_date=None, end_date=None):
    s = get_session()
    try:
        q = s.query(Lead).order_by(Lead.created_at.desc())
        rows = q.all()
        dicts = [row_to_dict(r) for r in rows]
        df = pd.DataFrame(dicts)
        if df.empty:
            return df
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        return df.reset_index(drop=True)
    finally:
        s.close()


@st.cache_data(ttl=30)
def get_leads_df_cached(start_date=None, end_date=None):
    # caching wrapper
    return fetch_leads(start_date, end_date)


def add_lead(payload: dict):
    s = get_session()
    try:
        lead = Lead(
            source=payload.get("source"),
            source_details=payload.get("source_details"),
            contact_name=payload.get("contact_name"),
            contact_phone=payload.get("contact_phone"),
            contact_email=payload.get("contact_email"),
            property_address=payload.get("property_address"),
            damage_type=payload.get("damage_type"),
            assigned_to=payload.get("assigned_to"),
            notes=payload.get("notes"),
            estimated_value=payload.get("estimated_value") or 0.0,
            sla_hours=int(payload.get("sla_hours") or 24),
            sla_entered_at=payload.get("sla_entered_at") or datetime.utcnow(),
            qualified=bool(payload.get("qualified")),
            cost_to_acquire=float(payload.get("cost_to_acquire") or 0.0),
            owner=payload.get("owner") or None
        )
        s.add(lead)
        s.commit()
        s.refresh(lead)
        return lead.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def update_lead(lead_id: int, updates: dict, actor: str = None, note: str = None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.id == lead_id).first()
        if not lead:
            return False
        old_status = lead.status
        for k, v in updates.items():
            if hasattr(lead, k):
                setattr(lead, k, v)
        # write history
        h = LeadHistory(
            lead_id=lead_id,
            updated_by=actor or "",
            update_time=datetime.utcnow(),
            old_status=old_status,
            new_status=lead.status,
            note=note or ""
        )
        s.add(h)
        s.add(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def compute_remaining_sla_seconds(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or 24))
        remaining = deadline - datetime.utcnow()
        return max(remaining.total_seconds(), 0), (remaining.total_seconds() <= 0)
    except Exception:
        return float("inf"), False


def compute_priority(lead_row: dict, weights=None):
    if weights is None:
        weights = {"value_weight": 0.6, "sla_weight": 0.4, "value_baseline": 5000.0}
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
        vscore = min(1.0, val / max(1.0, weights.get("value_baseline", 5000.0)))
    except Exception:
        vscore = 0.0
    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    try:
        if isinstance(sla_entered, str):
            sla_entered = datetime.fromisoformat(sla_entered)
        deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
        time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    except Exception:
        time_left_h = 9999.0
    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)
    score = vscore * weights["value_weight"] + sla_score * weights["sla_weight"]
    return max(0.0, min(1.0, score))


# ----- UI / App -----
st.set_page_config(page_title="Project X — Stable", layout="wide")

# CSS
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
body, .stApp { font-family: 'Comfortaa', sans-serif; background: #ffffff; color: #0b1220; }
.header { display:flex; justify-content:space-between; align-items:center; padding:12px 6px; border-bottom:1px solid #eee; }
.brand { font-weight:800; font-size:20px; }
.small { color:#6b7280; font-size:13px; }
.kpi { background:#000; color:#fff; padding:12px; border-radius:10px; margin:6px; }
.kpi .title { color:white; font-weight:700; font-size:13px; }
.kpi .value { font-weight:900; font-size:20px; margin-top:6px; }
.progress { height:8px; border-radius:6px; margin-top:8px; }
.lead-card { border-radius:10px; padding:10px; border:1px solid #eef2f7; margin-bottom:8px; }
.badge { background:#ef4444; color:#fff; padding:6px 10px; border-radius:8px; font-weight:700; cursor:pointer; }
.small-muted { color:#6b7280; font-size:12px; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='header'><div><div class='brand'>Project X — Pipeline (Stable)</div><div class='small'>Sales & lead pipeline — SLA & CPA</div></div></div>", unsafe_allow_html=True)
with col2:
    # lightweight alert badge
    if "alerts_open" not in st.session_state:
        st.session_state.alerts_open = False

    # lazy compute overdue count
    try:
        df_tmp = get_leads_df_cached(None, None)
        overdue_count = int(df_tmp[(df_tmp["status"].str.lower() == "overdue")].shape[0]) if not df_tmp.empty else 0
    except Exception:
        overdue_count = 0

    if overdue_count:
        if st.button(f"🔔 {overdue_count}"):
            st.session_state.alerts_open = not st.session_state.alerts_open
    else:
        st.markdown("<div class='small-muted'>No SLA alerts</div>", unsafe_allow_html=True)

# Sidebar: login & navigation & quick controls
with st.sidebar:
    st.markdown("## Account")
    user_name = st.text_input("Username", value=st.session_state.get("user", ""))
    if st.button("Login"):
        if user_name.strip() == "":
            st.warning("Please enter a username")
        else:
            # create user if not exists
            s = get_session()
            try:
                u = s.query(User).filter(User.username == user_name).first()
                if not u:
                    u = User(username=user_name, full_name=user_name, role="Admin" if user_name.lower() == "admin" else "Viewer")
                    s.add(u); s.commit()
                st.session_state.user = u.username
                st.session_state.role = u.role
                st.success(f"Signed in as {st.session_state.user} ({st.session_state.role})")
            finally:
                s.close()

    if st.session_state.get("user"):
        st.markdown(f"Signed in as **{st.session_state.user}**")
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()  # safe here, after clearing state we'll rerun to login prompt

    st.markdown("---")
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Pipeline Board", "Leads / Capture", "Analytics & SLA", "CPA & ROI", "Settings", "Users", "Exports"], index=0)
    st.session_state.page = page

    st.markdown("---")
    st.markdown("## Date range")
    dr_option = st.selectbox("Quick range", ["Today", "Last 7 days", "Last 30 days", "Custom"], index=0)
    if dr_option == "Today":
        start_sel = date.today(); end_sel = date.today()
    elif dr_option == "Last 7 days":
        start_sel = date.today() - timedelta(days=6); end_sel = date.today()
    elif dr_option == "Last 30 days":
        start_sel = date.today() - timedelta(days=29); end_sel = date.today()
    else:
        start_sel = st.date_input("Start date", value=date.today() - timedelta(days=6))
        end_sel = st.date_input("End date", value=date.today())

    st.session_state.start_date = start_sel
    st.session_state.end_date = end_sel

# ensure logged in
if "user" not in st.session_state or not st.session_state.get("user"):
    st.warning("Please login in the left sidebar to continue.")
    st.stop()

# pull leads for selected date range
start_d = st.session_state.get("start_date", None)
end_d = st.session_state.get("end_date", None)
leads_df = get_leads_df_cached(start_d, end_d)

# show alerts dropdown if toggled
if st.session_state.get("alerts_open", False):
    st.markdown("### SLA / Overdue Leads")
    st.markdown("*Close this panel by clicking the bell again.*")
    if leads_df.empty:
        st.info("No leads found")
    else:
        overdue_df = leads_df[leads_df["status"].str.lower() == "overdue"]
        if overdue_df.empty:
            st.info("No overdue leads")
        else:
            for _, r in overdue_df.sort_values("created_at").iterrows():
                remain_sec, is_overdue = compute_remaining_sla_seconds(r["sla_entered_at"], r["sla_hours"])
                hours_left = int(remain_sec // 3600) if remain_sec not in (float("inf"), None) else "—"
                st.markdown(f"<div class='lead-card'><b>#{int(r['id'])} — {r['contact_name'] or 'No name'}</b>  ·  <span style='color:#22c55e;'>${r['estimated_value']:,.0f}</span><br><span class='small-muted'>Status: {r['status']} • Owner: {r.get('owner') or '—'}</span><div style='float:right'><span style='color:#dc2626;font-weight:700;'>{'OVERDUE' if is_overdue else str(hours_left)+'h left'}</span></div></div>", unsafe_allow_html=True)

# ------------------ PAGES ------------------

# 1) Pipeline Board
def page_pipeline_board(df: pd.DataFrame):
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("*High-level pipeline performance at a glance. Use filters to drill into details.*")

    if df.empty:
        st.info("No leads in selected range. Create a lead in Leads / Capture.")
        return

    total_leads = len(df)
    qualified = int(df[df["qualified"] == True].shape[0])
    sla_contacts = int(df[df["contacted"] == True].shape[0])
    awarded_count = int(df[df["status"].str.lower() == "awarded"].shape[0])
    lost_count = int(df[df["status"].str.lower() == "lost"].shape[0])
    inspection_scheduled_count = int(df[df["inspection_scheduled"] == True].shape[0])
    estimate_sent_count = int(df[df["estimate_submitted"] == True].shape[0])
    pipeline_job_value = float(df["estimated_value"].sum() or 0.0)
    active_leads = total_leads - (awarded_count + lost_count)

    sla_success_pct = (sla_contacts / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified / total_leads * 100) if total_leads else 0.0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_pct = (inspection_scheduled_count / qualified * 100) if qualified else 0.0

    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "#2563eb"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "#0ea5a4"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "#a855f7"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444"),
        ("Estimates Sent", f"{estimate_sent_count}", "#6d28d9"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "#22c55e"),
    ]

    # Render 2 rows: 4 + 3
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7 = st.columns(3)
    cols = [c1, c2, c3, c4, c5, c6, c7]
    for col, item in zip(cols, KPI_ITEMS):
        title, value, color = item
        pct = 50  # default progress visual; you can compute more meaningful percent per KPI
        col.markdown(f"""
            <div class="kpi">
              <div class="title">{title}</div>
              <div class="value" style="color:{color};">{value}</div>
              <div class="progress" style="background:{color}; width:{pct}%;"></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Lead Pipeline Stages")
    st.markdown("*Distribution of leads across stages.*")

    # Show donut (in analytics only per your earlier ask) — here show simplified counts bar to avoid missing libs
    stage_counts = df["status"].value_counts().to_dict()
    st.write(stage_counts)

    st.markdown("---")
    st.subheader("TOP 5 PRIORITY LEADS")
    st.markdown("*Highest urgency leads by priority score (computed internally).*")

    # compute priority for each lead dict and show top5
    pr_list = []
    for _, r in df.iterrows():
        score = compute_priority(r.to_dict())
        pr_list.append({**r.to_dict(), "priority_score": score})
    pr_df = pd.DataFrame(pr_list).sort_values("priority_score", ascending=False).head(5)

    if pr_df.empty:
        st.info("No priority leads")
    else:
        for _, r in pr_df.iterrows():
            rem_sec, overdue = compute_remaining_sla_seconds(r["sla_entered_at"], r["sla_hours"])
            hours_left = int(rem_sec // 3600) if rem_sec not in (float("inf"), None) else "—"
            time_html = f"<span style='color:#dc2626;font-weight:700;'>{'OVERDUE' if overdue else str(hours_left)+'h left'}</span>"
            money_html = f"<span style='color:#22c55e;font-weight:800;'>${r['estimated_value']:,.0f}</span>"
            st.markdown(f"<div class='lead-card'><b>#{int(r['id'])} — {r['contact_name'] or 'No name'}</b>  <div style='float:right'>{money_html}</div><br><div class='small-muted'>Status: {r['status']}</div><div style='margin-top:6px'>{time_html}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("All Leads (expand a card to edit / change status)")
    st.markdown("*Click a lead to open the editor.*")

    # Quick filter and search interface
    col_a, col_b, col_c = st.columns([2, 2, 3])
    with col_a:
        status_filter = st.selectbox("Filter status", options=["All"] + sorted(df["status"].dropna().unique().tolist()))
    with col_b:
        owner_filter = st.text_input("Filter owner (quick)")
    with col_c:
        q = st.text_input("Search (name, phone, email, address)")

    df_view = df.copy()
    if status_filter and status_filter != "All":
        df_view = df_view[df_view["status"] == status_filter]
    if owner_filter:
        df_view = df_view[df_view["owner"].str.contains(owner_filter, case=False, na=False)]
    if q:
        ql = q.lower()
        df_view = df_view[df_view.apply(lambda r: ql in str(r["contact_name"]).lower() or ql in str(r["contact_phone"]).lower() or ql in str(r["contact_email"]).lower() or ql in str(r["property_address"]).lower(), axis=1)]

    # show first N rows for speed
    rows_to_show = min(100, df_view.shape[0])
    st.dataframe(df_view.sort_values("created_at", ascending=False).head(rows_to_show))

    # Expandable editors
    for _, row in df_view.head(rows_to_show).iterrows():
        with st.expander(f"Lead #{int(row['id'])} — {row['contact_name'] or 'No name'}"):
            colL, colR = st.columns([3, 1])
            with colL:
                st.write("**Details**")
                st.write(f"Source: {row['source']}  •  Damage: {row.get('damage_type')}")
                st.write(f"Address: {row.get('property_address')}")
                st.write(f"Notes: {row.get('notes')}")
            with colR:
                entered = row.get("sla_entered_at") or row.get("created_at")
                rem_sec, overdue = compute_remaining_sla_seconds(entered, row.get("sla_hours") or 24)
                if overdue:
                    st.markdown("<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                else:
                    hours = int(rem_sec // 3600)
                    mins = int((rem_sec % 3600) // 60)
                    st.markdown(f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours}h {mins}m left</div>", unsafe_allow_html=True)

            # editor
            new_status = st.selectbox("Status", options=["New", "Contacted", "Inspection Scheduled", "Inspection Completed", "Estimate Submitted", "Awarded", "Lost"], index=0, key=f"status_edit_{row['id']}")
            new_owner = st.text_input("Assigned to", value=row.get("owner") or "", key=f"owner_edit_{row['id']}")
            new_est_val = st.number_input("Job Value Estimate (USD)", value=float(row.get("estimated_value") or 0.0), min_value=0.0, step=100.0, key=f"est_edit_{row['id']}")
            upload = st.file_uploader("Upload awarded invoice (optional)", type=["pdf", "jpg", "jpeg", "png"], key=f"inv_{row['id']}")

            if st.button("Save", key=f"save_{row['id']}"):
                updates = {
                    "status": new_status,
                    "owner": new_owner,
                    "estimated_value": float(new_est_val or 0.0),
                    "sla_entered_at": row.get("sla_entered_at") or row.get("created_at")
                }
                if upload:
                    # save upload
                    path = os.path.join(UPLOAD_DIR, f"lead_{int(row['id'])}_{upload.name}")
                    with open(path, "wb") as f:
                        f.write(upload.getbuffer())
                    updates["awarded_invoice"] = path
                try:
                    update_lead(int(row["id"]), updates, actor=st.session_state.get("user"), note="Updated via UI")
                    st.success("Lead updated")
                    # clear cache
                    get_leads_df_cached.clear()
                except Exception as e:
                    st.error(f"Failed to update lead: {e}")
                    st.write(traceback.format_exc())

# 2) Leads / Capture
def page_leads_capture():
    st.header("📇 Lead Capture")
    with st.form("lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox("Lead Source", ["Google Ads", "Organic Search", "Referral", "Phone", "Insurance", "Facebook", "Instagram", "Other"])
            source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google...")
            contact_name = st.text_input("Contact name", placeholder="John Doe")
            contact_phone = st.text_input("Contact phone", placeholder="+1-555-0123")
            contact_email = st.text_input("Contact email", placeholder="name@example.com")
        with c2:
            property_address = st.text_input("Property address", placeholder="123 Main St, City, State")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to", placeholder="Estimator name")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes", placeholder="Additional context...")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        cost_to_acquire = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            payload = {
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
                "sla_hours": int(sla_hours),
                "sla_entered_at": datetime.utcnow(),
                "qualified": True if qualified_choice == "Yes" else False,
                "cost_to_acquire": float(cost_to_acquire),
                "owner": assigned_to or None
            }
            try:
                add_lead(payload)
                st.success("Lead created")
                get_leads_df_cached.clear()
            except Exception as e:
                st.error(f"Failed to create lead: {e}")
                st.write(traceback.format_exc())

    st.markdown("---")
    st.subheader("Recent Leads")
    df = get_leads_df_cached(start_d, end_d)
    if df.empty:
        st.info("No leads yet.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(200))


# 3) Analytics & SLA
def page_analytics_sla():
    st.header("📈 Analytics & SLA")
    st.markdown("*SLA trend and pipeline breakdown.*")
    df = get_leads_df_cached(start_d, end_d)
    if df.empty:
        st.info("No leads.")
        return

    # SLA trend: count overdue per day for last 30 days
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(29, -1, -1)]
    counts = []
    for d in days:
        start_d_ = datetime.combine(d, datetime.min.time())
        end_d_ = datetime.combine(d, datetime.max.time())
        subset = df[(df["created_at"] >= start_d_) & (df["created_at"] <= end_d_)]
        overdue_count = subset[subset["status"].str.lower() == "overdue"].shape[0]
        counts.append(overdue_count)
    # chart
    fig = plt.figure(figsize=(8, 3))
    plt.plot(days, counts, marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Pipeline Stages (counts)")
    st.table(df["status"].value_counts().rename_axis("stage").reset_index(name="count"))


# 4) CPA & ROI
def page_cpa_roi():
    st.header("💰 CPA & ROI")
    df = get_leads_df_cached(start_d, end_d)
    if df.empty:
        st.info("No leads")
        return
    total_spend = float(df["cost_to_acquire"].sum())
    df_awarded = df[df["status"].str.lower() == "awarded"]
    won_count = int(df_awarded.shape[0])
    cpa = (total_spend / won_count) if (won_count and total_spend) else (0.0 if total_spend else 0.0)
    pipeline_value = float(df["estimated_value"].sum())
    roi = pipeline_value - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    st.markdown(f"**Total Marketing Spend:** ${total_spend:,.2f}")
    st.markdown(f"**Conversions (Won):** {won_count}")
    st.markdown(f"**CPA:** ${cpa:,.2f}")
    st.markdown(f"**ROI:** ${roi:,.2f} ({roi_pct:.1f}%)")

    # chart: spend vs conversions simple
    fig = plt.figure()
    plt.bar(["Spend", "Conversions"], [total_spend, won_count], color=["#ef4444", "#2563eb"])
    st.pyplot(fig)


# 5) Settings
def page_settings():
    st.header("⚙️ Settings")
    st.markdown("*Lead sources, basic app settings*")
    st.markdown("Lead sources are visible in lead capture (configurable).")
    st.text("Note: Role management, SMTP and advanced settings are in Users page.")


# 6) Users (role manager)
def page_users():
    st.header("👥 Users & Roles")
    s = get_session()
    try:
        users = s.query(User).order_by(User.username).all()
        if not users:
            st.info("No users yet (they are created on login).")
        else:
            for u in users:
                cols = st.columns([2, 2, 2, 1])
                name = cols[0].text_input("Full name", value=u.full_name or "", key=f"uname_{u.username}")
                role = cols[1].selectbox("Role", ["Viewer", "Estimator", "Adjuster", "Tech", "Admin"], index=["Viewer", "Estimator", "Adjuster", "Tech", "Admin"].index(u.role if u.role in ["Viewer", "Estimator", "Adjuster", "Tech", "Admin"] else "Viewer"), key=f"urole_{u.username}")
                if cols[3].button("Save", key=f"saveuser_{u.username}"):
                    try:
                        uu = s.query(User).filter(User.username == u.username).first()
                        uu.full_name = st.session_state.get(f"uname_{u.username}", uu.full_name)
                        uu.role = st.session_state.get(f"urole_{u.username}", uu.role)
                        s.commit()
                        st.success(f"Saved {u.username}")
                    except Exception:
                        s.rollback()
    finally:
        s.close()


# 7) Exports
def page_exports():
    st.header("📤 Exports")
    df = get_leads_df_cached(start_d, end_d)
    if df.empty:
        st.info("No leads")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download leads.csv", csv_bytes, file_name="leads_export.csv", mime="text/csv")


# route pages
page_name = st.session_state.get("page", "Pipeline Board")
try:
    if page_name == "Pipeline Board":
        page_pipeline_board(leads_df)
    elif page_name == "Leads / Capture":
        page_leads_capture()
    elif page_name == "Analytics & SLA":
        page_analytics_sla()
    elif page_name == "CPA & ROI":
        page_cpa_roi()
    elif page_name == "Settings":
        page_settings()
    elif page_name == "Users":
        page_users()
    elif page_name == "Exports":
        page_exports()
    else:
        st.info("Page not found.")
except Exception as e:
    st.error("An error occurred while rendering the page.")
    st.write(traceback.format_exc())

# footer
st.markdown("---")
st.markdown("<div class='small'>Project X — Stable · Streamlit</div>", unsafe_allow_html=True)
