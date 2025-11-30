# project_x_ui_update.py
# Streamlit single-file app with updated UI:
# - Donut chart in Analytics (pipeline stages)
# - Alert bell icon + dropdown with dismissible alerts
# - Top 5 priority leads as black cards (money green, time-left red)
# - CPA & ROI on black cards with colored titles
# - Medium-size spend/conversions chart
# - Main page titles red
# - Font Comfortaa
# - Submit buttons red long with white text
# - Selected date text shown above pages
# - Settings placeholders for PDF estimates, email followups, calendar, AI recommendations

import os
from datetime import datetime, timedelta, date
import traceback

import streamlit as st
import pandas as pd

# optional plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------- CONFIG ----------
DB_FILE = "project_x_ui_update.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# ---------- MODELS ----------
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
    status = Column(String, default="New")  # possible: New, Contacted, Inspection Scheduled, Inspection Completed, Estimate Submitted, Awarded, Lost, Overdue
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
    time_left_hours = Column(Integer, default=72)  # helper for UI demo

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
    updated_at = Column(DateTime, default=datetime.utcnow)
    old_status = Column(String)
    new_status = Column(String)
    note = Column(Text)

Base.metadata.create_all(engine)

# ---------- UTILITIES ----------
def get_session():
    return SessionLocal()

def row_to_dict(r):
    return {
        "id": r.id,
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
        "status": r.status,
        "created_at": r.created_at,
        "sla_hours": int(r.sla_hours or 24),
        "sla_entered_at": r.sla_entered_at or r.created_at,
        "contacted": bool(r.contacted),
        "inspection_scheduled": bool(r.inspection_scheduled),
        "inspection_scheduled_at": r.inspection_scheduled_at,
        "inspection_completed": bool(r.inspection_completed),
        "estimate_submitted": bool(r.estimate_submitted),
        "awarded_date": r.awarded_date,
        "awarded_invoice": r.awarded_invoice,
        "lost_date": r.lost_date,
        "qualified": bool(r.qualified),
        "cost_to_acquire": float(r.cost_to_acquire or 0.0),
        "owner": r.owner,
        "priority_score": float(r.priority_score or 0.0),
        "time_left_hours": int(r.time_left_hours or 72)
    }

@st.cache_data(ttl=30)
def load_leads(start_date=None, end_date=None):
    s = get_session()
    try:
        q = s.query(Lead).order_by(Lead.created_at.desc()).all()
        rows = [row_to_dict(r) for r in q]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if start_date:
            df = df[df["created_at"] >= datetime.combine(start_date, datetime.min.time())]
        if end_date:
            df = df[df["created_at"] <= datetime.combine(end_date, datetime.max.time())]
        return df.reset_index(drop=True)
    finally:
        s.close()

def add_lead(payload):
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
            estimated_value=float(payload.get("estimated_value") or 0.0),
            sla_hours=int(payload.get("sla_hours") or 24),
            sla_entered_at=payload.get("sla_entered_at") or datetime.utcnow(),
            qualified=bool(payload.get("qualified")),
            cost_to_acquire=float(payload.get("cost_to_acquire") or 0.0),
            owner=payload.get("owner") or None,
            priority_score=float(payload.get("priority_score") or 0.0),
            time_left_hours=int(payload.get("time_left_hours") or 72)
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

def update_lead_db(lead_id, updates, actor=None, note=None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.id==lead_id).first()
        if not lead:
            return False
        old_status = lead.status
        for k,v in updates.items():
            if hasattr(lead, k):
                setattr(lead, k, v)
        s.add(LeadHistory(lead_id=lead_id, updated_by=actor or "", old_status=old_status, new_status=lead.status, note=note or ""))
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
        remain = deadline - datetime.utcnow()
        return max(remain.total_seconds(), 0.0), (remain.total_seconds() <= 0)
    except Exception:
        return float("inf"), False

def compute_priority_simple(lead_row):
    # value baseline scoring + SLA urgency
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
    except Exception:
        val = 0.0
    value_score = min(1.0, val / 5000.0)
    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    try:
        if isinstance(sla_entered, str):
            sla_entered = datetime.fromisoformat(sla_entered)
        deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
        time_left_h = max((deadline - datetime.utcnow()).total_seconds()/3600.0, 0.0)
    except Exception:
        time_left_h = 9999.0
    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)
    return round(value_score*0.6 + sla_score*0.4, 3)

# ---------- UI CSS ----------
st.set_page_config(page_title="Project X — UI Update", layout="wide")
st.markdown(""" 
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
body, .stApp { font-family: 'Comfortaa', sans-serif; background: #ffffff; color: #0b1220; }
.header { display:flex; justify-content:space-between; align-items:center; padding:10px 12px; border-bottom:1px solid #f1f5f9; }
.brand { font-weight:800; font-size:20px; color:#b91c1c; } /* main title red */
.subtle { color:#64748b; font-size:13px; }
.kpi-card { background:#000; color:#fff; padding:14px; border-radius:10px; min-height:94px; }
.kpi-title { font-weight:700; font-size:13px; color:#fff; }
.kpi-value { font-weight:900; font-size:20px; margin-top:8px; }
.progress { height:8px; border-radius:8px; margin-top:8px; }
.topbar { display:flex; gap:12px; align-items:center; }
.alert-bell { cursor:pointer; font-size:20px; padding:8px; border-radius:8px; background:#fff; color:#111827; border:1px solid #e6e6e6; }
.alert-badge { background:#ef4444; color:white; padding:4px 8px; border-radius:12px; font-weight:700; margin-left:6px; }
.alert-panel { position:fixed; top:64px; right:28px; width:360px; background:#111827; color:white; padding:12px; border-radius:12px; box-shadow:0 10px 30px rgba(0,0,0,0.2); z-index:9999; }
.alert-row { display:flex; justify-content:space-between; align-items:center; padding:8px 6px; border-bottom:1px solid rgba(255,255,255,0.04); }
.card-black { background:#000; color:white; padding:12px; border-radius:10px; }
.top5-card { background:#000; color:white; padding:14px; border-radius:12px; min-height:80px; }
.btn-primary { background:#b91c1c; color:white; padding:10px 18px; border-radius:10px; font-weight:700; border:none; width:100%; cursor:pointer; }
.small-muted { color:#94a3b8; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR: LOGIN & NAV ----------
with st.sidebar:
    st.markdown("## Account")
    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.role = "Viewer"
    user_input = st.text_input("Username", value=st.session_state.get("user") or "")
    if st.button("Login"):
        if user_input.strip() == "":
            st.warning("Enter a username")
        else:
            s = get_session()
            try:
                u = s.query(User).filter(User.username==user_input).first()
                if not u:
                    u = User(username=user_input, full_name=user_input, role="Admin")
                    s.add(u); s.commit()
                st.session_state.user = u.username
                st.session_state.role = u.role
                st.success(f"Signed in as {u.username} ({u.role})")
            finally:
                s.close()
    if st.session_state.get("user"):
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Pipeline Dashboard", "Leads / Capture", "Analytics", "CPA & ROI", "Settings", "Users", "Exports"], index=0)
    st.session_state.page = page

    st.markdown("---")
    st.markdown("## Date Range")
    quick = st.selectbox("Quick", ["Today","Last 7 days","Last 30 days","Custom"], index=0)
    if quick == "Today":
        start_sel = date.today(); end_sel = date.today()
    elif quick == "Last 7 days":
        start_sel = date.today() - timedelta(days=6); end_sel = date.today()
    elif quick == "Last 30 days":
        start_sel = date.today() - timedelta(days=29); end_sel = date.today()
    else:
        dates = st.date_input("Start - End", [date.today()-timedelta(days=6), date.today()])
        start_sel, end_sel = (dates[0], dates[1]) if isinstance(dates, (list,tuple)) else (date.today()-timedelta(days=6), date.today())
    st.session_state.start_date = start_sel
    st.session_state.end_date = end_sel

# ensure login
if not st.session_state.get("user"):
    st.warning("Please sign in from the sidebar to continue.")
    st.stop()

# show selected date range text above pages (requested)
st.markdown(f"**Selected range:** {st.session_state.start_date.isoformat()} → {st.session_state.end_date.isoformat()}")

# load leads (cached)
leads_df = load_leads(st.session_state.start_date, st.session_state.end_date)

# ---------- Alert bell setup ----------
if "alerts" not in st.session_state:
    # initialize alerts list using current overdue leads
    try:
        overdue_count = int(leads_df[leads_df["status"].str.lower()=="overdue"].shape[0]) if not leads_df.empty else 0
    except Exception:
        overdue_count = 0
    st.session_state.alerts = []
    # populate example alerts so user sees them
    if overdue_count > 0:
        for _, r in (leads_df[leads_df["status"].str.lower()=="overdue"].head(5).iterrows() if not leads_df.empty else []):
            st.session_state.alerts.append({"id": int(r["id"]), "msg": f"Lead #{int(r['id'])} overdue — {r.get('contact_name')}", "time": datetime.utcnow().isoformat()})
    st.session_state.alerts_dismissed = []

def render_alert_bell():
    overdue_count = len([a for a in st.session_state.alerts if a["id"] not in st.session_state.alerts_dismissed])
    col1, col2 = st.columns([1,8])
    with col1:
        if st.button("🔔", key="alert_bell"):
            st.session_state.show_alerts = not st.session_state.get("show_alerts", False)
    with col2:
        if overdue_count:
            st.markdown(f"<span class='alert-badge'>{overdue_count} overdue</span>", unsafe_allow_html=True)

    if st.session_state.get("show_alerts", False):
        # render right-hand dropdown panel
        html = "<div class='alert-panel'>"
        html += "<div style='display:flex;justify-content:space-between;align-items:center;'><b>🚨 SLA Alerts</b><span style='cursor:pointer;color:#fff;font-weight:800;' onclick=\"window.parent.document.querySelectorAll('[data-testid=stButton]')[0].click()\">✖</span></div><hr style='border-color:#ffffff22'/>"
        for a in st.session_state.alerts:
            if a["id"] in st.session_state.alerts_dismissed:
                continue
            html += f"<div class='alert-row'><div><b>Lead {a['id']}</b><div class='small-muted'>{a['msg']}</div></div><div><button onclick='window.parent.postMessage({{\"dismiss\":{a['id']}}}, \"*\")' class='btn'>Dismiss</button></div></div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# listen for dismiss messages from client (JS). Fallback: provide a dismiss UI
# Because Streamlit can't directly receive postMessage, also provide UI to dismiss.
if st.session_state.alerts:
    st.markdown("**Alerts:**")
    for a in st.session_state.alerts:
        if a["id"] in st.session_state.alerts_dismissed:
            continue
        cols = st.columns([8,1])
        cols[0].markdown(f"**Lead {a['id']}** — {a['msg']}  \n<small class='small-muted'>{a['time']}</small>", unsafe_allow_html=True)
        if cols[1].button("X", key=f"dismiss_{a['id']}"):
            st.session_state.alerts_dismissed.append(a["id"])
            st.success("Alert dismissed")

# ---------- PAGE: Pipeline Dashboard ----------
if st.session_state.page == "Pipeline Dashboard":
    st.markdown("<h1 style='color:#b91c1c'>TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR</h1>", unsafe_allow_html=True)
    st.markdown("<em style='color:#6b7280'>High-level pipeline performance at a glance. Use the filters above to refine the data.</em>", unsafe_allow_html=True)

    # compute KPIs
    total_leads = int(leads_df.shape[0]) if not leads_df.empty else 0
    qualified = int(leads_df[leads_df["qualified"]==True].shape[0]) if not leads_df.empty else 0
    sla_success = int(leads_df["contacted"].sum()) if not leads_df.empty else 0
    awarded_count = int(leads_df[leads_df["status"].str.lower()=="awarded"].shape[0]) if not leads_df.empty else 0
    lost_count = int(leads_df[leads_df["status"].str.lower()=="lost"].shape[0]) if not leads_df.empty else 0
    inspection_scheduled_count = int(leads_df[leads_df["inspection_scheduled"]==True].shape[0]) if not leads_df.empty else 0
    estimate_sent_count = int(leads_df[leads_df["estimate_submitted"]==True].shape[0]) if not leads_df.empty else 0
    pipeline_job_value = float(leads_df["estimated_value"].sum()) if not leads_df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)
    sla_success_pct = (sla_success / total_leads * 100) if total_leads else 0.0
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
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "#22c55e")
    ]

    # render KPI cards in 2 rows (4 + 3)
    cols = st.columns(4)
    for col, (title, value, color) in zip(cols, KPI_ITEMS[:4]):
        pct = random.randint(30, 90)
        col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{color};'>{value}</div><div class='progress' style='background:{color}; width:{pct}%;'></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(3)
    for col, (title, value, color) in zip(cols2, KPI_ITEMS[4:]):
        pct = random.randint(30, 90)
        col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{color};'>{value}</div><div class='progress' style='background:{color}; width:{pct}%;'></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#b91c1c'>TOP 5 PRIORITY LEADS</h3>", unsafe_allow_html=True)
    st.markdown("<em style='color:#64748b'>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)

    # compute priority list and show top 5 as black cards
    priority_list = []
    if not leads_df.empty:
        for _, r in leads_df.iterrows():
            score = compute_priority_simple(r.to_dict())
            remaining_sec, overdue = compute_remaining_sla_seconds(r.get("sla_entered_at"), r.get("sla_hours"))
            time_left_h = int(remaining_sec // 3600) if remaining_sec not in (float("inf"), None) else 9999
            priority_list.append({
                "id": int(r["id"]),
                "name": r.get("contact_name") or "No name",
                "value": float(r.get("estimated_value") or 0.0),
                "time_left_h": time_left_h,
                "score": score,
                "status": r.get("status")
            })
    pr_df = pd.DataFrame(priority_list).sort_values("score", ascending=False).head(5)
    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        cols = st.columns(5)
        for c, (_, row) in zip(cols, pr_df.iterrows()):
            # black top5 card
            c.markdown(f"<div class='top5-card'><div style='font-weight:800;font-size:14px'>{row['name']}</div><div style='margin-top:6px;'><span style='color:#22c55e;font-weight:800;'>${row['value']:,.0f}</span> &nbsp;&nbsp; <span style='color:#ef4444;font-weight:700;'>{row['time_left_h']}h left</span></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#b91c1c'>All Leads (expand to edit / change status)</h3>", unsafe_allow_html=True)
    st.markdown("<em style='color:#64748b'>Expand a lead to edit details, upload invoice, and change status.</em>", unsafe_allow_html=True)

    # Search/filter UI
    c1, c2, c3 = st.columns([2,2,4])
    with c1:
        status_filter = st.selectbox("Status", options=["All"] + (sorted(leads_df["status"].dropna().unique().tolist()) if not leads_df.empty else []))
    with c2:
        owner_filter = st.text_input("Owner")
    with c3:
        q = st.text_input("Search name / phone / email / address")

    df_view = leads_df.copy() if not leads_df.empty else pd.DataFrame()
    if not df_view.empty:
        if status_filter and status_filter != "All":
            df_view = df_view[df_view["status"]==status_filter]
        if owner_filter:
            df_view = df_view[df_view["owner"].str.contains(owner_filter, case=False, na=False)]
        if q:
            ql = q.lower()
            df_view = df_view[df_view.apply(lambda r: ql in str(r["contact_name"]).lower() or ql in str(r["contact_phone"]).lower() or ql in str(r["contact_email"]).lower() or ql in str(r["property_address"]).lower(), axis=1)]
        # display results
        st.dataframe(df_view.sort_values("created_at", ascending=False).head(200))
        # expanders for edit
        for _, r in df_view.sort_values("created_at", ascending=False).head(100).iterrows():
            with st.expander(f"Lead #{int(r['id'])} — {r.get('contact_name') or 'No name'}"):
                colA, colB = st.columns([3,1])
                with colA:
                    st.write(f"**Source:** {r.get('source') or '—'}  |  **Assigned:** {r.get('owner') or '—'}")
                    st.write(f"**Address:** {r.get('property_address') or '—'}")
                    st.write(f"**Notes:** {r.get('notes') or '—'}")
                    st.write(f"**Created:** {r.get('created_at')}")
                    new_notes = st.text_area("Notes", value=r.get("notes") or "", key=f"notes_{int(r['id'])}")
                with colB:
                    entered = r.get("sla_entered_at") or r.get("created_at")
                    rem_sec, overdue_flag = compute_remaining_sla_seconds(entered, r.get("sla_hours") or 24)
                    if overdue_flag:
                        st.markdown("<div style='color:#ef4444;font-weight:800;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                    else:
                        hrs = int(rem_sec // 3600) if rem_sec not in (float("inf"), None) else "-"
                        mins = int((rem_sec % 3600) // 60) if rem_sec not in (float("inf"), None) else "-"
                        st.markdown(f"<div style='color:#ef4444;font-weight:700;'>⏳ {hrs}h {mins}m left</div>", unsafe_allow_html=True)

                new_status = st.selectbox("Status", ["New","Contacted","Inspection Scheduled","Inspection Completed","Estimate Submitted","Awarded","Lost","Overdue"], index=0, key=f"status_edit_{int(r['id'])}")
                new_owner = st.text_input("Assigned to", value=r.get("owner") or "", key=f"owner_edit_{int(r['id'])}")
                new_est_val = st.number_input("Job Value Estimate (USD)", value=float(r.get("estimated_value") or 0.0), min_value=0.0, step=100.0, key=f"estval_{int(r['id'])}")
                awarded_invoice_file = st.file_uploader("Upload Invoice (Awarded)", type=["pdf","jpg","png"], key=f"inv_{int(r['id'])}")
                # main submit button - red long
                if st.button("💾 Update Lead", key=f"save_lead_{int(r['id'])}"):
                    updates = {"status": new_status, "owner": new_owner, "estimated_value": float(new_est_val or 0.0), "notes": new_notes}
                    if awarded_invoice_file is not None:
                        path = os.path.join(UPLOAD_DIR, f"lead_{int(r['id'])}_inv_{awarded_invoice_file.name}")
                        with open(path, "wb") as f:
                            f.write(awarded_invoice_file.getbuffer())
                        updates["awarded_invoice"] = path
                    try:
                        update_lead_db(int(r['id']), updates, actor=st.session_state.user, note="Updated via UI")
                        st.success("Lead updated")
                        load_leads.cache_clear()
                    except Exception as e:
                        st.error(f"Failed to update lead: {e}")
                        st.write(traceback.format_exc())

# ---------- PAGE: Analytics ----------
elif st.session_state.page == "Analytics":
    st.markdown("<h1 style='color:#b91c1c'>Lead Pipeline Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<em style='color:#64748b'>Donut shows distribution across pipeline stages. Line charts show SLA trend over time.</em>", unsafe_allow_html=True)

    if leads_df.empty:
        st.info("No leads to analyze.")
    else:
        # Donut pie chart for pipeline stages
        stage_counts = leads_df["status"].value_counts().reset_index()
        stage_counts.columns = ["stage","count"]
        if PLOTLY_AVAILABLE:
            fig = px.pie(stage_counts, names="stage", values="count", hole=0.5, title="Lead Pipeline Stages (Donut)")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.table(stage_counts)

        st.markdown("---")
        st.markdown("### SLA / Overdue Trend (last 30 days)")
        # simple simulated SLA trend: count overdue per day
        today = datetime.utcnow().date()
        days = [today - timedelta(days=i) for i in range(29, -1, -1)]
        counts = []
        for d in days:
            sdt = datetime.combine(d, datetime.min.time())
            edt = datetime.combine(d, datetime.max.time())
            sub = leads_df[(leads_df["created_at"] >= sdt) & (leads_df["created_at"] <= edt)]
            overdue = sub[sub["status"].str.lower()=="overdue"].shape[0]
            counts.append(overdue)
        import matplotlib.pyplot as plt
        fig2 = plt.figure(figsize=(8,3))
        plt.plot(days, counts, marker="o")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

# ---------- PAGE: CPA & ROI ----------
elif st.session_state.page == "CPA & ROI":
    st.markdown("<h1 style='color:#b91c1c'>CPA & ROI</h1>", unsafe_allow_html=True)
    st.markdown("<em style='color:#64748b'>Total marketing spend vs conversions and ROI. Use date selector above to filter.</em>", unsafe_allow_html=True)

    total_spend = float(leads_df["cost_to_acquire"].sum()) if not leads_df.empty else 0.0
    conversions = int(leads_df[leads_df["status"].str.lower()=="awarded"].shape[0]) if not leads_df.empty else 0
    cpa = (total_spend / conversions) if conversions and total_spend else 0.0
    revenue = float(leads_df[leads_df["status"].str.lower()=="awarded"]["estimated_value"].sum()) if not leads_df.empty else 0.0
    roi_val = revenue - total_spend
    roi_pct = (roi_val / total_spend * 100) if total_spend else 0.0

    # Show metrics on black cards; titles colored differently (Red, Blue, Orange, Green)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='card-black'><div style='color:#ef4444;font-weight:800;'>Total Marketing Spend</div><div style='font-size:20px;font-weight:900;'>${total_spend:,.2f}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card-black'><div style='color:#3b82f6;font-weight:800;'>Conversions (Won)</div><div style='font-size:20px;font-weight:900;'>{conversions}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card-black'><div style='color:#f97316;font-weight:800;'>CPA</div><div style='font-size:20px;font-weight:900;'>${cpa:,.2f}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card-black'><div style='color:#22c55e;font-weight:800;'>ROI</div><div style='font-size:20px;font-weight:900;'>${roi_val:,.2f} ({roi_pct:.1f}%)</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Spend vs Conversions")
    # medium size chart
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,4))
    plt.bar(["Total Spend","Conversions"], [total_spend, conversions], color=["#ef4444","#2563eb"])
    plt.title("Total Marketing Spend vs Conversions")
    st.pyplot(fig)

# ---------- PAGE: Leads / Capture ----------
elif st.session_state.page == "Leads / Capture":
    st.markdown("<h1 style='color:#b91c1c'>Leads — Capture</h1>", unsafe_allow_html=True)
    st.markdown("<em style='color:#64748b'>Create a lead. All inputs are saved and viewable by date.</em>", unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox("Lead Source", ["Google Ads","Website","Referral","Phone","Facebook","Instagram","TikTok","Other"])
            source_details = st.text_input("Source details (UTM / notes)")
            contact_name = st.text_input("Contact name")
            contact_phone = st.text_input("Contact phone")
            contact_email = st.text_input("Contact email")
        with c2:
            property_address = st.text_input("Property address")
            damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"])
            assigned_to = st.text_input("Assigned to")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No","Yes"])
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        cost_to_acquire = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Create Lead", help="Create a new lead")
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
                "estimated_value": float(estimated_value or 0.0),
                "sla_hours": int(sla_hours),
                "sla_entered_at": datetime.utcnow(),
                "qualified": True if qualified_choice=="Yes" else False,
                "cost_to_acquire": float(cost_to_acquire or 0.0),
                "owner": assigned_to or None,
                "priority_score": compute_priority_simple({"estimated_value": estimated_value, "sla_hours": sla_hours, "sla_entered_at": datetime.utcnow()}),
                "time_left_hours": int(sla_hours)
            }
            try:
                add_lead(payload)
                st.success("Lead created")
                load_leads.cache_clear()
            except Exception as e:
                st.error(f"Failed to create lead: {e}")
                st.write(traceback.format_exc())

    st.markdown("---")
    st.subheader("Recent leads")
    if leads_df.empty:
        st.info("No leads yet.")
    else:
        st.dataframe(leads_df.sort_values("created_at", ascending=False).head(100))

# ---------- PAGE: Settings ----------
elif st.session_state.page == "Settings":
    st.markdown("<h1 style='color:#b91c1c'>Settings</h1>", unsafe_allow_html=True)
    st.markdown("<em style='color:#64748b'>Configure lead sources and Phase 2/3 feature toggles (placeholders).</em>", unsafe_allow_html=True)

    st.markdown("### Lead Sources (visible in lead capture)")
    platforms = ["Google Ads","Website","Referral","Facebook","Instagram","TikTok","LinkedIn","Hotline"]
    for p in platforms:
        st.checkbox(p, value=True, key=f"src_{p}")

    st.markdown("---")
    st.markdown("### Phase 2 / Phase 3 Feature Toggles")
    st.checkbox("Estimate PDF generator (placeholder)", key="feature_est_pdf")
    st.checkbox("Automated email follow-ups (placeholder)", key="feature_emails")
    st.checkbox("Team scheduling calendar (placeholder)", key="feature_calendar")
    st.checkbox("Internal AI recommendations (placeholder)", key="feature_ai")
    st.markdown("These features are visible as placeholders and can be connected to backends (SMTP, calendar APIs, AI models) on request.")

# ---------- PAGE: Users ----------
elif st.session_state.page == "Users":
    st.markdown("<h1 style='color:#b91c1c'>Users & Roles</h1>", unsafe_allow_html=True)
    s = get_session()
    try:
        users = s.query(User).order_by(User.username).all()
        if not users:
            st.info("No users yet (users are created on login).")
        else:
            for u in users:
                cols = st.columns([3,2,1])
                new_name = cols[0].text_input("Full name", value=u.full_name or "", key=f"uname_{u.username}")
                new_role = cols[1].selectbox("Role", ["Viewer","Estimator","Adjuster","Tech","Admin"], index=0, key=f"urole_{u.username}")
                if cols[2].button("Save", key=f"save_{u.username}"):
                    uu = s.query(User).filter(User.username==u.username).first()
                    uu.full_name = st.session_state.get(f"uname_{u.username}", uu.full_name)
                    uu.role = st.session_state.get(f"urole_{u.username}", uu.role)
                    s.commit()
                    st.success("Saved")
    finally:
        s.close()

# ---------- PAGE: Exports ----------
elif st.session_state.page == "Exports":
    st.markdown("<h1 style='color:#b91c1c'>Exports</h1>", unsafe_allow_html=True)
    df = leads_df.copy()
    if df.empty:
        st.info("No leads to export.")
    else:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", data=csv, file_name="leads_export.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>Project X — UI Update (Comfortaa font) · Built for restoration pipeline</div>", unsafe_allow_html=True)
