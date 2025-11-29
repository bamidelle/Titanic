# project_x_full_upgraded.py
# Upgraded single-file Streamlit app: fixes login/navigation + scheduler + estimates + customer portal + auto-followups + perf polish
# Requirements: streamlit, pandas, sqlalchemy, plotly (optional), fpdf (optional)
# Run: streamlit run project_x_full_upgraded.py

import os
import time
from datetime import datetime, timedelta, date
import io
import random
import traceback

import streamlit as st
import pandas as pd

# Optional libs
try:
    import plotly.express as px
except Exception:
    px = None

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# Database (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

BASE_DIR = os.getcwd()
DB_FILE = os.path.join(BASE_DIR, "project_x_upgraded.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# -----------------------
# Models
# -----------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
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
    owner = Column(String, default="")  # user assigned

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    updated_by = Column(String)
    update_time = Column(DateTime, default=datetime.utcnow)
    old_status = Column(String)
    new_status = Column(String)
    note = Column(Text)

Base.metadata.create_all(bind=engine)

# -----------------------
# Utilities
# -----------------------
def get_session():
    return SessionLocal()

def save_file(uploaded, prefix="file"):
    if uploaded is None:
        return None
    name = f"{prefix}_{int(datetime.utcnow().timestamp())}_{uploaded.name}"
    path = os.path.join(UPLOAD_FOLDER, name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

@st.cache_data(ttl=30)
def cached_leads(start_date=None, end_date=None):
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append({
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
                "sla_hours": r.sla_hours,
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
                "owner": r.owner or ""
            })
        df = pd.DataFrame(data)
        if df.empty:
            return df
        if start_date and end_date:
            sdt = datetime.combine(start_date, datetime.min.time())
            edt = datetime.combine(end_date, datetime.max.time())
            df = df[(df["created_at"] >= sdt) & (df["created_at"] <= edt)].copy()
        return df
    finally:
        s.close()

def add_lead_to_db(payload):
    s = get_session()
    try:
        lead = Lead(**payload)
        s.add(lead)
        s.commit()
        return lead.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def update_lead_db(lead_id, updates, actor=None, note=None):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.id == lead_id).first()
        if not lead:
            return False
        old_status = lead.status
        for k, v in updates.items():
            setattr(lead, k, v)
        s.add(lead)
        s.add(LeadHistory(lead_id=lead_id, updated_by=actor or "", old_status=old_status, new_status=lead.status, note=note or ""))
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def compute_sla_remaining(entered, hours):
    if entered is None:
        entered = datetime.utcnow()
    if isinstance(entered, str):
        try:
            entered = datetime.fromisoformat(entered)
        except:
            entered = datetime.utcnow()
    deadline = entered + timedelta(hours=int(hours or 24))
    rem = deadline - datetime.utcnow()
    return rem.total_seconds(), rem.total_seconds() <= 0

# -----------------------
# UI Styling and header
# -----------------------
st.set_page_config(page_title="Project X — Pro", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
body, .stApp { font-family: 'Comfortaa', sans-serif; background: #ffffff; }
.header {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 18px; border-bottom:1px solid #f0f0f0;
}
.brand { font-weight:800; font-size:20px; color:#0b1220; }
.subtle { color:#6b7280; font-size:13px; }
.kpi-card { background:#000; color:#fff; padding:12px; border-radius:10px; min-height:96px; }
.kpi-title { font-size:12px; color:#fff; font-weight:700; }
.kpi-value { font-size:22px; font-weight:900; margin-top:6px; }
.progress-wrap{ background:#111; height:8px; border-radius:8px; margin-top:8px; overflow:hidden; }
.progress-fill{ height:100%; transition:width .4s ease; }
.topbar { display:flex; gap:10px; align-items:center; }
.button-like { background:#000;color:#fff;padding:8px 12px;border-radius:8px; cursor:pointer; }
.small-muted{ color:#6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOGIN (fixes the earlier issue)
# -----------------------
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

# Simple login UI in the sidebar
with st.sidebar:
    st.markdown("## Account")
    username = st.text_input("Username", value=st.session_state.user or "")
    if st.button("Login"):
        if username.strip() == "":
            st.warning("Enter a username")
        else:
            # create or fetch user
            s = get_session()
            try:
                u = s.query(User).filter(User.username == username).first()
                if not u:
                    u = User(username=username, full_name=username, role="Admin" if username.lower() == "admin" else "Viewer")
                    s.add(u); s.commit()
                st.session_state.user = u.username
                st.session_state.role = u.role
                st.session_state.full_name = u.full_name or u.username
                # refresh UI so main app appears immediately
                st.experimental_rerun()
            finally:
                s.close()
    if st.session_state.user:
        st.markdown(f"Signed in as **{st.session_state.user}**  \nRole: *{st.session_state.role or 'Viewer'}*")
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

# block before login
if not st.session_state.user:
    st.markdown("<div style='padding:18px'><h2>Welcome — please sign in</h2><p class='small-muted'>Use any username (e.g. 'AyoBami'). We'll create a profile for you.</p></div>", unsafe_allow_html=True)
    st.stop()

# ---------------
# NAVIGATION
# ---------------
PAGES = ["Pipeline", "Leads", "Scheduler", "Estimates", "CPA & ROI", "Customer Portal", "Auto-Followups", "Settings", "Users", "Exports"]
if "page" not in st.session_state:
    st.session_state.page = "Pipeline"

with st.container():
    st.markdown(f"""
    <div class='header'>
      <div>
        <span class='brand'>Project X — Pro</span><br>
        <span class='small-muted'>Restoration Pipeline & Field Ops</span>
      </div>
      <div class='topbar'>
        <div class='small-muted'>User: <b>{st.session_state.user}</b></div>
        <div style='width:12px'></div>
        <select id='nav_select' onChange='window.location.href=\"?nav=\"+this.value'>
          <option value='Pipeline'>Pipeline</option>
        </select>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Streamlit-native nav using radio for speed
selected = st.sidebar.radio("Go to", PAGES, index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0)
st.session_state.page = selected

# global date range controls (top-right like Google Ads)
col_l, col_r = st.columns([3,1])
with col_r:
    date_quick = st.selectbox("Range", ["Today","Last 7 days","Last 30 days","All","Custom"], index=0)
    if date_quick == "Today":
        dt_start = date.today(); dt_end = date.today()
    elif date_quick == "Last 7 days":
        dt_start = date.today() - timedelta(days=6); dt_end = date.today()
    elif date_quick == "Last 30 days":
        dt_start = date.today() - timedelta(days=29); dt_end = date.today()
    elif date_quick == "All":
        dt_start = None; dt_end = None
    else:
        custom = st.date_input("Start - End", [date.today() - timedelta(days=6), date.today()])
        if isinstance(custom, (list,tuple)) and len(custom)==2:
            dt_start, dt_end = custom[0], custom[1]
        else:
            dt_start, dt_end = date.today(), date.today()

# -----------------------
# Page: Pipeline (main)
# -----------------------
def page_pipeline():
    st.title("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance. KPI cards show current snapshot.</em>", unsafe_allow_html=True)

    df = cached_leads(dt_start, dt_end)
    total_leads = len(df)
    qualified = len(df[df["qualified"]==True]) if not df.empty else 0
    sla_success = int(df["contacted"].sum()) if not df.empty else 0
    awarded = len(df[df["status"].str.lower()=="awarded"]) if not df.empty else 0
    lost = len(df[df["status"].str.lower()=="lost"]) if not df.empty else 0
    closed = awarded + lost
    conversion_rate = (awarded/closed*100) if closed else 0
    inspection_pct = (df["inspection_scheduled"].sum() / max(1, qualified) * 100) if not df.empty else 0
    estimate_sent = int(df["estimate_submitted"].sum()) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    total_spend = float(df["cost_to_acquire"].sum()) if not df.empty else 0.0
    active = total_leads - (awarded + lost)

    KPI = [
        ("Active Leads", active, "#2563eb", (active / max(1,total_leads))*100 if total_leads else 0),
        ("SLA Success", f"{(sla_success/ max(1,total_leads))*100:.1f}%", "#0ea5a4", (sla_success / max(1,total_leads))*100 if total_leads else 0),
        ("Qualification Rate", f"{(qualified/ max(1,total_leads))*100:.1f}%", "#a855f7", (qualified / max(1,total_leads))*100 if total_leads else 0),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "#f97316", conversion_rate),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "#ef4444", inspection_pct),
        ("Estimates Sent", estimate_sent, "#6d28d9", (estimate_sent / max(1,total_leads))*100 if total_leads else 0),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "#22c55e", min(100, pipeline_job_value / max(1, total_leads*5000) * 100))
    ]

    # two rows: 4 + 3
    r1 = st.columns(4)
    r2 = st.columns(3)
    for col,(title,val,color,pct) in zip(r1, KPI[:4]):
        col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{color};'>{val}</div><div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%;background:{color};'></div></div></div>", unsafe_allow_html=True)
    for col,(title,val,color,pct) in zip(r2, KPI[4:]):
        col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{color};'>{val}</div><div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%;background:{color};'></div></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("TOP 5 PRIORITY LEADS")
    st.markdown("<em>Priority is computed from estimated value, SLA urgency and flags (internal scoring).</em>", unsafe_allow_html=True)

    # compute score simple formula (value / baseline) + SLA urgency
    weights = {"value_baseline":5000.0, "sla_weight":0.4, "value_weight":0.6}
    pr_list = []
    for _, row in df.iterrows():
        val = row.get("estimated_value") or 0.0
        vs = min(1.0, val / weights["value_baseline"])
        sla_sec, overdue = compute_sla_remaining(row.get("sla_entered_at") or row.get("created_at"), row.get("sla_hours") or 24)
        time_left_h = sla_sec/3600.0 if sla_sec not in (None, float("inf")) else 9999.0
        sla_component = max(0.0, (72.0 - min(time_left_h,72.0)) / 72.0)
        score = vs * weights["value_weight"] + sla_component * weights["sla_weight"]
        pr_list.append({"id": row["id"], "name": row["contact_name"], "value": val, "time_left_h": time_left_h, "overdue": overdue, "score": score, "status": row["status"]})
    pr_df = pd.DataFrame(pr_list).sort_values("score", ascending=False).head(5)
    if pr_df.empty:
        st.info("No priority leads")
    else:
        for _, r in pr_df.iterrows():
            time_html = f"<span style='color:#ef4444;font-weight:700;'>{int(r['time_left_h'])}h</span>" if not r["overdue"] else "<span style='color:#ef4444;font-weight:900;'>OVERDUE</span>"
            money_html = f"<span style='color:#22c55e;font-weight:800;'>${r['value']:,.0f}</span>"
            st.markdown(f"<div style='padding:10px;border-radius:8px;border:1px solid #eee;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;'><div><b>#{int(r['id'])} — {r['name'] or 'No name'}</b><div class='small-muted'>Status: {r['status']}</div></div><div style='text-align:right;'>{money_html}<div style='margin-top:6px;color:#ef4444'>{time_html}</div></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("All leads")
    st.markdown("<em>Expand a lead to edit status, assign owner, add notes, upload invoice.</em>", unsafe_allow_html=True)

    # Search and filters
    c1, c2, c3 = st.columns([2,2,4])
    with c1:
        f_status = st.selectbox("Status", ["All"] + sorted(df["status"].dropna().unique().tolist()) if not df.empty else ["All"])
    with c2:
        f_owner = st.text_input("Owner (quick filter)")
    with c3:
        q = st.text_input("Search name / phone / email / address")
    df_view = df.copy()
    if f_status and f_status != "All":
        df_view = df_view[df_view["status"]==f_status]
    if f_owner:
        df_view = df_view[df_view["owner"].str.contains(f_owner, case=False, na=False)]
    if q:
        ql = q.lower()
        df_view = df_view[df_view.apply(lambda r: ql in str(r["contact_name"]).lower() or ql in str(r["contact_phone"]).lower() or ql in str(r["contact_email"]).lower() or ql in str(r["property_address"]).lower(), axis=1)]

    # show limited rows for speed, allow expand
    show_n = st.number_input("Rows to show", min_value=5, max_value=500, value=25, step=5)
    st.dataframe(df_view.sort_values("created_at", ascending=False).head(show_n))

    # quick lead create inline
    with st.expander("Quick add lead"):
        form = st.form("quick_add")
        cname = form.text_input("Name")
        cphone = form.text_input("Phone")
        cemail = form.text_input("Email")
        csource = form.selectbox("Source", ["Google Ads","Website","Referral","Facebook","Instagram","TikTok","LinkedIn","Other"])
        cval = form.number_input("Est. job value", min_value=0.0, value=0.0, step=100.0)
        ccost = form.number_input("Cost to acquire", min_value=0.0, value=0.0, step=1.0)
        csubmit = form.form_submit_button("Add")
        if csubmit:
            payload = {
                "contact_name": cname, "contact_phone": cphone, "contact_email": cemail,
                "source": csource, "estimated_value": float(cval), "cost_to_acquire": float(ccost),
                "sla_entered_at": datetime.utcnow(), "created_at": datetime.utcnow(), "status": "New"
            }
            try:
                add_lead_to_db(payload)
                st.success("Lead added")
                # clear cache and rerun
                cached_leads.clear()
                st.experimental_rerun()
            except Exception as e:
                st.error("Failed to add lead: " + str(e))

# -----------------------
# Page: Scheduler
# -----------------------
def page_scheduler():
    st.title("📅 Tech Job Calendar — Scheduler")
    st.markdown("<em>Assign inspection dates to leads and view upcoming jobs.</em>", unsafe_allow_html=True)
    df = cached_leads(dt_start, dt_end)
    if df.empty:
        st.info("No leads in selected range.")
        return
    # show upcoming scheduled inspection jobs
    scheduled = df[df["inspection_scheduled"]==True].copy()
    scheduled["inspection_date"] = scheduled["inspection_scheduled_at"]
    scheduled = scheduled.sort_values("inspection_date").head(200)
    st.subheader("Upcoming Inspections")
    if scheduled.empty:
        st.info("No inspections scheduled yet.")
    else:
        st.dataframe(scheduled[["id","contact_name","assigned_to","inspection_scheduled_at","property_address"]].rename(columns={"inspection_scheduled_at":"inspection_date"}))

    # schedule new inspection
    st.markdown("---")
    st.subheader("Schedule an Inspection")
    ids = df["id"].tolist()
    sel = st.selectbox("Lead", [""] + [f"#{i} — {df.loc[df['id']==i,'contact_name'].values[0] if i in df['id'].values else ''}" for i in ids])
    if sel:
        lid = int(sel.split()[0].lstrip("#"))
        dt = st.date_input("Inspection Date", date.today())
        time_input = st.time_input("Inspection time", datetime.now().time())
        tech = st.text_input("Assign Tech", value="")
        if st.button("Schedule"):
            dt_full = datetime.combine(dt, time_input)
            try:
                update_lead_db(lid, {"inspection_scheduled": True, "inspection_scheduled_at": dt_full, "assigned_to": tech}, actor=st.session_state.user, note="Inspection scheduled")
                st.success("Inspection scheduled")
                cached_leads.clear()
            except Exception as e:
                st.error("Failed to schedule: " + str(e))

# -----------------------
# Page: Estimates (PDF/HTML)
# -----------------------
def page_estimates():
    st.title("🧾 Estimates & Invoices")
    st.markdown("<em>Create quick estimates and download as HTML or PDF.</em>", unsafe_allow_html=True)
    df = cached_leads(dt_start, dt_end)
    if df.empty:
        st.info("No leads.")
        return
    lid = st.selectbox("Lead for estimate", [""] + [f"#{int(r)} — {df.loc[df['id']==r,'contact_name'].values[0]}" for r in df["id"].tolist()])
    if lid:
        lead_id = int(lid.split()[0].lstrip("#"))
        lead_row = df[df["id"]==lead_id].iloc[0].to_dict()
        st.write("Lead:", lead_row["contact_name"], lead_row["contact_phone"])
        items = st.text_area("Estimate line items (one per line, format: description | qty | unit_price)", value="Inspection | 1 | 150\nRemediation | 1 | 1200")
        notes = st.text_area("Notes / Terms", value="Estimate valid for 30 days.")
        if st.button("Generate Estimate"):
            try:
                lines = []
                total = 0.0
                for line in items.splitlines():
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        desc, qty, up = parts[0], float(parts[1]), float(parts[2])
                        amt = qty * up
                        lines.append({"desc":desc,"qty":qty,"unit":up,"amt":amt})
                        total += amt
                # build HTML
                html = f"""
                <html><head><meta charset='utf-8'><title>Estimate #{lead_id}</title></head>
                <body style='font-family:Arial;padding:20px;'>
                <h2>Estimate for {lead_row['contact_name']}</h2>
                <p>Address: {lead_row.get('property_address','')}</p>
                <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%'>
                <tr><th>Description</th><th>Qty</th><th>Unit</th><th>Amount</th></tr>
                """
                for it in lines:
                    html += f"<tr><td>{it['desc']}</td><td>{it['qty']}</td><td>${it['unit']:.2f}</td><td>${it['amt']:.2f}</td></tr>"
                html += f"<tr><td colspan='3' style='text-align:right;font-weight:bold;'>Total</td><td style='font-weight:bold;'>${total:.2f}</td></tr>"
                html += f"</table><p>{notes}</p></body></html>"
                b = html.encode("utf-8")
                st.success("Estimate generated")
                st.download_button("Download HTML Estimate", data=b, file_name=f"estimate_{lead_id}.html", mime="text/html")
                if FPDF_AVAILABLE:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(0,10, f"Estimate for {lead_row['contact_name']}", ln=True)
                    pdf.cell(0,8, f"Address: {lead_row.get('property_address','')}", ln=True)
                    pdf.ln(6)
                    for it in lines:
                        pdf.cell(0,8, f"{it['desc']} — {it['qty']} x ${it['unit']:.2f} = ${it['amt']:.2f}", ln=True)
                    pdf.ln(6)
                    pdf.cell(0,8, f"Total: ${total:.2f}", ln=True)
                    pdf_out = pdf.output(dest='S').encode('latin-1')
                    st.download_button("Download PDF Estimate", data=pdf_out, file_name=f"estimate_{lead_id}.pdf", mime="application/pdf")
                else:
                    st.info("fpdf not installed — PDF download disabled (HTML available).")
            except Exception as e:
                st.error("Failed to generate estimate: " + str(e))

# -----------------------
# Page: CPA & ROI
# -----------------------
def page_cpa():
    st.title("💰 CPA & ROI")
    st.markdown("<em>Marketing spend (cost_to_acquire) vs conversions (Awarded).</em>", unsafe_allow_html=True)
    df = cached_leads(dt_start, dt_end)
    if df.empty:
        st.info("No data.")
        return
    total_spend = float(df["cost_to_acquire"].sum())
    conversions = int(df[df["status"].str.lower()=="awarded"].shape[0])
    cpa = total_spend / conversions if conversions else 0.0
    revenue = float(df[df["status"].str.lower()=="awarded"]["estimated_value"].sum())
    roi_val = revenue - total_spend
    roi_pct = (roi_val/total_spend*100) if total_spend else 0.0

    st.markdown(f"**Total Marketing Spend:** <span style='color:#ef4444;font-weight:700;'>${total_spend:,.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"**Conversions (Won):** <span style='color:#2563eb;font-weight:700;'>{conversions}</span>", unsafe_allow_html=True)
    st.markdown(f"**CPA:** <span style='color:#f97316;font-weight:700;'>${cpa:,.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"**ROI:** <span style='color:#22c55e;font-weight:700;'>${roi_val:,.2f} ({roi_pct:.1f}%)</span>", unsafe_allow_html=True)

    # chart (simple)
    if px:
        agg = df.copy()
        agg["date"] = agg["created_at"].dt.date
        daily = agg.groupby("date").agg(total_spend=("cost_to_acquire","sum"), conversions=("id", lambda s: df.loc[s.index,"status"].str.lower().eq("awarded").sum())).reset_index()
        fig = px.line(daily, x="date", y=["total_spend","conversions"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("plotly not available — chart disabled.")

# -----------------------
# Page: Customer Portal (lightweight)
# -----------------------
def page_customer_portal():
    st.title("🔒 Customer Portal (View by email + token)")
    st.markdown("<em>A lightweight view that allows customers to check their lead status with email + access token (email-based token).</em>", unsafe_allow_html=True)
    email = st.text_input("Customer Email")
    token = st.text_input("Access token (enter last 4 characters of email for demo)")
    if st.button("View"):
        if not email:
            st.warning("Enter an email")
        else:
            # simple token check: last 4 letters must match for demo
            ok = token == (email[-4:] if len(email) >=4 else email)
            if not ok:
                st.error("Invalid token")
            else:
                s = get_session()
                try:
                    leads = s.query(Lead).filter(Lead.contact_email == email).order_by(Lead.created_at.desc()).all()
                    if not leads:
                        st.info("No records found for this email")
                    else:
                        for L in leads:
                            st.markdown(f"**Lead #{L.id} — {L.contact_name}**  \nStatus: **{L.status}**  \nEstimated Value: ${L.estimated_value or 0:,.2f}  \nCreated: {L.created_at}")
                finally:
                    s.close()

# -----------------------
# Page: Auto-Followups
# -----------------------
def page_autofollow():
    st.title("✉️ Auto Follow-up Workflows")
    st.markdown("<em>Define message templates and queue follow-ups. Sending is simulated (no SMTP by default).</em>", unsafe_allow_html=True)
    st.subheader("Templates")
    if "af_templates" not in st.session_state:
        st.session_state.af_templates = {"Default": "Hi {name}, following up on your request for {address}. Reply to schedule."}
    tname = st.text_input("Template name", value="FollowUp")
    tbody = st.text_area("Template body (use {name}, {address})", value="Hi {name}, following up on your lead at {address}. Please reply.")
    if st.button("Save Template"):
        st.session_state.af_templates[tname] = tbody
        st.success("Template saved")
    st.write("Saved templates:")
    for k,v in st.session_state.af_templates.items():
        st.markdown(f"**{k}** — {v}")

    st.markdown("---")
    st.subheader("Queue follow-ups")
    df = cached_leads(dt_start, dt_end)
    lead_options = [""] + [f"#{int(r)} — {df.loc[df['id']==r,'contact_name'].values[0]}" for r in df["id"].tolist()] if not df.empty else [""]
    sel = st.selectbox("Select lead to queue follow-up", lead_options)
    tpl = st.selectbox("Template", list(st.session_state.af_templates.keys()))
    when = st.date_input("Send date", date.today())
    if st.button("Queue Follow-up"):
        if not sel:
            st.warning("Select lead")
        else:
            lid = int(sel.split()[0].lstrip("#"))
            # store queue in session (simple persistent list)
            if "af_queue" not in st.session_state:
                st.session_state.af_queue = []
            st.session_state.af_queue.append({"lead_id": lid, "template": tpl, "send_on": when.isoformat(), "queued_by": st.session_state.user})
            st.success("Queued")
    st.subheader("Queued follow-ups")
    qdf = pd.DataFrame(st.session_state.get("af_queue", []))
    if qdf.empty:
        st.info("No queued follow-ups")
    else:
        st.dataframe(qdf)

    if st.button("Export queued as CSV"):
        dfq = pd.DataFrame(st.session_state.get("af_queue", []))
        st.download_button("Download CSV", data=dfq.to_csv(index=False).encode("utf-8"), file_name="followup_queue.csv", mime="text/csv")

# -----------------------
# Page: Settings & Users
# -----------------------
def page_settings():
    st.title("⚙️ Settings")
    st.markdown("<em>App-wide settings, lead source manager, alert toggles, and user/role admin.</em>", unsafe_allow_html=True)
    st.subheader("Lead Sources")
    sources = ["Google Ads","Website","Referral","Facebook","Instagram","TikTok","LinkedIn","Yelp","Walk-In","Hotline"]
    for s in sources:
        st.checkbox(s, value=True, key=f"src_{s}")

    st.subheader("Alerts")
    st.checkbox("Enable SLA email alerts (requires SMTP)", key="enable_sla_alerts")
    st.checkbox("Show alert bell", value=True, key="show_bell")

    st.subheader("Users (Admin only)")
    if st.session_state.role != "Admin":
        st.info("Only Admin can manage users")
    else:
        s = get_session()
        try:
            users = s.query(User).all()
            for u in users:
                cols = st.columns([3,2,2])
                cols[0].text_input("Full name", value=u.full_name or "", key=f"u_name_{u.username}")
                cols[1].selectbox("Role", ["Viewer","Estimator","Adjuster","Tech","Admin"], index=["Viewer","Estimator","Adjuster","Tech","Admin"].index(u.role if u.role in ["Viewer","Estimator","Adjuster","Tech","Admin"] else "Viewer"), key=f"u_role_{u.username}")
                if cols[2].button("Save", key=f"save_u_{u.username}"):
                    s2 = get_session()
                    try:
                        uu = s2.query(User).filter(User.username==u.username).first()
                        uu.full_name = st.session_state.get(f"u_name_{u.username}", uu.full_name)
                        uu.role = st.session_state.get(f"u_role_{u.username}", uu.role)
                        s2.commit()
                        st.success("Saved")
                    finally:
                        s2.close()
        finally:
            s.close()

# -----------------------
# Page: Exports
# -----------------------
def page_exports():
    st.title("📤 Exports")
    st.markdown("Export leads, estimates, follow-up queue.")
    df = cached_leads(dt_start, dt_end)
    if df.empty:
        st.info("No leads")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")

# -----------------------
# ROUTE
# -----------------------
try:
    page = st.session_state.page
    if page == "Pipeline":
        page_pipeline()
    elif page == "Leads":
        # simply reuse pipeline list
        st.title("Leads")
        st.markdown("<em>Full lead list and management.</em>", unsafe_allow_html=True)
        df = cached_leads(dt_start, dt_end)
        if df.empty:
            st.info("No leads")
        else:
            st.dataframe(df.sort_values("created_at", ascending=False).head(200))
    elif page == "Scheduler":
        page_scheduler()
    elif page == "Estimates":
        page_estimates()
    elif page == "CPA & ROI":
        page_cpa()
    elif page == "Customer Portal":
        page_customer_portal()
    elif page == "Auto-Followups":
        page_autofollow()
    elif page == "Settings":
        page_settings()
    elif page == "Users":
        st.title("Users")
        s = get_session()
        try:
            ulist = s.query(User).all()
            st.dataframe(pd.DataFrame([{"username":u.username,"full_name":u.full_name,"role":u.role} for u in ulist]))
        finally:
            s.close()
    elif page == "Exports":
        page_exports()
    else:
        st.info("Page not implemented")
except Exception as e:
    st.error("App error: " + str(e))
    st.write(traceback.format_exc())

# -----------------------
# End
# -----------------------
st.markdown("<hr><div class='small-muted'>Project X — Pro · Built with Streamlit</div>", unsafe_allow_html=True)
