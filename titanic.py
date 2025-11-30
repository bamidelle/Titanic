import streamlit as st
import pandas as pd
import datetime
import os
import joblib
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base, Session
from sqlalchemy.orm.exc import DetachedInstanceError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- DATABASE SETUP ----------
BASE_DIR = os.path.dirname(__file__)
CLOUD_DB_DIR = os.path.join(BASE_DIR, "shake5_db")

# Ensure DB folder exists safely (Streamlit Cloud safe fallback)
try:
    os.makedirs(CLOUD_DB_DIR, exist_ok=True)
except PermissionError:
    CLOUD_DB_DIR = BASE_DIR  # fallback safe location

DB_FILE = os.path.join(CLOUD_DB_DIR, "shake5_leads.db")
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
session: scoped_session[Session] = scoped_session(SessionLocal)
Base = declarative_base()


# ---------- LEAD MODEL ----------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    client = Column(String)
    source = Column(String)
    cost = Column(Float)
    urgency = Column(String, default="Low")
    conversion = Column(String, default="Pending")
    score = Column(Float, default=0)
    sla_hours = Column(Float, default=24)
    created = Column(DateTime, default=datetime.datetime.utcnow)


# Create DB tables
Base.metadata.create_all(bind=engine)


# ---------- SESSION STATE ----------
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "notification_count" not in st.session_state:
    st.session_state.notification_count = 0
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=30)
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.date.today()


# ---------- ALERT & NOTIFICATION BELL ----------
def add_alert(msg: str):
    st.session_state.alerts.append(msg)
    st.session_state.notification_count = len(st.session_state.alerts)


def alerts_bell():
    # Google Ads style black button with alert count
    bell_label = f"🔔 Alerts ({st.session_state.notification_count})" if st.session_state.notification_count > 0 else "🔔 Alerts (0)"
    st.sidebar.button(bell_label)


# ---------- DATA FETCHING ----------
@st.cache_data
def get_leads_df(start, end):
    s: Session = session()
    records = []

    try:
        leads = s.query(Lead).all()
        for l in leads:
            if start <= l.created.date() <= end:
                records.append({
                    "ID": l.id,
                    "Client": l.client,
                    "Source": l.source,
                    "Cost": l.cost,
                    "Urgency": l.urgency,
                    "Conversion": l.conversion,
                    "Score": l.score,
                    "SLA (hrs)": l.sla_hours,
                    "Created": l.created.date()
                })
    except Exception:
        pass

    s.close()
    return pd.DataFrame(records)


def df_merged():
    return get_leads_df(st.session_state.start_date, st.session_state.end_date)


# ---------- OVERDUE SLA CHECK ----------
def check_overdue_leads():
    df = df_merged()
    if df.empty:
        return 0

    overdue_count = 0
    now = datetime.datetime.utcnow()

    for row in df.to_dict("records"):
        created_dt = datetime.datetime.combine(row["Created"], datetime.time())
        sla_limit = created_dt + datetime.timedelta(hours=row.get("SLA (hrs)", 24))
        hours_left = (sla_limit - now).total_seconds() / 3600

        if hours_left <= 0 and row["Conversion"] != "Won":
            overdue_count += 1

    if overdue_count > 0:
        add_alert(f"{overdue_count} Lead(s) exceeded SLA! ⚠")

    return overdue_count


# ---------- INTERNAL ML MODEL ----------
def load_internal_ml():
    MODEL_FILE = os.path.join(CLOUD_DB_DIR, "raff_bundle.pkl")
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except:
            return None
    return None


def auto_train_model():
    MODEL_FILE = os.path.join(CLOUD_DB_DIR, "raff_bundle.pkl")
    df = df_merged()
    if df.empty:
        return None

    training_df = df.dropna()
    training_df["label"] = training_df["Conversion"].apply(lambda x: 1 if x == "Won" else 0)
    X = training_df[["Cost", "SLA (hrs)"]].values
    y = training_df["label"].values

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=140, random_state=42)
        model.fit(X_train, y_train)

        bundle = {"model": model, "trained": datetime.datetime.utcnow()}
        joblib.dump(bundle, MODEL_FILE)
        add_alert("ML model auto-trained ✅")
        return model

    except Exception as e:
        add_alert(f"ML training broke: {str(e)}")
        return None


def compute_lead_scores():
    df = df_merged()
    if df.empty:
        return df

    model = load_internal_ml()
    if not model:
        model = auto_train_model()
    if not model:
        return df

    s = session()

    for row in df.to_dict("records"):
        try:
            lead = s.get(Lead, int(row["ID"]))
            if lead:
                prob = model.predict_proba([[row["Cost"], lead.sla_hours]])[0][1]
                lead.score = round(float(prob * 100), 2)
        except:
            pass

    s.commit()
    s.close()
    return df_merged()


# ---------- PAGES ----------
def page_dashboard():
    alerts_bell()
    df = compute_lead_scores().sort_values("Score", ascending=False) if not df_merged().empty else df_merged()

    # Remove donut chart from pipeline dashboard
    st.markdown("<style>body{background:white;} .block-container{background:white;}</style>", unsafe_allow_html=True)

    st.title("📊 Dashboard")
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI CARDS IN 2 ROWS WITH SPACING
    if not df.empty:
        kpis = [["Total Leads", len(df)], ["Avg Score", round(df.Score.mean(), 2)], ["Overdue SLA", check_overdue_leads()]]
        row1, row2 = st.columns(3), st.columns(3)

        for i, (label, value) in enumerate(kpis[:3]):
            row1[i].metric(label, value)

        st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP 5 PRIORITY LEADS UI BLOCK
    st.markdown("### 🔥 Top 5 Priority Leads")
    if not df.empty:
        top5 = df.head(5)
        now = datetime.datetime.utcnow()
        s = session()

        cols = st.columns(5)
        for i, row in enumerate(top5.to_dict("records")):
            lead = s.get(Lead, row["ID"])
            created_dt = datetime.datetime.combine(row["Created"], datetime.time())
            sla_limit = created_dt + datetime.timedelta(hours=lead.sla_hours)
            hours_left = max(0, (sla_limit - now).total_seconds() / 3600)

            with cols[i]:
                st.markdown(
                f"<div style='background:#000; padding:12px; border-radius:14px;'>"
                f"<h6 style='color:#fff;'>#{row['ID']} - {row['Client']}</h6>"
                f"<p style='color:{row['UrgencyColor']};'>🔥 {row['UrgencyLabel']}</p>"
                f"<p style='color:red;'>⏳ {hours_left:.1f} hrs left</p>"
                f"<p style='color:lime;'>💰 ${row['Cost']:.2f}</p>"
                f"<p style='color:#fff;'>⭐ Score: {row['Score']}</p></div>",
                unsafe_allow_html=True
                )

        s.close()


def page_lead_capture():
    st.markdown("<style>body{background:white;} .block-container{background:white;}</style>", unsafe_allow_html=True)
    st.title("➕ New Lead")

    client = st.text_input("Client Name", placeholder="Enter client name")
    source = st.selectbox("Lead Source", ["Google Ads", "Referral", "Website", "Partner"])
    cost = st.number_input("Acquisition Cost ($)", min_value=0.0, step=1.0)

    # SLA HOURS INPUT VALIDATION > 0
    sla_hours = st.number_input("SLA Response Time (hours)", min_value=1.0, step=1.0, help="Must be greater than 0")

    urgency = st.selectbox("Urgency Level", ["High", "Medium", "Low"])

    if st.button("Save Lead"):
        if sla_hours <= 0:
            add_alert("SLA hours must be greater than 0 ❌")
            return

        s = session()
        new = Lead(
            client=client or "Unknown",
            source=source,
            cost=float(cost),
            urgency=urgency,
            conversion="Pending",
            score=0,
            sla_hours=float(sla_hours),
            created=datetime.datetime.utcnow()
        )
        s.add(new)
        s.commit()

        # Avoid DetachedInstanceError by cloning lead object
        try:
            lead_id = new.id
        except DetachedInstanceError:
            lead_id = s.query(Lead).order_by(Lead.id.desc()).first().id

        add_alert(f"Lead #{lead_id} saved ✅")
        s.close()


def page_analytics():
    st.markdown("<style>body{background:white;} .block-container{background:white;}</style>", unsafe_allow_html=True)
    st.title("📈 Analytics & SLA")

    df = df_merged()
    if df.empty:
        st.info("No analytic data")
        return

    # COST VS CONVERSION BAR CHART AT TOP OF ANALYTICS
    cv = df.groupby("Conversion")["Cost"].sum().reset_index()
    st.bar_chart(cv.set_index("Conversion"))


def page_settings_admin():
    st.markdown("<style>body{background:white;} .block-container{background:white;}</style>", unsafe_allow_html=True)
    st.title("⚙️ Settings (Admin)")
    st.info("User roles, lead assignment, and audit trails will be managed via WordPress frontend later.")


def page_export_import():
    st.title("Export / Import")
    df = df_merged()
    if df.empty:
        st.info("No leads to export")
    else:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "leads.csv", "text/csv")


def page_ml_internal():
    st.title("🤖 ML Engine")
    if st.button("Run auto-train"):
        auto_train_model()


# ---------- ROUTING ----------
if nav == "Dashboard":
    page_dashboard()
elif nav == "Lead Capture":
    page_lead_capture()
elif nav == "Analytics":
    page_analytics()
elif nav == "Settings":
    page_settings_admin()
elif nav == "Export/Import":
    page_export_import()
elif nav == "ML":
    page_ml_internal()
