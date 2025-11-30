import streamlit as st
import pandas as pd
import datetime, os, joblib
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- DB SETUP ----------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "titan_ml_db")
if not os.path.exists(DB_PATH):
    try:
        os.makedirs(DB_PATH, exist_ok=True)
    except PermissionError:
        DB_PATH = BASE_DIR  # fallback safe for streamlit cloud

DB_FILE = os.path.join(DB_PATH, "titanic_leads.db")
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread":False})
SessionLocal = sessionmaker(bind=engine)
session = scoped_session(SessionLocal)
Base = declarative_base()

# ---------- MODELS ----------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    client = Column(String)
    source = Column(String)
    cost = Column(Float)
    conversion = Column(String)  # Won, Lost, Pending
    score = Column(Float)  # ML lead score
    sla_hours = Column(Float)
    created = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ---------- SESSION STATE INIT ----------
if "alerts_show" not in st.session_state: st.session_state.alerts_show = True
if "alerts" not in st.session_state: st.session_state.alerts = []
if "notification_count" not in st.session_state: st.session_state.notification_count = 0
if "start_date" not in st.session_state: st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=30)
if "end_date" not in st.session_state: st.session_state.end_date = datetime.date.today()

# ---------- HELPER FUNCTIONS ----------

@st.cache_data
def get_leads_df(start, end):
    s = session()
    leads = []
    try:
        for l in s.query(Lead).all():
            if start <= l.created.date() <= end:
                leads.append({
                    "ID": l.id,
                    "Client": l.client,
                    "Source": l.source,
                    "Cost": l.cost,
                    "Conversion": l.conversion,
                    "Score": l.score,
                    "SLA (hrs)": l.sla_hours,
                    "Created": l.created.date()
                })
    except Exception:
        pass
    s.close()
    return pd.DataFrame(leads)

def df_merged(): 
    return get_leads_df(st.session_state.start_date, st.session_state.end_date)

def add_alert(msg):
    st.session_state.alerts.append(msg)
    st.session_state.notification_count = len(st.session_state.alerts)

def clear_alerts():
    st.session_state.alerts.clear()
    st.session_state.notification_count = 0

def check_overdue():
    df = df_merged()
    if df.empty:
        return pd.DataFrame(), 0
    now = datetime.datetime.utcnow()
    overdue=[]
    for row in df.to_dict("records"):
        created_dt = datetime.datetime.combine(row["Created"], datetime.time())
        sla_dt = created_dt + datetime.timedelta(hours=row.get("SLA (hrs)",24))
        hours_left = (sla_dt - now).total_seconds()/3600
        if hours_left <= 0 and row["Conversion"]!="Won":
            overdue.append(row)
    return pd.DataFrame(overdue), len(overdue)

def train_internal_ml():
    df = df_merged()
    if df.empty or "Conversion" not in df.columns:
        return None
    df = df.dropna()
    df["label"] = df["Conversion"].apply(lambda x: 1 if x=="Won" else 0)
    X = df[["cost","sla_hours"]]
    y = df["label"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestClassifier(n_estimators=120, random_state=42)
        model.fit(X_train,y_train)
        MODEL_PATH = os.path.join(DB_PATH,"leadscore_model.pkl")
        joblib.dump(model, MODEL_PATH)
        return model
    except Exception as e:
        add_alert(f"ML training failed: {str(e)}")
        return None

def load_internal_ml():
    MODEL_PATH = os.path.join(DB_PATH,"leadscore_model.pkl")
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            return None
    return None

def compute_scores():
    df = df_merged()
    model = load_internal_ml() or train_internal_ml()
    if df.empty or model is None:
        return df
    s = session()
    for row in df.to_dict("records"):
        try:
            lead = s.query(Lead).get(int(row["ID"]))
            if lead:
                pred = model.predict_proba([[row["Cost"], lead.sla_hours]])[0][1]
                lead.score = round(float(pred*100),2)
        except:
            pass
    s.commit()
    s.close()
    return df_merged()

# ---------- PAGE COMPONENTS ----------
def alerts_bell_ui():
    if st.session_state.notification_count > 0:
        st.sidebar.button(f"🔔 Alerts ({st.session_state.notification_count})", on_click=lambda:None)
    else:
        st.sidebar.button("🔔 Alerts (0)")

def alerts_popup():
    if st.session_state.alerts and st.session_state.alerts_show:
        for i,msg in enumerate(list(st.session_state.alerts)):
            st.markdown(
                f"<div style='background:#111; color:white; padding:8px; border-radius:8px; margin:4px 0;'>"
                f"{msg}"
                f"<button onclick='window.parent.postMessage({{\"action\":\"close_alert\",\"i\":{i}}},\"*\")' "
                f"style='float:right; background:none; border:none; color:#888; font-size:16px;'>✖</button></div>",
                unsafe_allow_html=True
            )

# ---------- NAVIGATION ----------
st.set_page_config(page_title="Titan Backend", layout="wide", initial_sidebar_state="expanded")

nav = st.sidebar.radio("Navigation", [
    "Dashboard","Lead Capture","Analytics","Settings","Export/Import","ML"
])

# style override for white background
st.markdown("<style>body {background:white;} .block-container {background:white;}</style>", unsafe_allow_html=True)

# ---------- SETTINGS PAGE ----------
def page_settings():
    st.title("⚙️ Admin Settings")
    s=session()
    users = [
        {"User":"Ayobami","Role":"Admin"},
        {"User":"Estimator Team","Role":"Editor"},
        {"User":"Adjusters","Role":"Viewer"}
    ]
    st.dataframe(users)
    if st.button("Clear All Alerts"):
        clear_alerts()
    s.close()

# ---------- EXPORT PAGE ----------
def page_export():
    st.title("Export / Import")
    df = df_merged()
    if df.empty: 
        st.info("No leads to export")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Leads CSV", csv, "leads.csv", "text/csv")

# ---------- ML PAGE ----------
def page_ml():
    st.title("Internal ML")
    if st.button("Train ML Model"):
        model = train_internal_ml()
        if model:
            add_alert("ML training completed ✅")

# ---------- DASHBOARD ----------
def page_dashboard():
    alerts_bell_ui()
    alerts_popup()
    alerts_popup()
    df = compute_scores()
    st.title("📊 Lead Pipeline Dashboard")

    st.markdown("<br>", unsafe_allow_html=True)

    overdue_df, over_count = check_overdue()
    if over_count>0: 
        add_alert(f"{over_count} leads overdue SLA ⚠️")

    # KPI CARDS 2 ROWS
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Leads", len(df) if not df.empty else 0)
    with cols[1]:
        st.metric("High Score Leads", len(df[df.Score>70]) if not df.empty else 0)
    with cols[2]:
        st.metric("Overdue SLA", over_count)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # LINE CHART FOR STAGES
    if not df.empty:
        stage_df = df.groupby("Conversion").size().reset_index(name="Count")
        st.line_chart(stage_df.set_index("Conversion"))

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP 5 PRIORITY BLACK CARDS
    if not df.empty:
        top5 = df.sort_values("Score", ascending=False).head(5)
        st.markdown("### 🔥 Top 5 Priority Leads")
        top_cols = st.columns(5)
        now = datetime.datetime.utcnow()
        for i,row in enumerate(top5.to_dict("records")):
            created_dt = datetime.datetime.combine(row["Created"], datetime.time())
            sla_limit = created_dt + datetime.timedelta(hours=s.query(Lead).get(row["ID"]).sla_hours)
            hours_left = max(0, (sla_limit-now).total_seconds()/3600)
            with top_cols[i]:
                st.markdown(
                f"<div style='background:#000; color:white; padding:10px; border-radius:12px;'>"
                f"<h6>{row['Client']}</h6>"
                f"<p>⏳ <span style='color:red'>{hours_left:.1f} hrs left</span></p>"
                f"<p>💰 <span style='color:#0f0'>{row['Cost']:.2f}</span></p>"
                f"<p>⭐ Score: {row['Score']}</p></div>",
                unsafe_allow_html=True)

# ---------- LEAD CAPTURE ----------
def page_lead_capture():
    st.title("➕ Capture New Lead")
    client = st.text_input("Client Name","")
    source = st.selectbox("Lead Source",["Google Ads","Referral","Website","Cold Call"])
    cost = st.number_input("Acquisition Cost",min_value=0.0,step=1.0)
    sla_hours = st.number_input("SLA Response time (hours)", min_value=1.0, step=1.0)
    if st.button("Save Lead"):
        if sla_hours<=0:
            add_alert("SLA must be greater than 0 hours ❌")
            return
        s=session()
        new=Lead(
            client=client or "Unknown",
            source=source,
            cost=float(cost),
            conversion="Pending",
            score=0,
            sla_hours=float(sla_hours),
            created=datetime.datetime.utcnow()
        )
        s.add(new)
        s.commit()
        add_alert(f"Lead #{new.id} Saved ✅")
        s.close()

# ---------- ANALYTICS ----------
def page_analytics():
    alerts_bell_ui()
    df = df_merged()
    if df.empty:
        st.info("No analytics available")
        return
    st.title("📈 Analytics & SLA")

    # COST VS CONVERSION BAR CHART
    conv_df = df.groupby("Conversion")["Cost"].sum().reset_index()
    st.bar_chart(conv_df.set_index("Conversion"))

# ---------- ROUTER ----------
if nav=="Dashboard": page_dashboard()
elif nav=="Lead Capture": page_lead_capture()
elif nav=="Analytics": page_analytics()
elif nav=="Settings": page_settings()
elif nav=="Export/Import": page_export()
elif nav=="ML": page_ml()
elif nav=="Lead Capture": page_lead_capture()

