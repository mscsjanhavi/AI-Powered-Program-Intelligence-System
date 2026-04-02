import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG (LIGHT + CLEAN)
# -------------------------------
st.set_page_config(page_title="AI Program Dashboard", layout="wide")

st.title("📊 AI-Powered Program Intelligence Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("project_data.csv", header=None)
df = df[0].str.split(",", expand=True)

df.columns = df.iloc[0]
df = df[1:]
df = df.reset_index(drop=True)

# -------------------------------
# DATA CLEANING
# -------------------------------
date_cols = ["Start_Date", "Planned_End_Date", "Actual_End_Date"]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df["Estimated_Hours"] = pd.to_numeric(df["Estimated_Hours"])
df["Actual_Hours"] = pd.to_numeric(df["Actual_Hours"])

# -------------------------------
# KPIs + FEATURES
# -------------------------------
df["Delay_Days"] = (df["Actual_End_Date"] - df["Planned_End_Date"]).dt.days
df["Is_Delayed"] = df["Delay_Days"].apply(lambda x: 1 if x > 0 else 0)
df["Effort_Variance"] = df["Actual_Hours"] - df["Estimated_Hours"]
df["Planned_Duration"] = (df["Planned_End_Date"] - df["Start_Date"]).dt.days

df["Risk_Score"] = df["Is_Delayed"]*2 + (df["Effort_Variance"] > 0)*1

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

owner_filter = st.sidebar.multiselect(
    "Owner", df["Owner"].unique(), default=df["Owner"].unique()
)
status_filter = st.sidebar.multiselect(
    "Status", df["Status"].unique(), default=df["Status"].unique()
)

df_filtered = df[
    (df["Owner"].isin(owner_filter)) &
    (df["Status"].isin(status_filter))
]

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tasks", len(df_filtered))
col2.metric("Completed", df_filtered[df_filtered["Status"]=="Completed"].shape[0])
col3.metric("Delayed", df_filtered[df_filtered["Is_Delayed"]==1].shape[0])
col4.metric("High Risk", df_filtered[df_filtered["Risk_Score"]>=2].shape[0])

# -------------------------------
# CHARTS
# -------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("📊 Task Status Distribution")
    st.bar_chart(df_filtered["Status"].value_counts())

with colB:
    st.subheader("⏱ Estimated vs Actual Hours")
    st.bar_chart(df_filtered[["Estimated_Hours", "Actual_Hours"]])

# -------------------------------
# RISK HEATMAP
# -------------------------------
st.subheader("🔥 Risk Overview")

def highlight_risk_row(row):
    if row["Risk_Score"] >= 2:
        return ["background-color: #ffcccc"] * len(row)
    elif row["Risk_Score"] == 1:
        return ["background-color: #fff3cd"] * len(row)
    else:
        return ["background-color: #d4edda"] * len(row)

st.dataframe(df_filtered.style.apply(highlight_risk_row, axis=1))

# -------------------------------
# AI PREDICTION
# -------------------------------
st.subheader("🤖 Predict Task Delay")

col1, col2, col3 = st.columns(3)

planned_duration = col1.number_input("Planned Duration", 1, 50, 10)
estimated_hours = col2.number_input("Estimated Hours", 1, 200, 40)
actual_hours = col3.number_input("Actual Hours", 1, 200, 50)

# Train model
df_model = df.dropna(subset=["Is_Delayed"])
X = df_model[["Planned_Duration", "Estimated_Hours", "Actual_Hours"]].fillna(0)
y = df_model["Is_Delayed"]

model = RandomForestClassifier()
model.fit(X, y)

if st.button("🔮 Predict"):
    prediction = model.predict([[planned_duration, estimated_hours, actual_hours]])

    if prediction[0] == 1:
        st.error("⚠️ High chance of delay")
    else:
        st.success("✅ Task likely on time")

# -------------------------------
# AI REPORT
# -------------------------------
st.subheader("🧠 Generate AI Report")

if st.button("📄 Generate Report"):
    total = len(df_filtered)
    completed = df_filtered[df_filtered["Status"]=="Completed"].shape[0]
    delayed = df_filtered[df_filtered["Is_Delayed"]==1].shape[0]
    high_risk = df_filtered[df_filtered["Risk_Score"]>=2].shape[0]

    status = "Project is facing delays" if delayed > 0 else "Project is on track"
    risk = "High risk tasks present" if high_risk > 0 else "Risk level is low"

    st.write("### 📊 Executive Summary")
    st.write(f"Status: {status}")
    st.write(f"Total Tasks: {total}")
    st.write(f"Completed: {completed}")
    st.write(f"Delayed: {delayed}")

    st.write("### ⚠️ Risk Analysis")
    st.write(risk)

    st.write("### 💡 Recommendations")
    st.write("- Focus on delayed tasks")
    st.write("- Reallocate resources")
    st.write("- Monitor progress closely")

# -------------------------------
# DATA TABLE
# -------------------------------
st.subheader("📋 Project Data")
st.dataframe(df_filtered)