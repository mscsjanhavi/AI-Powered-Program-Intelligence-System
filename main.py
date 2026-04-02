import pandas as pd

# --- LOAD & FIX DATA ---
df = pd.read_csv("project_data.csv", header=None)
df = df[0].str.split(",", expand=True)

df.columns = df.iloc[0]
df = df[1:]
df = df.reset_index(drop=True)

# --- CONVERT TYPES ---
date_cols = ["Start_Date", "Planned_End_Date", "Actual_End_Date"]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df["Estimated_Hours"] = pd.to_numeric(df["Estimated_Hours"])
df["Actual_Hours"] = pd.to_numeric(df["Actual_Hours"])

# --- KPIs ---
df["Delay_Days"] = (df["Actual_End_Date"] - df["Planned_End_Date"]).dt.days
df["Is_Delayed"] = df["Delay_Days"].apply(lambda x: 1 if x > 0 else 0)

df["Effort_Variance"] = df["Actual_Hours"] - df["Estimated_Hours"]

# --- FEATURE ENGINEERING ---
df["Planned_Duration"] = (df["Planned_End_Date"] - df["Start_Date"]).dt.days
df["Actual_Duration"] = (df["Actual_End_Date"] - df["Start_Date"]).dt.days

df["Is_Completed"] = df["Status"].apply(lambda x: 1 if x == "Completed" else 0)
df["Overrun"] = df["Effort_Variance"].apply(lambda x: 1 if x > 0 else 0)

# --- RISK SCORE ---
def calculate_risk(row):
    risk = 0
    if row["Is_Delayed"] == 1:
        risk += 2
    if row["Overrun"] == 1:
        risk += 1
    if row["Status"] == "In Progress":
        risk += 1
    return risk

df["Risk_Score"] = df.apply(calculate_risk, axis=1)

# --- SUMMARY ---
print("\n--- KPI SUMMARY ---")

total_tasks = len(df)
completed_tasks = df[df["Status"] == "Completed"].shape[0]
completion_percent = (completed_tasks / total_tasks) * 100
delayed_tasks = df[df["Is_Delayed"] == 1].shape[0]

print(f"Total Tasks: {total_tasks}")
print(f"Completed Tasks: {completed_tasks}")
print(f"Completion %: {completion_percent:.2f}%")
print(f"Delayed Tasks: {delayed_tasks}")

# --- FINAL DATA ---
print("\n--- FINAL DATA ---")
print(df)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Select features (inputs)
features = ["Planned_Duration", "Estimated_Hours", "Actual_Hours"]

# Drop rows with missing values
df_model = df.dropna(subset=features + ["Is_Delayed"])

X = df_model[features]
y = df_model["Is_Delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")


# --- NEW TASK INPUT ---
new_task = pd.DataFrame({
    "Planned_Duration": [5],
    "Estimated_Hours": [20],
    "Actual_Hours": [18]
})


prediction = model.predict(new_task)

print("\n--- NEW TASK PREDICTION ---")

if prediction[0] == 1:
    print("⚠️ This task is likely to be DELAYED")
else:
    print("✅ This task is likely ON TIME")



# --- NEW TASK INPUT ---
new_task = pd.DataFrame({
    "Planned_Duration": [10],
    "Estimated_Hours": [50],
    "Actual_Hours": [60]
})

# Predict
prediction = model.predict(new_task)

print("\n--- NEW TASK PREDICTION ---")

if prediction[0] == 1:
    print("⚠️ This task is likely to be DELAYED")
else:
    print("✅ This task is likely ON TIME")


total_tasks = len(df)
completed_tasks = df[df["Status"] == "Completed"].shape[0]
delayed_tasks = df[df["Is_Delayed"] == 1].shape[0]

high_risk_tasks = df[df["Risk_Score"] >= 2].shape[0]



print("\n--- AI POWERED REPORT (SIMULATED) ---\n")

if delayed_tasks > 0:
    status = "Project is experiencing delays."
else:
    status = "Project is on track."

if high_risk_tasks > 0:
    risk = "There are high-risk tasks requiring attention."
else:
    risk = "Risk level is low."

report = f"""
Executive Summary:
{status}

Risk Analysis:
{risk}

Recommendations:
- Focus on delayed tasks
- Optimize resource allocation
- Monitor progress closely
"""

print(report)