import pandas as pd

RAW_PATH = "Capstone_Predictive_main/data/engine_data.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure",
    "lub oil temp", "Coolant temp", "Engine condition" ]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Failure distribution:")
print(df["Failure"].value_counts())
