import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "Capstone_Predictive_main/data/engine_data.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)
float_cols = df.select_dtypes(include=['float64', 'float32']).columns
df[float_cols] = df[float_cols].round(2)

X = df.drop(columns=["Engine Condition"])
y = df["Engine Condition"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["Type"].unique()))
