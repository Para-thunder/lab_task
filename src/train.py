import pandas as pd
import yaml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Extract parameters
test_size = params["features"]["test_size"]
random_state = params["features"]["random_state"]
n_estimators = params["train"]["n_estimators"]
max_depth = params["train"]["max_depth"]
average = params["evaluate"]["average"]

# Load dataset
df = pd.read_csv("data/iris_dataset.csv")

# Split dataset
X = df.drop("Species", axis=1)
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=average)

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"✅ Training complete — Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
