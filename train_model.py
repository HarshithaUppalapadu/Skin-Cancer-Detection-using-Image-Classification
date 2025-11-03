# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    # 1️⃣ Load dataset
    data = pd.read_csv("dataset.csv")   # replace with your dataset name
    X = data.drop("target", axis=1)     # replace "target" with your actual target column
    y = data["target"]

    # 2️⃣ Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3️⃣ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4️⃣ Save model
    joblib.dump(model, "model.pkl")
    print("✅ Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_and_save_model()
