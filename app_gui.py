# app_gui.py
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# 1️⃣ Load saved model
model = joblib.load("model.pkl")

# 2️⃣ Create GUI
root = tk.Tk()
root.title("ML Prediction App")

tk.Label(root, text="Enter feature values:").pack()

# Example inputs
entry1 = tk.Entry(root)
entry1.pack()
entry2 = tk.Entry(root)
entry2.pack()

def predict():
    try:
        # Read input values
        val1 = float(entry1.get())
        val2 = float(entry2.get())

        # Convert to DataFrame (adjust column names as per your dataset)
        input_data = pd.DataFrame([[val1, val2]], columns=["feature1", "feature2"])
        
        # Predict
        result = model.predict(input_data)[0]
        messagebox.showinfo("Prediction Result", f"Predicted Class: {result}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Predict", command=predict).pack()
root.mainloop()
