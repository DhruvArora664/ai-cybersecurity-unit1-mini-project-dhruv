"""
Building an AI-Driven Cyber Threat Awareness and Detection Prototype
Academic Mini-Project: Cyber Threat Detection using the TON_IoT Network Dataset
"""

import os
import glob
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import kagglehub


def main():
    print("=" * 70)
    print("AI-Driven Cyber Threat Awareness & Detection Prototype (TON_IoT)")
    print("=" * 70)

    # -------------------------------------------------------------
    # Task 2: Dataset Acquisition & Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    print("\n[+] Downloading / Checking TON_IoT Dataset via kagglehub...")
    dataset_dir = kagglehub.dataset_download("arnobbhowmik/ton-iot-network-dataset")
    print(f"[+] Dataset located at: {dataset_dir}")

    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    print(f"[+] Discovered {len(csv_files)} CSV files. Selecting: {os.path.basename(csv_files[0])}")
    selected_csv = csv_files[0]

    # Load representative sample (or full dataset)
    df = pd.read_csv(selected_csv, nrows=100000, low_memory=False)
    print(f"[+] Loaded records: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n--- Class Distribution ---")
    if 'label' in df.columns:
        print(df['label'].value_counts(dropna=False))

    # Generate EDA Plots and save to file
    print("\n[+] Generating Task 2 EDA Visualizations...")
    sns.set_theme(style="darkgrid", palette="mako")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Count plot of normal vs suspicious
    label_names = {0: 'Normal (0)', 1: 'Suspicious / Attack (1)'}
    plot_labels = df['label'].map(label_names) if df['label'].dtype in [np.int64, int] else df['label']
    sns.countplot(x=plot_labels, ax=axes[0], palette=["#2ec4b6", "#e71d36"])
    axes[0].set_title("Traffic Distribution: Normal vs. Suspicious Activities", fontweight="bold")
    axes[0].set_xlabel("Traffic Category")
    axes[0].set_ylabel("Flow Count")

    # Plot 2: Distribution of key network feature
    key_feat = 'duration' if 'duration' in df.columns else 'src_bytes'
    temp_feat = pd.to_numeric(df[key_feat], errors='coerce').fillna(0)
    log_feat = np.log1p(temp_feat[temp_feat >= 0])
    sns.histplot(data=df, x=log_feat, hue='label', kde=True, bins=35, ax=axes[1], palette=["#2ec4b6", "#e71d36"], element="step")
    axes[1].set_title(f"Distribution of Network Feature: log1p({key_feat})", fontweight="bold")
    axes[1].set_xlabel(f"log(1 + {key_feat})")
    axes[1].set_ylabel("Density / Frequency")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=200)
    plt.close()
    print("[+] Saved EDA visualization to eda_plots.png")

    # -------------------------------------------------------------
    # Task 3: Machine Learning Threat Classification (Decision Tree)
    # -------------------------------------------------------------
    print("\n[+] Preprocessing Telemetry Data...")
    df_clean = df.replace(['-', '?', ' -', '- '], np.nan)
    target_col = 'label'
    y = df_clean[target_col].astype(int)

    cols_to_drop = [c for c in ['ts', 'src_ip', 'dst_ip', 'type', 'label'] if c in df_clean.columns]
    X_raw = df_clean.drop(columns=cols_to_drop)

    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    for c in num_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce').fillna(
            X_raw[c].median() if not X_raw[c].dropna().empty else 0
        )
    for c in cat_cols:
        X_raw[c] = X_raw[c].fillna('unknown')
        le = LabelEncoder()
        X_raw[c] = le.fit_transform(X_raw[c].astype(str))

    scaler = StandardScaler()
    X_scaled = X_raw.copy()
    X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train split: {len(X_train):,} | Test split: {len(X_test):,}")

    print("\n[+] Training Decision Tree Classifier...")
    dt_start = time.time()
    dt = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    dt_time = time.time() - dt_start

    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_prec = precision_score(y_test, dt_preds, zero_division=0)
    dt_rec = recall_score(y_test, dt_preds, zero_division=0)
    dt_f1 = f1_score(y_test, dt_preds, zero_division=0)

    print("-" * 55)
    print(f"Decision Tree Accuracy : {dt_acc * 100:.2f}%")
    print(f"Decision Tree Precision: {dt_prec:.4f}")
    print(f"Decision Tree Recall   : {dt_rec:.4f}")
    print(f"Decision Tree F1-Score : {dt_f1:.4f}")
    print(f"Decision Tree Time     : {dt_time:.4f}s")
    print("-" * 55)

    # -------------------------------------------------------------
    # Task 4: Deep Learning Threat Classification (MLP Neural Net)
    # -------------------------------------------------------------
    print("\n[+] Training Multi-Layer Perceptron (MLP) Neural Network...")
    mlp_start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=25, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - mlp_start

    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    mlp_prec = precision_score(y_test, mlp_preds, zero_division=0)
    mlp_rec = recall_score(y_test, mlp_preds, zero_division=0)
    mlp_f1 = f1_score(y_test, mlp_preds, zero_division=0)

    print("-" * 55)
    print(f"MLP Neural Net Accuracy : {mlp_acc * 100:.2f}%")
    print(f"MLP Neural Net Precision: {mlp_prec:.4f}")
    print(f"MLP Neural Net Recall   : {mlp_rec:.4f}")
    print(f"MLP Neural Net F1-Score : {mlp_f1:.4f}")
    print(f"MLP Neural Net Time     : {mlp_time:.4f}s")
    print("-" * 55)

    print(f"\n[Comparison] MLP training took {mlp_time:.2f}s vs Decision Tree {dt_time:.2f}s ({mlp_time/max(dt_time, 1e-4):.1f}x slower).")
    print(f"[Comparison] Decision Tree F1: {dt_f1:.4f} | MLP F1: {mlp_f1:.4f}")

    # -------------------------------------------------------------
    # Task 5: Ethical Reflection & Aggressive Posture Simulation
    # -------------------------------------------------------------
    print("\n[+] Running Aggressive Security Posture Simulation...")
    mlp_probs = mlp.predict_proba(X_test)[:, 1]
    strict_thresh = 0.30
    strict_preds = (mlp_probs >= strict_thresh).astype(int)

    cm = confusion_matrix(y_test, strict_preds)
    tn, fp, fn, tp = cm.ravel()
    total_normal = tn + fp
    fp_pct = (fp / total_normal) * 100

    print("=" * 65)
    print(f"SIMULATION: Threat Threshold set to > {strict_thresh*100:.0f}%")
    print(f"Total Normal User Sessions Tested : {total_normal:,}")
    print(f"False Positives (Users Blocked)   : {fp:,} ({fp_pct:.2f}%)")
    print(f"True Positives (Attacks Blocked)  : {tp:,}")
    print(f"False Negatives (Attacks Missed)  : {fn:,}")
    # Save Confusion Matrix Visual Comparison
    print("\n[+] Generating Confusion Matrix Comparison Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_default = confusion_matrix(y_test, mlp_preds)

    ConfusionMatrixDisplay(confusion_matrix=cm_default, display_labels=['Normal', 'Suspicious']).plot(
        ax=axes[0], cmap='Blues', colorbar=False
    )
    axes[0].set_title("Standard Posture (Threshold = 0.50)", fontweight='bold')
    axes[0].grid(False)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Suspicious']).plot(
        ax=axes[1], cmap='Reds', colorbar=False
    )
    axes[1].set_title(f"Aggressive Posture (Threshold = {strict_thresh:.2f})", fontweight='bold')
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig("confusion_matrix_simulation.png", dpi=200)
    plt.close()
    print("[+] Saved confusion matrix visualization to confusion_matrix_simulation.png")

    print("\n[!] Execution completed successfully! Notebook available at src/main.ipynb.")


if __name__ == "__main__":
    main()
