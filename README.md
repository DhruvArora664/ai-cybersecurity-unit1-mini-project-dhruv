# AI-Driven Cyber Threat Awareness and Detection Prototype

An academic cybersecurity and machine learning mini-project that processes telemetry from the **TON_IoT Network Dataset**, explores cyber attack patterns, trains and benchmarks Machine Learning (**DecisionTreeClassifier**) and Deep Learning (**MLPClassifier**) detection engines, and evaluates security trade-offs (False Positives vs. False Negatives) under aggressive security postures.

---

## 📁 Project Directory Structure

```text
ai-cybersecurity-unit1-mini-project-<name>/
├── data/                  # Raw and cached dataset directory (gitignored)
│   └── .gitkeep
├── src/                   # Source code and notebooks
│   └── main.ipynb         # End-to-end reproducible Jupyter Notebook
├── .gitignore             # Git ignore rules for virtual environments and data files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation and reproduction guide
```

---

## 🚀 Setup & Installation Instructions

### 1. Clone the Repository & Navigate to Project
```bash
git clone <your-repository-url>
cd ai-cybersecurity-unit1-mini-project-<name>
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS / Google Colab Terminal:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate

# On Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook or Jupyter Lab
```bash
jupyter notebook src/main.ipynb
```
*(Alternatively, upload `src/main.ipynb` to Google Colab and execute each cell sequentially).*

---

## 📊 Dataset Overview: TON_IoT Network Dataset

- **Source**: Australian Defence Force Academy (ADFA) / UNSW Canberra Cyber & Kaggle (`arnobbhowmik/ton-iot-network-dataset`).
- **Telemetry**: Real-world IoT and network traffic collected using Zeek (Bro) network analysis framework.
- **Labels**:
  - `label`: Binary classification (`0` = Normal traffic, `1` = Suspicious / Attack traffic).
  - `type`: Attack category (e.g., `dos`, `ddos`, `scan`, `password`, `injection`, `xss`, `ransomware`, `backdoor`).

---

## 🔬 Workflow and Pipeline Tasks

1. **Workspace & Environment Setup**: Clean git-controlled repository with pinned dependencies and data cache exclusions.
2. **Exploratory Data Analysis (EDA)**:
   - Automated ingestion via `kagglehub`.
   - Feature inspection, null handling, and distribution profiling.
   - Comparative visualizations of attack vectors vs. normal baseline traffic.
3. **Machine Learning Model (Decision Tree)**:
   - Data preprocessing: Categorical encoding (`proto`, `service`, `conn_state`), dropping high-cardinality metadata (`ts`, `src_ip`, `dst_ip`), and numerical standardization (`StandardScaler`).
   - Benchmark evaluation using Accuracy, Precision, Recall, and F1-Score.
4. **Deep Learning Model (Multi-Layer Perceptron / MLP)**:
   - Neural network training with `MLPClassifier` using backpropagation.
   - Wall-clock training time profiling.
   - Comprehensive performance, inference latency, and complexity comparison with ML.
5. **Ethical Reflection & Threat Posture Simulation**:
   - Simulation of low-threshold threat mitigation (`threshold = 0.30`).
   - Quantification of False Positives (legitimate users and services falsely blocked).
   - Analysis of operational security trade-offs (User Denial of Service vs. Zero-Day Threat Mitigation).
