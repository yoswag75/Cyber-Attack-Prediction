# 🛡️ Cyber Attack Prediction System

A comprehensive **Machine Learning and Deep Learning application** designed to detect network intrusions and cyber attacks. This system utilizes an interactive **Streamlit dashboard** to demonstrate the power of ensemble learning and advanced data augmentation techniques (**SMOTE + Tomek**) in handling imbalanced cybersecurity datasets.

---

## 🚀 Key Features

* **Interactive Dashboard:** Built with Streamlit for real-time data analysis and model interaction.
* **Multi-Model Architecture:**
* **Random Forest Classifier:** Baseline model utilizing balanced class weights.
* **Deep Learning (ANN):** Custom neural network optimized with Batch Normalization and Dropout layers.
* **Augmented Ensemble:** Advanced model trained on data balanced via SMOTE + Tomek links.


* **Automated EDA:** Real-time Exploratory Data Analysis including correlation heatmaps and distribution plots.
* **Real-time Prediction Engine:** A simulation engine that generates random network traffic samples to predict their nature (Normal vs. Attack).
* **Performance Metrics:** Detailed visualization of ROC-AUC curves, Confusion Matrices, and Learning Curves.

---

## 🛠️ Tech Stack

| Category | Technologies |
| --- | --- |
| **Frontend** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, Imbalanced-Learn (SMOTE/Tomek), XGBoost, LightGBM |
| **Deep Learning** | TensorFlow (Keras) |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## 📦 Installation

Follow these steps to set up the environment and run the application locally.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cyber-attack-prediction.git
cd cyber-attack-prediction

```

### 2. Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn xgboost lightgbm plotly scipy

```

---

## ▶️ Usage

To launch the application, run the following command in your terminal:

```bash
streamlit run main.py

```

### Navigating the App

The dashboard is divided into several sections:

1. **🏠 Home:** Overview of the project, architecture details, and objectives.
2. **📊 Data Analysis:** Click **"Load Dataset"** to fetch data. Explore feature statistics and correlation matrices.
3. **🤖 Model Training:** Click **"Start Training Pipeline"** to preprocess data, train the ANN and Random Forest models, and apply Data Augmentation.
4. **📈 Results:** Compare model accuracy, ROC-AUC scores, and confusion matrices side-by-side.
5. **🔮 Predictions:** Generate random synthetic network traffic and observe how different models vote on whether the traffic is an "Attack" or "Normal."

---

## 🧠 Model Architectures

### 1. Deep Learning Model (ANN)

We utilize a custom Artificial Neural Network designed for binary classification.

* **Input Layer:** Dynamically sized based on feature selection.
* **Hidden Layers:**
* `Dense (256 units)` + `BatchNorm` + `Dropout(0.3)`
* `Dense (128 units)` + `BatchNorm` + `Dropout(0.3)`
* `Dense (64 units)` + `BatchNorm` + `Dropout(0.2)`


* **Output Layer:** Sigmoid activation function.
* **Optimization:** Adam Optimizer with Early Stopping and Learning Rate Reduction on Plateau.

### 2. Data Augmentation Strategy

Cybersecurity datasets are notoriously imbalanced (few attacks vs. massive amounts of normal traffic). We address this using **SMOTE + Tomek**:

* **SMOTE (Synthetic Minority Over-sampling Technique):** Generates synthetic examples of attack vectors to balance the dataset.
* **Tomek Links:** Cleans the data by removing overlapping samples between classes to create a clearer decision boundary.

---

## 📂 Project Structure

```text
├── main.py                     # Main Streamlit application entry point
├── requirements.txt            # List of python dependencies
├── README.md                   # Project documentation
└── cybersecurity_intrusion_data.csv  # (Optional, fetched automatically if missing)

```

---

## 📊 Dataset

The application is capable of automatically fetching a sample dataset if one is not present. The dataset includes network traffic features such as:

* **Protocol Type:** TCP, UDP, ICMP
* **Encryption Used:** AES, DES, None
* **Browser Type**
* **Session Duration** & **Packet Size**
* **Login Attempts**

---

## 📝 License

This project is licensed under the GNU License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
