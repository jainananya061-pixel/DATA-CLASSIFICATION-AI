# 🌸 Data Classification Using AI

> **Artificial Intelligence Project 2**  
> An end-to-end Machine Learning web application built with **Streamlit** and **Scikit-learn** to classify flower species from morphological measurements.

---

## 📌 Project Overview

**Data Classification Using AI** is a beginner-friendly machine learning project designed for academic demonstration and hands-on understanding of supervised classification. The system trains a **Multinomial Logistic Regression** model on the famous **Iris dataset** (built directly into Scikit-learn) and serves an interactive web interface allowing users to explore data, evaluate model metrics, and test custom predictions in real time.

---

## 🚀 Key Features

1. **Modern AI Dashboard UI:**
   - Styled with a dark navy background (`#0B0F19`), neon purple and blue accents, custom glassmorphism cards, and intuitive navigation.
2. **Zero External Downloads:**
   - Uses Scikit-learn's built-in Iris dataset (`load_iris()`). Runs 100% offline out-of-the-box.
3. **Comprehensive Dataset Explorer:**
   - Displays 150 samples across 4 botanical features: Sepal Length, Sepal Width, Petal Length, and Petal Width (in cm).
   - Filter by species (*Setosa*, *Versicolor*, *Virginica*).
   - Descriptive summary statistics table (mean, standard deviation, percentiles).
4. **Reproducible Machine Learning Pipeline:**
   - **80:20 Stratified Train/Test Split** (120 training rows, 30 testing rows) with `random_state=42` to eliminate data leakage.
   - **Logistic Regression Classifier** (`max_iter=200`, `solver='lbfgs'`).
5. **Rigorous Evaluation Metrics:**
   - **~96.7% Test Accuracy** (29 out of 30 samples classified correctly).
   - Interactive **Confusion Matrix Heatmap** (Seaborn).
   - Complete **Classification Report** table with Precision, Recall, and F1-Scores.
6. **Live Interactive Prediction:**
   - Real-time sliders for all 4 flower dimensions.
   - 1-click **Quick Presets** to test *Setosa*, *Versicolor*, or *Virginica* configurations.
   - Predicts target species with confidence score and class probability distribution bar chart.
7. **Educational AI Workflow Guide:**
   - Visual flowchart illustrating the 6 key stages of applied AI:  
     `Data Loading` ➔ `Data Exploration` ➔ `Train/Test Split` ➔ `Model Training` ➔ `Evaluation` ➔ `Prediction`

---

## 🛠️ Technology Stack

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language |
| **Web Interface** | Streamlit | Reactive web dashboard & input widgets |
| **Machine Learning** | Scikit-learn (`sklearn`) | Dataset loading, train/test split & Logistic Regression |
| **Data Manipulation** | Pandas & NumPy | Tabular data processing and numerical arrays |
| **Data Visualizations**| Matplotlib & Seaborn | Confusion matrix heatmaps, boxplots, and distributions |

---

## 🔄 The AI Workflow

```text
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│  1. Data Load   │ ────> │  2. Data Exploration │ ────> │ 3. Train/Test Split  │
│ Scikit-learn    │       │ Check shapes, nulls, │       │ 80% Train : 20% Test │
│ 150 Iris rows   │       │ feature statistics   │       │ Stratified (Seed 42) │
└─────────────────┘       └──────────────────────┘       └──────────────────────┘
                                                                    │
┌─────────────────┐       ┌──────────────────────┐                  ▼
│  6. Prediction  │ <──── │    5. Evaluation     │ <──── ┌──────────────────────┐
│ Real-time test  │       │ Accuracy (~96.7%),   │       │  4. Model Training   │
│ with probabilities      │ Confusion Matrix, F1 │       │ Logistic Regression  │
└─────────────────┘       └──────────────────────┘       └──────────────────────┘
```

---

## 📂 Repository Structure

```text
ai-data-classification/
├── app.py              # Main Streamlit web application & ML pipeline
├── requirements.txt    # Project Python dependencies
├── README.md           # Project documentation, guide & setup instructions
├── .gitignore          # Git exclusion rules (virtual environments, cache)
└── LICENSE             # MIT Open Source License
```

---

## 💻 Installation & Local Setup

Follow these simple steps to run the project on your local machine:

### 1. Clone or Download the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-data-classification.git
cd ai-data-classification
```

### 2. (Optional but Recommended) Create a Virtual Environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

After running the command, your default browser will open automatically to:
```text
http://localhost:8501
```

---

## 📸 Application Screenshots

| Section | Description |
| :--- | :--- |
| **Hero Dashboard** | Dark navy interface displaying top KPI metric cards and workflow stages |
| **Dataset Explorer** | Filterable interactive data table, descriptive statistics, and boxplot distributions |
| **Model Evaluation** | Confusion matrix heatmap, precision/recall/F1 metrics, and mathematical equations |
| **Live Flower Predictor** | Interactive sliders and probability breakdown for instant classification |

*(You can add your screenshots here after running the app: `![Dashboard](screenshot.png)`)*

---

## 📤 Step-by-Step GitHub Upload Instructions

To publish this project to your GitHub account:

### Step 1: Create a New Repository on GitHub
1. Log in to [GitHub](https://github.com/).
2. Click the **`+`** icon in the top right corner and select **New repository**.
3. Name your repository: `ai-data-classification`.
4. Choose **Public**.
5. Leave "Add a README file", ".gitignore", and "license" **unchecked** (we already created them).
6. Click **Create repository**.

### Step 2: Push Your Local Code using Git
Open your terminal inside the `ai-data-classification` folder and run:

```bash
# 1. Initialize git
git init

# 2. Stage all files
git add .

# 3. Create your first commit
git commit -m "feat: initial commit of AI Data Classification project"

# 4. Rename default branch to main
git branch -M main

# 5. Link to your remote GitHub repository
# Replace YOUR_USERNAME with your actual GitHub username:
git remote add origin https://github.com/YOUR_USERNAME/ai-data-classification.git

# 6. Push the code
git push -u origin main
```

---

## 🎓 Viva Voce / Evaluation Q&A

1. **Why Logistic Regression instead of complex neural networks?**
   - The Iris dataset has only 150 samples and 4 features. Logistic Regression provides high accuracy (~96.7%), trains in milliseconds, is resistant to overfitting on small data, and outputs explainable probabilities.
2. **Why is data leakage prevented?**
   - The train/test split (80:20) is performed before evaluation. The model only learns from the 120 training samples and is tested against 30 strictly unseen samples.
3. **What is the purpose of the Softmax function?**
   - Softmax converts raw log-odds (logits) into a probability distribution over the 3 species that sums to 1.0 (100%).

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
