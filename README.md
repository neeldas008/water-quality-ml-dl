# Water Quality Prediction — ML & Deep Learning

## Use Case

Access to clean water is a fundamental necessity for 600 million Indians
who face high-to-extreme water stress. The Central Pollution Control Board
(CPCB) monitors water quality across thousands of wells and sampling
locations across India, measuring chemical and physical parameters annually.

Manual interpretation of these measurements is slow and inconsistent.
Machine Learning can help authorities flag unsafe water faster, prioritize
treatment efforts, and allocate resources more effectively.

## What I Wanted to Achieve

This project has two prediction goals:

**1. Regression — Predict Water Quality Index (WQI)**
WQI is a numerical score that summarizes overall water quality.
Predicting WQI helps engineers quantify how safe or unsafe a water
source is on a continuous scale.

**2. Classification — Predict Water Quality Category**
Classify water into one of 5 categories:
- Excellent
- Good
- Poor
- Very Poor yet Drinkable
- Unsuitable for Drinking

This helps communities make immediate, actionable decisions about
whether water is safe to consume.

## My Approach

Rather than jumping straight to Deep Learning, I built one notebook
per algorithm — from the simplest baseline to a full PyTorch neural
network. This allowed a direct, apples-to-apples comparison of how
each algorithm thinks, where it fails, and what it teaches.

## Dataset

- Source   : Central Pollution Control Board (CPCB), India
- Samples  : 19,029 water quality measurements
- Years    : 2019, 2020, 2021, 2022
- Features : pH, EC, CO3, HCO3, Cl, SO4, NO3, TH, Ca, Mg, Na, K, F, TDS
- Location : Latitude, Longitude, State, District, Block, Village

## Notebooks

| Notebook | Algorithm | Key Concept |
|---|---|---|
| 00_EDA_and_Preprocessing | Data Cleaning | Encoding, Scaling, Train/Test Split |
| 01_Linear_Logistic_Regression | Linear / Logistic Regression | Baseline, Coefficients |
| 02_Decision_Tree | Decision Tree | Overfitting, Validation Curve |
| 03b_Bagging_DecisionTree | Bagging | Bootstrap Sampling, Variance Reduction |
| 03_Random_Forest | Random Forest | Feature Importance, Ensemble |
| 04_XGBoost | XGBoost | Boosting, Early Stopping, Learning Rate |
| 05_KNN | K-Nearest Neighbors | Distance Metrics, K Selection |
| 06_SVM | Support Vector Machine | Kernel Trick, C Parameter |
| 07_Deep_Learning_PyTorch | PyTorch MLP | Training Curves, BatchNorm, Dropout |
| 08_Model_Comparison | All Models | Final Comparison, Overfitting Analysis |

## Results Summary

### Classification (Accuracy & F1 Score)

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~0.96 | ~0.96 |
| Decision Tree | ~0.96 | ~0.96 |
| Bagging | ~0.96 | ~0.96 |
| Random Forest | ~0.97 | ~0.97 |
| XGBoost | ~0.97 | ~0.97 |
| KNN | ~0.95 | ~0.95 |
| SVM | ~0.93 | ~0.93 |
| Deep Learning | ~0.96 | ~0.96 |

*(Run notebooks to see exact values)*

## Key Findings

1. **Simple models compete** — Logistic Regression matched XGBoost at ~96%
   accuracy. Clean, well-scaled tabular data rewards simple models.

2. **Overfitting is visible** — Decision Tree scored Train=100%, Test=95.7%.
   The validation curve showed exactly where the tree started memorizing data.

3. **WQI is a derived formula** — Linear Regression scored R²=1.0 not because
   the model is perfect, but because WQI is mathematically calculated from the
   same chemical features used as inputs. Always understand your data before
   trusting your metrics.

4. **Feature importance** — EC (Electrical Conductivity), TDS, and Chlorides
   consistently ranked as the most important features across Random Forest
   and XGBoost — matching real-world domain knowledge.

5. **Deep Learning needs tuning** — A PyTorch MLP is competitive but does not
   clearly beat XGBoost on structured tabular data without extensive
   hyperparameter tuning.

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch
```

Run notebooks in order starting from `00_EDA_and_Preprocessing.ipynb`.
Each notebook saves results to a `.pkl` file used by subsequent notebooks.

## Project Structure

```
Assignment_Other_ML/
│
├── 00_EDA_and_Preprocessing.ipynb
├── 01_Linear_Logistic_Regression.ipynb
├── 02_Decision_Tree.ipynb
├── 03b_Bagging_DecisionTree.ipynb
├── 03_Random_Forest.ipynb
├── 04_XGBoost.ipynb
├── 05_KNN.ipynb
├── 06_SVM.ipynb
├── 07_Deep_Learning_PyTorch.ipynb
├── 08_Model_Comparison.ipynb
└── README.md
```

## Author
Neel Das
