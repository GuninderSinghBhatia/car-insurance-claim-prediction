# 🚗 Car Insurance Claim Prediction

Predicting whether a car insurance policyholder will file a claim, using demographic, financial, vehicle and driving-history data. End-to-end ML project: data cleaning → EDA → feature encoding → modelling → handling class imbalance → threshold tuning → interpretation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![pandas](https://img.shields.io/badge/pandas-data-150458)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

📓 **Notebook:** [`car_insurance_claims.ipynb`](car_insurance_claims.ipynb)
🔗 **Kaggle version (with all outputs and charts):** [kaggle.com/code/guninder07/car-insurance-claims](https://www.kaggle.com/code/guninder07/car-insurance-claims)

---

## 📌 Problem

Insurers need to know which customers are likely to file a claim so they can price policies correctly and manage risk. This is a **binary classification** problem:

- **Target:** `CLAIM_FLAG` (1 = filed a claim, 0 = no claim)
- **Challenge:** the classes are imbalanced. Only ~27% of policyholders claim, so a model can look accurate while missing most real claims.

## 📊 Dataset

[Car Insurance Claim Data](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data) on Kaggle. It has **10,302 rows × 27 columns**.

| Group | Features |
|---|---|
| Demographics | `AGE`, `GENDER`, `MSTATUS`, `KIDSDRIV`, `HOMEKIDS`, `PARENT1`, `EDUCATION`, `OCCUPATION` |
| Financial | `INCOME`, `HOME_VAL`, `YOJ` (years on job) |
| Vehicle | `CAR_TYPE`, `CAR_USE`, `CAR_AGE`, `BLUEBOOK` (car value), `RED_CAR` |
| Driving history | `MVR_PTS` (licence points), `CLM_FREQ`, `OLDCLAIM`, `REVOKED`, `TIF` (time in force) |
| Location / commute | `URBANICITY`, `TRAVTIME` |
| Target | `CLAIM_FLAG` (`CLM_AMT` is the claim amount and is removed to avoid leakage) |

## 🧹 Data cleaning

| Issue | Fix |
|---|---|
| Money stored as text (`$67,349`) | Removed `$` and `,` and cast to float (`INCOME`, `HOME_VAL`, `BLUEBOOK`, `OLDCLAIM`, `CLM_AMT`) |
| `z_` prefix on categories (`z_No`, `z_SUV`, …) | Stripped from every text column |
| Identifier columns | Dropped `ID` and `BIRTH` (`AGE` already covers birth date) |
| Impossible values | Negative `CAR_AGE` (e.g. −3) set to missing |
| Missing values | `AGE` (7 rows) dropped. `YOJ`, `INCOME`, `HOME_VAL`, `CAR_AGE` filled with the median. `OCCUPATION` filled with `"Unknown"` |
| Target leakage | Dropped `CLM_AMT` (only known after a claim) |
| Useless feature | Dropped `RED_CAR` (no effect on claim rate) |

Final dataset: **10,295 rows**, **33 features** after encoding.

## 🔍 Key EDA findings

| Factor | Claim rate |
|---|---|
| Urban drivers vs rural | **31.7%** vs **6.5%** (≈5× higher) |
| Commercial vs private vehicle use | **34.6%** vs **22.0%** |
| Licence revoked vs not | **45.1%** vs **24.0%** |
| Red car vs other colour | 26.0% vs 26.9% (no difference) |
| Overall | 26.6% |

Claimants also have more licence points, more past claims and slightly lower income.

## ⚙️ Feature engineering

- **Binary mapping:** `PARENT1`, `MSTATUS`, `REVOKED`, `GENDER`, `CAR_USE`, `URBANICITY` → 0/1
- **Ordinal encoding:** `EDUCATION` (<High School = 0 … PhD = 4)
- **One-hot encoding:** `OCCUPATION`, `CAR_TYPE` (`drop_first=True`)
- **Split:** 80/20 stratified train/test split (`random_state=42`)
- **Scaling:** `StandardScaler` fitted on the training set only (used for logistic regression)

## 🤖 Models and results

Test set: 2,059 policyholders (548 claims).

| Model | Threshold | ROC-AUC | Accuracy | Claim recall | Claim precision | Claim F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.808 | 0.787 | 0.411 | 0.662 | 0.507 |
| Logistic Regression (balanced) | 0.50 | 0.807 | 0.718 | **0.745** | 0.481 | 0.584 |
| Random Forest (balanced) | 0.50 | **0.812** | 0.785 | 0.336 | 0.702 | 0.454 |
| **Random Forest (tuned threshold)** | **0.35** | **0.812** | 0.770 | 0.642 | 0.559 | **0.598** |

**What these numbers show**

1. The baseline logistic regression has 79% accuracy but **misses ~59% of claims**. Accuracy alone is misleading here.
2. `class_weight='balanced'` raises claim recall from 0.41 to **0.75**.
3. The Random Forest has the best ranking ability (ROC-AUC 0.812), but the default 0.5 cutoff is too strict for the minority class.
4. **Lowering the threshold to 0.35** (chosen by best F1) gives the best balance: it catches 64% of claims at 56% precision.

### Threshold trade-off (Random Forest)

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.50 | 0.336 | 0.702 | 0.454 |
| 0.45 | 0.453 | 0.667 | 0.539 |
| 0.40 | 0.542 | 0.614 | 0.576 |
| **0.35** | **0.642** | **0.559** | **0.598** |
| 0.30 | 0.719 | 0.497 | 0.588 |
| 0.25 | 0.794 | 0.446 | 0.571 |

An insurer that cares most about catching risky customers can move to 0.30 or 0.25 for higher recall.

## 🧠 What drives claims?

**Random Forest importance:** `BLUEBOOK` (car value), `INCOME`, `TRAVTIME`, `HOME_VAL`, `AGE`, `OLDCLAIM`, `URBANICITY`, `MVR_PTS`

**Logistic regression (signs show direction)**, top factors that increase claim risk:
`URBANICITY` (+0.95), SUV (+0.36), Sports car (+0.35), `REVOKED` (+0.31), `CLM_FREQ` (+0.28), `MVR_PTS` (+0.26), `TRAVTIME` (+0.24)

**In short:** urban drivers with a history of claims or licence points, a revoked licence, sporty or SUV vehicles, and commercial use are the highest-risk group.

## 🚀 How to run

```bash
git clone https://github.com/GuninderSinghBhatia/car-insurance-claim-prediction.git
cd car-insurance-claim-prediction
pip install -r requirements.txt
```

1. Download `car_insurance_claim.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data).
2. Change the `pd.read_csv(...)` path in the notebook to point to your copy.
3. Open `car_insurance_claims.ipynb` in Jupyter and run all cells.

You can also open the Kaggle link above and click **Copy & Edit**. The data is already attached there.

## 📁 Project structure

```
car-insurance-claim-prediction/
├── car_insurance_claims.ipynb   # full analysis and models
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## 🔮 Next steps

- Gradient boosting (XGBoost / LightGBM / CatBoost)
- Cross-validated hyper-parameter search
- Probability calibration for premium pricing
- A second model to predict claim **amount** (`CLM_AMT`) for claimants

## 👤 Author

**Guninder Singh Bhatia** · B.Tech CSE, Thapar Institute of Engineering and Technology
[GitHub](https://github.com/GuninderSinghBhatia) · [Kaggle](https://www.kaggle.com/guninder07)
