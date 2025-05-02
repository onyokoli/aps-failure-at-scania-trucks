# Scania Trucks APS Failure Prediction

## Overview

This project aims to predict failures in the Air Pressure System (APS) of Scania trucks using operational data. The primary goal is to minimize a specific cost metric associated with misclassifications:
* **Cost Type 1 (False Positive):** 10 units (unnecessary check)
* **Cost Type 2 (False Negative):** 500 units (missing a faulty truck)

The project involves data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, feature engineering, and model comparison based on the custom cost metric.

## Dataset

* **Source:** Scania CV AB (via IDA 2016 Industrial Challenge)
* **Training Set:** 60,000 examples (59,000 negative class, 1,000 positive class)
* **Test Set:** 16,000 examples
* **Features:** 170 anonymized operational data features (counters and histogram bins). Missing values denoted by 'na'.
* **Target:** 'class' (positive: APS failure, negative: other failure/no failure)
* **Challenge:** Highly imbalanced dataset and a cost-sensitive objective

*Data files (requires access/download from source):*
* `aps_failure_training_set.csv`
* `aps_failure_test_set.csv`
* `aps_failure_description.txt` (Contains detailed description and license)

## Methodology

1.  **Data Loading & EDA:** Loaded data using pandas, performed initial analysis (info, describe, class distribution visualization).
2.  **Preprocessing:**
    * Replaced 'na' strings with NaN.
    * Converted features to numeric types.
    * Imputed missing values using the median strategy.
    * Scaled features using `StandardScaler`.
3.  **Baseline Modeling:** Trained and evaluated several classification models on the preprocessed baseline features:
    * Logistic Regression (with `class_weight='balanced'`)
    * Support Vector Machine (SVM) (with `class_weight='balanced'`)
    * Random Forest (with `class_weight='balanced'`)
    * XGBoost (with `scale_pos_weight`)
    * LightGBM (with `scale_pos_weight`)
4.  **Feature Engineering (BONUS Task):** Identified 7 histogram variables based on the description file. Created 21 new features summarizing each histogram (sum, mean, std per row). Re-evaluated models with the combined feature set (original counters + new histogram summaries).
5.  **Model Comparison:** Compared models based primarily on the custom cost metric achieved on the test set.

## Key Results

* **Best Performing Model:** **Tuned Logistic Regression** using the **baseline feature set** (original 170 features, preprocessed).
* **Optimal Hyperparameters:** `C=0.001`, `penalty='l2'` (with `solver='liblinear'`, `class_weight='balanced'`).
* **Minimum Cost Achieved (Test Set):** **13,270**
* **Tuned Logistic Regression Performance (Test Set):**
    * Custom Cost: 13,270
    * AUC: 0.9900
    * Confusion Matrix: TN=15198, FP=427, FN=18, TP=357
    * Recall (Positive Class): 0.95
    * Precision (Positive Class): 0.46
* **Feature Engineering:** The attempt to use histogram summary statistics (sum, mean, std) did not improve performance based on the custom cost metric; costs increased for all models with these added features. The baseline feature set proved more effective.
* **Hyperparameter Tuning:** Tuning the Logistic Regression model significantly reduced the cost from the baseline of 17,680 to the final 13,270, primarily by further reducing the number of costly False Negatives (from 28 to 18).

## Setup & Usage

1.  **Environment:**
    * Python 3.x
    * Required Libraries:
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn jupyter
        # Optional (if running locally and want to test these models):
        # pip install xgboost lightgbm
        ```
2.  **Data:** Place the dataset files (`aps_failure_training_set.csv`, `aps_failure_test_set.csv`, `aps_failure_description.txt`) in the appropriate location relative to the notebook (e.g., in the same directory or a subdirectory).
3.  **Run Notebook:** Execute the cells in the Jupyter Notebook sequentially.

## Notebook

* [Link to main notebook file, e.g., APS_Failure_Analysis.ipynb](./APS_Failure_Baseline.ipynb) 

