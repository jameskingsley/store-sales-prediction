# Retail Store Sales Forecasting Engine

An end-to-end Machine Learning pipeline designed to forecast retail store sales using advanced feature engineering, target power transformation, and 10-Fold Cross-Validation with CatBoost.

---

## Problem Statement & Goal

Accurately predicting product-level sales across multiple store formats and geographic tiers is critical for inventory optimization and demand planning. 

Retail sales data often exhibits extreme positive skewness, zeroes in product metrics, and non-linear interactions between product price and shelf placement. This pipeline addresses these real-world data distribution challenges to minimize Root Mean Squared Error (RMSE).

---

## Architecture & Pipeline Overview
Raw Data ─ Feature Engineering ─ Target Scaling (y^0.35) ─ 10-Fold CV Engine ─ Calibration & Inverse ─ Submission

### Core Pipeline Highlights:
1. **Domain Imputation & Cleaning:**
   * Replaced non-informative zero values in `shelf_visibility` with granular product-level means.
   * Standardized non-edible product classifications based on product prefixes (`NC`).

2. **Ratio & Interaction Feature Engineering:**
   * **Relative Price Ratios:** Price comparisons relative to category, store format, and location tier averages ($P_i / \bar{P}_{\text{category}}$).
   * **Relative Visibility Ratios:** Shelf space allocation relative to store and category baselines.
   * **Elasticity Interactions:** Non-linear transformations including $\log(1 + \text{price})$, $\text{price}^2$, and $\text{price} \times \text{visibility}$.

3. **Target Power Transformation ($y^{0.35}$):**
   * Compresses extreme high-sales outliers without losing relative ordering, making the objective landscape smoother for gradient boosting optimization.

4. **10-Fold Cross-Validation Architecture:**
   * Prevents target leakage during model evaluation.
   * Reduces prediction variance via out-of-fold model ensemble averaging.

5. **Post-Processing Calibration:**
   * Performs exact scale factor alignment on inverse-transformed predictions ($\hat{y}^{1 / 0.35}$) to eliminate global distribution bias before saving final outputs.

---

