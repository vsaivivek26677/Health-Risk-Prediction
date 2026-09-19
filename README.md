# Health Condition Risk Prediction

An end-to-end classical machine learning pipeline that predicts an individual's health condition (`fit`, `at-risk`, or `unhealthy`) from lifestyle, physiological, and behavioural attributes such as sleep, heart rate, BMI, diet, activity level, and stress.

The entire pipeline is implemented with a classical ML stack (pandas, scikit-learn, scipy) — no deep learning.

## Dataset

- **Rows:** 690,088
- **Raw features:** 14 (numeric and categorical), plus a unique `id` column
- **Target:** `health_condition` with three classes — `at-risk` (~86%), `unhealthy` (~8%), `fit` (~6%)
- **Missingness:** up to ~12% in individual columns
- **Numeric features:** `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`
- **Categorical features:** `diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender`

The dataset is severely class-imbalanced, which shapes several downstream decisions (metric choice, class weighting, resampling considerations).

## Pipeline overview

1. **Data loading & integrity checks** — verifies `id` uniqueness, checks for duplicate rows, and profiles memory usage before and after dtype optimisation (categoricals + float32 downcasting).
2. **Data quality assessment** — missingness by column, IQR-based outlier screening, and range sanity checks against physiologically plausible bounds.
3. **Exploratory data analysis** — target distribution, univariate distributions for numeric and categorical features, feature-vs-target relationships, and a correlation heatmap of numeric features.
4. **Statistical hypothesis testing**
   - One-way ANOVA, Levene's test, and Kruskal-Wallis for numeric features, with eta-squared effect sizes
   - Chi-square tests of independence and Cramer's V for categorical features
   - Benjamini-Hochberg false discovery rate correction across all tests
   - A dedicated test for whether *missingness itself* is informative (associated with the target)
5. **Feature engineering** — physiologically motivated ratios (e.g. calories per step, water intake per BMI), a WHO BMI category, and missing-value indicators for columns flagged as informatively missing.
6. **Train / dev / held-out test split** — an 80/20 stratified split into `train_full` / `test_full` (test set touched exactly once, at the end), plus a stratified 120K-row `dev` subsample for fast model screening and hyperparameter search.
7. **Preprocessing pipeline** — median imputation + `RobustScaler` for numeric features, most-frequent imputation + one-hot encoding for categorical features, unified in a single `ColumnTransformer` shared across all candidate models.
8. **Baseline model comparison** — 3-fold stratified cross-validation over six class-weighted candidates:
   - `DummyClassifier` (naive floor)
   - `LogisticRegression`, `SGDClassifier`
   - `RandomForestClassifier`, `ExtraTreesClassifier`
   - `HistGradientBoostingClassifier`

   Models are ranked on macro-averaged F1, which weighs the minority classes (`fit`, `unhealthy`) equally with the dominant `at-risk` class.
9. **Hyperparameter tuning** — `RandomizedSearchCV` over the winning model's hyperparameters on the `dev` set, still optimising macro F1.
10. **Final evaluation** — the tuned model is refit on the full training split and evaluated once on the untouched held-out test set, reporting accuracy, balanced accuracy, macro/weighted F1, macro precision/recall, and weighted ROC-AUC, plus a confusion matrix (raw and row-normalised).
11. **Model interpretation** — native feature importances (for tree-based models) cross-checked against permutation importance and the effect sizes from the statistical testing stage.
12. **Error analysis** — per-class recall and a comparison of feature distributions between correctly and incorrectly classified samples, focused on the `fit` and `unhealthy` minority classes.
13. **Artifact persistence** — the fitted pipeline, label encoder, and key result tables are saved to disk for reuse without rerunning the notebook.

## Repository contents

| File | Description |
|---|---|
| `health_pipeline.ipynb` | The full analysis and modelling notebook |
| `train.csv` | Input dataset (not included) |

Running the notebook produces the following artifacts:

| Artifact | Description |
|---|---|
| `final_health_risk_pipeline.joblib` | Fitted preprocessing + model pipeline |
| `label_encoder.joblib` | Fitted label encoder for the target classes |
| `model_comparison_cv_results.csv` | Cross-validated metrics for all candidate models |
| `anova_kruskal_results.csv` | ANOVA / Kruskal-Wallis results for numeric features |
| `chi_square_results.csv` | Chi-square test results for categorical features |
| `final_test_metrics.csv` | Final held-out test set metrics |

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- joblib

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib
```

## Usage

1. Place `train.csv` in the same directory as the notebook (must contain the columns listed under Dataset above, including `id` and `health_condition`).
2. Launch Jupyter and run the notebook top to bottom:

   ```bash
   jupyter notebook health_pipeline.ipynb
   ```

3. Trained artifacts and result tables are written to the working directory once the notebook completes.

## Data

The raw dataset (`train.csv`) is not included in this repository. Supply your own dataset with the schema described above, or point `RAW_PATH` in the notebook to your data file's location.

## Key results

- Class imbalance is severe enough that plain accuracy is not a meaningful selection metric; models are selected on macro F1 and trained with class weighting throughout.
- No pair of raw numeric features is strongly correlated (|r| > 0.5), so multicollinearity is not a concern for the linear baselines.
- The best-performing model's feature importances (both native and permutation-based) broadly align with the features identified as having the largest effect sizes during statistical testing, which is a useful sanity check that the model is learning genuine signal.
- Minority-class recall (`fit`, `unhealthy`) remains the primary opportunity for further improvement.

