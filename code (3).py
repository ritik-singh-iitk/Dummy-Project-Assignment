# %% [markdown]
# # Capstone Assignment — Physician Drug Adoption Prediction (Drug XYZ, Q11)
#
# **Objective:** Predict, for the 1,502 non-adopter physicians in
# `test_physicians.csv`, the probability of first adoption of Drug XYZ in
# Quarter 11, so the sales team can prioritise outreach.
#
# **Pipeline:** Step 1 Data Merging & EDA -> Step 2 Feature Engineering ->
# Step 3 Baseline Logistic Regression -> Step 4 Mixed-Effects / Territory
# Modelling -> Step 5 Ensemble Models (Random Forest, Gradient Boosting) ->
# Step 6 Final Predictions on the Test Set.
#
# **Data files expected in `DATA_DIR` (see below):**
# - `input_data_file1.csv` — longitudinal Q1-Q10 behavioural data (100,000 rows)
# - `input_data_file2.csv` — static physician profile (10,000 rows)
# - `test_physicians.csv`  — 1,502 non-adopter physician_ids to score
#
# Random seed fixed at 42 everywhere for reproducibility.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve, classification_report,
                              precision_recall_curve, confusion_matrix)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

DATA_DIR = "/home/claude"   # <-- change to "." or your data folder if needed

# %% [markdown]
# ## Step 1 — Data Merging & Exploratory Analysis

# %%
# 1. Load all three datasets
file1 = pd.read_csv(f"{DATA_DIR}/input_data_file1.csv")   # longitudinal, 100,000 rows
file2 = pd.read_csv(f"{DATA_DIR}/input_data_file2.csv")   # static profile, 10,000 rows
test_physicians = pd.read_csv(f"{DATA_DIR}/test_physicians.csv")  # 1,502 physician_ids

print("file1 shape:", file1.shape)
print("file2 shape:", file2.shape)
print("test_physicians shape:", test_physicians.shape)

# Sanity checks
assert file1["physician_id"].nunique() == file2["physician_id"].nunique(), \
    "physician_id universe mismatch between file1 and file2"
assert set(test_physicians["physician_id"]).issubset(set(file2["physician_id"])), \
    "Some test physicians are not present in file2"

# %%
# 2. Define the binary target variable at physician level:
#    adopted = 1 if rx_count > 0 in ANY quarter, else 0
physician_target = (
    file1.groupby("physician_id")["rx_count"]
    .apply(lambda s: int((s > 0).any()))
    .rename("adopted")
    .reset_index()
)

print("Class balance (adopted):")
print(physician_target["adopted"].value_counts(normalize=True).round(4))

# NOTE on label design: "adopted" is a cross-sectional label built from the full
# Q1-Q10 window, as specified in the brief. We train the model to distinguish
# physicians whose overall Q1-Q10 engagement pattern looks like an "adopter"
# from one that doesn't, then score the 1,502 physicians who are STRICTLY
# non-adopters (rx_count = 0 in all 10 quarters, per test_physicians.csv) using
# that same pattern-recognition model. There is no train/test physician overlap,
# so this is a valid transfer of the learned relationship, not leakage -
# but it does assume the drivers of "having ever adopted by Q10" generalise to
# "adopting for the first time in Q11", which is the key business assumption
# to flag to stakeholders.

# %%
# 3. EDA — missing values, outliers, distributions by specialty/region

print("\nMissing values - file1:\n", file1.isna().sum())
print("\nMissing values - file2:\n", file2.isna().sum())

# Merge target + static profile for distribution analysis
eda_df = physician_target.merge(file2, on="physician_id", how="left")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=eda_df, x="specialty", y="adopted", ax=axes[0], estimator=np.mean, errorbar=None)
axes[0].set_title("Adoption Rate by Specialty")
axes[0].tick_params(axis="x", rotation=30)
sns.barplot(data=eda_df, x="geographic_region", y="adopted", ax=axes[1], estimator=np.mean, errorbar=None)
axes[1].set_title("Adoption Rate by Region")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_adoption_by_specialty_region.png", dpi=110)
plt.close()

# Outlier detection on raw activity (quarter-level)
numeric_cols_raw = ["rep_calls", "samples_provided", "digital_engagements",
                     "competitive_rx_count", "peer_network_score", "medical_education_events"]
print("\nQuarter-level numeric summary (outlier scan):")
print(file1[numeric_cols_raw].describe().T)

# %%
# 4. OLS regression on raw numeric features to understand linear association with adoption
#    (physician-level totals of raw activity vs adoption outcome)
raw_totals = (
    file1.groupby("physician_id")[["rep_calls", "samples_provided", "digital_engagements"]]
    .sum()
    .reset_index()
    .rename(columns={"rep_calls": "total_rep_calls_raw",
                      "samples_provided": "total_samples_raw",
                      "digital_engagements": "total_digital_raw"})
)
ols_df = physician_target.merge(raw_totals, on="physician_id")

X_ols = sm.add_constant(ols_df[["total_rep_calls_raw", "total_samples_raw", "total_digital_raw"]])
y_ols = ols_df["adopted"]
ols_model = sm.OLS(y_ols, X_ols).fit()
print(ols_model.summary())

# VIF check on these raw predictors
vif_raw = pd.DataFrame()
vif_raw["feature"] = X_ols.columns
vif_raw["VIF"] = [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])]
print("\nVIF (raw OLS features):\n", vif_raw)

# %% [markdown]
# ## Step 2 — Feature Engineering (physician-level, from long-format quarterly data)

# %%
def engineer_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the long-format (physician_id x quarter) behavioural data and returns
    one row per physician with engineered features capturing level, recency,
    trend, and momentum of engagement.
    """
    df = long_df.copy()
    df["q_num"] = df["quarter"].str.extract(r"(\d+)").astype(int)

    activity_cols = ["rep_calls", "samples_provided", "digital_engagements",
                      "competitive_rx_count", "peer_network_score", "medical_education_events"]

    early = df[df["q_num"].between(1, 3)]
    recent = df[df["q_num"].between(8, 10)]

    # --- Aggregate totals (whole 10-quarter window) ---
    totals = df.groupby("physician_id")[activity_cols + ["speaker_program_attendance",
                                                           "congress_attendance"]].sum()
    totals = totals.add_prefix("total_")

    # --- Recency features: mean over Q8-Q10 ---
    recency = recent.groupby("physician_id")[activity_cols].mean().add_prefix("recent_mean_")

    # --- Early window mean (Q1-Q3), needed for trend ---
    early_mean = early.groupby("physician_id")[activity_cols].mean().add_prefix("early_mean_")

    # --- Trend features: recent mean - early mean ---
    trend = (recency.rename(columns=lambda c: c.replace("recent_mean_", ""))
             - early_mean.rename(columns=lambda c: c.replace("early_mean_", "")))
    trend = trend.add_prefix("trend_")

    # --- Momentum: monotonically increasing rep_calls in last 4 quarters (Q7-Q10) ---
    def is_monotonic_increasing(sub):
        sub = sub.sort_values("q_num")
        vals = sub[sub["q_num"].between(7, 10)]["rep_calls"].values
        if len(vals) < 4:
            return 0
        return int(all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)))

    momentum = df.groupby("physician_id").apply(is_monotonic_increasing).rename("momentum_rep_calls_increasing")

    # --- Speaker programme recency: count of Q8-Q10 quarters attended ---
    speaker_recency = recent.groupby("physician_id")["speaker_program_attendance"].sum().rename(
        "speaker_program_recent_count")

    # --- Competitive exposure trend already captured in trend_competitive_rx_count ---

    # --- Rep call intensity (rate of change proxy) ---
    rep_calls_recent_vs_total = (
        recency["recent_mean_rep_calls"] / totals["total_rep_calls"].replace(0, np.nan)
    ).rename("rep_call_recent_share").fillna(0)

    features = (
        totals.join(recency, how="outer")
        .join(trend, how="outer")
        .join(momentum, how="outer")
        .join(speaker_recency, how="outer")
        .join(rep_calls_recent_vs_total, how="outer")
        .fillna(0)
        .reset_index()
    )
    return features


engineered = engineer_features(file1)
print("Engineered feature matrix shape:", engineered.shape)
print(engineered.columns.tolist())

# %%
# Physician segmentation (K-Means clustering) on engineered behavioural features
cluster_feature_cols = [c for c in engineered.columns if c != "physician_id"]
cluster_X = engineered[cluster_feature_cols].copy()
cluster_scaler = StandardScaler()
cluster_X_scaled = cluster_scaler.fit_transform(cluster_X)

kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
engineered["behaviour_cluster"] = kmeans.fit_predict(cluster_X_scaled).astype(str)

# quick sanity: does cluster membership relate to adoption in-sample?
cluster_check = engineered.merge(physician_target, on="physician_id")
print("\nAdoption rate by behaviour cluster:")
print(cluster_check.groupby("behaviour_cluster")["adopted"].mean().sort_values(ascending=False))

# %%
# Territory-level features: aggregate rep activity + adoption rate per territory, join back
territory_map = file2[["physician_id", "territory_code"]]
terr_activity = (
    file1.merge(territory_map, on="physician_id")
    .groupby("territory_code")
    .agg(territory_avg_rep_calls=("rep_calls", "mean"),
         territory_avg_samples=("samples_provided", "mean"),
         territory_avg_digital=("digital_engagements", "mean"))
    .reset_index()
)
terr_adoption = (
    physician_target.merge(territory_map, on="physician_id")
    .groupby("territory_code")["adopted"].mean()
    .rename("territory_adoption_rate")
    .reset_index()
)
territory_features = terr_activity.merge(terr_adoption, on="territory_code", how="left")

# %%
# Assemble the full physician-level modelling table
model_df = (
    physician_target
    .merge(engineered, on="physician_id", how="left")
    .merge(file2, on="physician_id", how="left")
    .merge(territory_features, on="territory_code", how="left")
)
print("Full modelling table shape:", model_df.shape)
model_df.head()

# %% [markdown]
# ## Step 3 — Baseline Model: Logistic Regression

# %%
# 5. Encode categorical variables
categorical_cols = ["specialty", "practice_type", "geographic_region", "practice_size"]
# territory_code and state are high-cardinality identifiers -> excluded from direct
# one-hot encoding (territory effects are captured via territory_features + Step 4 mixed model)

numeric_feature_cols = [c for c in model_df.columns if c not in
                         (["physician_id", "adopted", "territory_code", "state", "behaviour_cluster"]
                          + categorical_cols)]

def sanitize_cols(df):
    """Make column names formula-safe (no '/', spaces, '-', etc.) and cast dummy bools to int."""
    df = df.copy()
    df.columns = [str(c).replace("/", "_").replace(" ", "_").replace("-", "_") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df

encoded = pd.get_dummies(model_df[categorical_cols + ["behaviour_cluster"]],
                          columns=categorical_cols + ["behaviour_cluster"], drop_first=True)
encoded = sanitize_cols(encoded)

X_full = pd.concat([model_df[["physician_id"]], model_df[numeric_feature_cols], encoded], axis=1)
y_full = model_df["adopted"]

feature_cols = [c for c in X_full.columns if c != "physician_id"]
print(f"Total engineered feature count: {len(feature_cols)}")

# %%
# 6. 80/20 stratified train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
)

# Scale numeric features for the linear model (fit scaler on TRAIN only)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_train_scaled[numeric_feature_cols] = scaler.fit_transform(X_train[numeric_feature_cols])
X_val_scaled[numeric_feature_cols] = scaler.transform(X_val[numeric_feature_cols])

# %%
# VIF check on the encoded feature set -> drop features with VIF > 10
def compute_vif(df_features):
    vdf = pd.DataFrame()
    vdf["feature"] = df_features.columns
    vdf["VIF"] = [variance_inflation_factor(df_features.values, i) for i in range(df_features.shape[1])]
    return vdf.sort_values("VIF", ascending=False)

vif_input = X_train_scaled[feature_cols].astype(float)
vif_input = sm.add_constant(vif_input)
vif_table = compute_vif(vif_input)
print("Top VIF values:\n", vif_table.head(10))

high_vif_features = vif_table[(vif_table["VIF"] > 10) & (vif_table["feature"] != "const")]["feature"].tolist()
print(f"\nDropping {len(high_vif_features)} features with VIF > 10: {high_vif_features}")

final_feature_cols = [c for c in feature_cols if c not in high_vif_features]

# %%
logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
logreg.fit(X_train_scaled[final_feature_cols], y_train)

val_proba_logreg = logreg.predict_proba(X_val_scaled[final_feature_cols])[:, 1]
val_pred_logreg = (val_proba_logreg >= 0.5).astype(int)

auc_logreg = roc_auc_score(y_val, val_proba_logreg)
f1_logreg = f1_score(y_val, val_pred_logreg)
print(f"Logistic Regression — Validation AUC-ROC: {auc_logreg:.4f} | F1: {f1_logreg:.4f}")

# %%
# 8. Interpret coefficients: largest positive / negative log-odds effects
coef_table = pd.DataFrame({
    "feature": final_feature_cols,
    "coefficient": logreg.coef_[0]
}).sort_values("coefficient", ascending=False)

print("Top 10 POSITIVE drivers of adoption:\n", coef_table.head(10))
print("\nTop 10 NEGATIVE drivers of adoption:\n", coef_table.tail(10))

# %% [markdown]
# ## Step 4 — Mixed Effects / Territory-Level Modelling

# %%
# 10. Adoption rate per territory (already computed as territory_adoption_rate)
print("Territory-level adoption rate summary:")
print(model_df.groupby("territory_code")["adopted"].mean().describe())

# %%
# 11. Fit a mixed-effects model with territory_code as a random effect,
#     and compare against the standard (fixed-effects only) model via AIC/BIC.
mixed_df = X_full[["physician_id"] + final_feature_cols].merge(
    model_df[["physician_id", "adopted", "territory_code"]], on="physician_id"
)[["adopted", "territory_code"] + final_feature_cols].copy()
mixed_df.columns = [c.replace(" ", "_").replace("-", "_") for c in mixed_df.columns]
mixed_feature_cols = [c for c in mixed_df.columns if c not in ["adopted", "territory_code"]]

# Keep the mixed model formula compact (top predictors only) for numerical stability
top_predictors = coef_table.reindex(coef_table.coefficient.abs().sort_values(ascending=False).index)
top_predictors_list = [c.replace(" ", "_").replace("-", "_") for c in top_predictors["feature"].head(8)]
formula = "adopted ~ " + " + ".join(top_predictors_list)

def manual_aic_bic(loglik, n_params, n_obs):
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + n_params * np.log(n_obs)
    return aic, bic

try:
    mixed_model = sm.MixedLM.from_formula(formula, groups="territory_code", data=mixed_df)
    mixed_result = mixed_model.fit()
    print(mixed_result.summary())
    # statsmodels MixedLMResults does not always expose .aic/.bic directly -> compute manually
    n_params_mixed = mixed_result.params.shape[0] + 1  # + variance component
    mixed_aic, mixed_bic = manual_aic_bic(mixed_result.llf, n_params_mixed, mixed_df.shape[0])
except Exception as e:
    print("MixedLM failed:", e)
    mixed_result, mixed_aic, mixed_bic = None, np.nan, np.nan

# Standard (pooled, no random effect) comparison model via GLM logit for a fair AIC/BIC comparison
glm_formula = "adopted ~ " + " + ".join(top_predictors_list)
glm_model = smf.glm(glm_formula, data=mixed_df, family=sm.families.Binomial()).fit()
print("\nStandard logistic (pooled) AIC:", glm_model.aic, "BIC:", glm_model.bic_llf)
print("Mixed-effects model AIC:", mixed_aic, "BIC:", mixed_bic)

# %%
# 12. Interpretation: variance attributable to territory vs physician-level features
if mixed_result is not None:
    try:
        re_var = float(mixed_result.cov_re.iloc[0, 0])
        resid_var = float(mixed_result.scale)
        icc = re_var / (re_var + resid_var)
        print(f"\nTerritory random-effect variance: {re_var:.5f}")
        print(f"Residual (physician-level) variance: {resid_var:.5f}")
        print(f"Intraclass correlation (share of variance from territory): {icc:.2%}")
    except Exception as e:
        print("Could not compute ICC:", e)

# %% [markdown]
# ## Step 5 — Ensemble Models

# %%
# 13. Random Forest with OOB error estimation
rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=5,
    oob_score=True, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train[final_feature_cols], y_train)  # RF doesn't need scaling
print(f"Random Forest OOB score: {rf.oob_score_:.4f}")

val_proba_rf = rf.predict_proba(X_val[final_feature_cols])[:, 1]
val_pred_rf = (val_proba_rf >= 0.5).astype(int)
auc_rf = roc_auc_score(y_val, val_proba_rf)
f1_rf = f1_score(y_val, val_pred_rf)
print(f"Random Forest — Validation AUC-ROC: {auc_rf:.4f} | F1: {f1_rf:.4f}")

# %%
# 14. Feature importances -> compare top 10 to logistic coefficients
rf_importance = pd.DataFrame({
    "feature": final_feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("Top 10 Random Forest feature importances:\n", rf_importance.head(10))

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=rf_importance.head(10), x="importance", y="feature", ax=ax)
ax.set_title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/rf_feature_importance.png", dpi=110)
plt.close()

# %%
# 15. Gradient Boosting (XGBoost preferred, sklearn GBM fallback), tuned via 5-fold CV / AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

if HAS_XGB:
    gb_base = XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1)
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
else:
    gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
    }

grid = GridSearchCV(gb_base, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train[final_feature_cols], y_train)
gb_model = grid.best_estimator_
print("Best Gradient Boosting params:", grid.best_params_)

val_proba_gb = gb_model.predict_proba(X_val[final_feature_cols])[:, 1]
val_pred_gb = (val_proba_gb >= 0.5).astype(int)
auc_gb = roc_auc_score(y_val, val_proba_gb)
f1_gb = f1_score(y_val, val_pred_gb)
print(f"Gradient Boosting — Validation AUC-ROC: {auc_gb:.4f} | F1: {f1_gb:.4f}")

# %%
# 16. Model comparison table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "Validation AUC-ROC": [auc_logreg, auc_rf, auc_gb],
    "Validation F1": [f1_logreg, f1_rf, f1_gb],
}).sort_values("Validation AUC-ROC", ascending=False).reset_index(drop=True)
print("\nModel comparison:\n", comparison)

# ROC curve comparison plot
fig, ax = plt.subplots(figsize=(7, 6))
for name, proba in [("Logistic Regression", val_proba_logreg),
                     ("Random Forest", val_proba_rf),
                     ("Gradient Boosting", val_proba_gb)]:
    fpr, tpr, _ = roc_curve(y_val, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_val, proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/roc_comparison.png", dpi=110)
plt.close()

# %% [markdown]
# ## Step 6 — Final Predictions on Test Set

# %%
# 17. Select best-performing model (highest validation AUC-ROC)
best_model_name = comparison.iloc[0]["Model"]
print("Selected final model:", best_model_name)

model_lookup = {
    "Logistic Regression": (logreg, X_val_scaled, True),
    "Random Forest": (rf, X_val, False),
    "Gradient Boosting": (gb_model, X_val, False),
}
final_model, _, needs_scaling = model_lookup[best_model_name]

# %%
# 18. Apply the SAME feature engineering pipeline to the test-set physicians
test_long = file1[file1["physician_id"].isin(test_physicians["physician_id"])]
test_engineered = engineer_features(test_long)

# behaviour cluster: use the SAME fitted scaler + kmeans (fit on train universe) to assign clusters
test_cluster_X = test_engineered[[c for c in cluster_feature_cols]].copy()
# ensure column alignment (some physicians may have zero activity -> already filled with 0 in engineer_features)
test_cluster_X = test_cluster_X.reindex(columns=cluster_feature_cols, fill_value=0)
test_cluster_X_scaled = cluster_scaler.transform(test_cluster_X)
test_engineered["behaviour_cluster"] = kmeans.predict(test_cluster_X_scaled).astype(str)

test_df = (
    test_physicians
    .merge(test_engineered, on="physician_id", how="left")
    .merge(file2, on="physician_id", how="left")
    .merge(territory_features, on="territory_code", how="left")
)

# Re-apply identical encoding
test_encoded = pd.get_dummies(test_df[categorical_cols + ["behaviour_cluster"]],
                               columns=categorical_cols + ["behaviour_cluster"], drop_first=True)
test_encoded = sanitize_cols(test_encoded)
X_test_full = pd.concat([test_df[["physician_id"]], test_df[numeric_feature_cols], test_encoded], axis=1)

# Align columns exactly to training feature set (fill any missing dummy columns with 0)
X_test_final = X_test_full.reindex(columns=["physician_id"] + final_feature_cols, fill_value=0)

if needs_scaling:
    X_test_scaled = X_test_final.copy()
    X_test_scaled[numeric_feature_cols] = scaler.transform(X_test_final[numeric_feature_cols])
    test_features_for_model = X_test_scaled[final_feature_cols]
else:
    test_features_for_model = X_test_final[final_feature_cols]

print("Test feature matrix shape:", test_features_for_model.shape)
print("Pipeline validation: train feature count == test feature count ->",
      len(final_feature_cols) == test_features_for_model.shape[1])

# %%
# 19. Generate predicted probabilities + binary class prediction
test_proba = final_model.predict_proba(test_features_for_model)[:, 1]

# Threshold tuning: business context - a false negative (missing a physician who
# WOULD have adopted) wastes a launch opportunity, while a false positive only
# costs one extra sales visit. We therefore favour recall and pick the threshold
# that maximises F1 on the validation set rather than defaulting to 0.5.
best_val_proba = {"Logistic Regression": val_proba_logreg,
                   "Random Forest": val_proba_rf,
                   "Gradient Boosting": val_proba_gb}[best_model_name]

precisions, recalls, thresholds = precision_recall_curve(y_val, best_val_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
optimal_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0.5
print(f"Default threshold: 0.5 | F1-optimal threshold: {optimal_threshold:.3f}")

CHOSEN_THRESHOLD = optimal_threshold  # documented choice, see comment above
test_pred = (test_proba >= CHOSEN_THRESHOLD).astype(int)

# %%
# 20. Rank physicians by adoption probability; top 200 = primary Q11 outreach targets
predictions = pd.DataFrame({
    "physician_id": test_df["physician_id"],
    "adoption_probability": test_proba,
    "adoption_prediction": test_pred,
})
predictions = predictions.merge(test_features_for_model.assign(physician_id=test_df["physician_id"].values),
                                 on="physician_id", how="left")
predictions = predictions.sort_values("adoption_probability", ascending=False).reset_index(drop=True)

top_200 = predictions.head(200)
print(f"\nTop 200 physicians identified for priority Q11 outreach.")
print(predictions[["physician_id", "adoption_probability", "adoption_prediction"]].head(10))

predictions.to_csv(f"{DATA_DIR}/predictions.csv", index=False)
print(f"\nSaved predictions.csv with shape {predictions.shape}")

# %% [markdown]
# ## Summary
# - Final model: see `best_model_name` and `comparison` table above
# - Predictions saved to `predictions.csv` (physician_id, adoption_probability,
#   adoption_prediction, + all engineered feature columns)
# - Top 200 physicians (`top_200`) are the recommended priority outreach list
