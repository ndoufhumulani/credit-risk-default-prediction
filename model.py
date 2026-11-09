"""Models, evaluation, cost-based cut-off and fairness check for credit card default.

Reads credit_risk.csv (made by credit_risk.py) and writes:
  results/results.csv, results/odds_ratios.csv, results/fairness.csv, results/cost_by_cutoff.csv
  figures/08_roc_curves.png ... figures/12_feature_importance.png
Takes about 5-10 minutes, mostly for tuning XGBoost.
"""
import matplotlib
matplotlib.use("Agg")                      # save charts without opening windows
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             precision_score, recall_score, confusion_matrix)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

SEED = 42
COST_FN, COST_FP = 5, 1                    # missed defaulter costs 5, rejected good client costs 1
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

# =====================================================================
# 1. Data and new features
# =====================================================================
df = pd.read_csv("credit_risk.csv")
pay = [f"PAY_{i}" for i in range(1, 7)]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
paid = [f"PAY_AMT{i}" for i in range(1, 7)]

df["utilisation"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
df["months_late"] = (df[pay] > 0).sum(axis=1)
ratios = np.where(df[bill].to_numpy() > 0,
                  df[paid].to_numpy() / np.maximum(df[bill].to_numpy(), 1), 1.0)  # nothing owed -> 1
df["avg_payment_ratio"] = ratios.mean(axis=1)

protected = ["SEX", "AGE", "MARRIAGE"]
features = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + pay + bill + paid + \
           ["utilisation", "months_late", "avg_payment_ratio"]
X, y = df[features], df["default"]

# =====================================================================
# 2. Stratified 75/25 split; cap the ratio features using training percentiles only
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)
X_train, X_test = X_train.copy(), X_test.copy()
for col in ["utilisation", "avg_payment_ratio"]:
    lo, hi = X_train[col].quantile([0.01, 0.99])
    X_train[col] = X_train[col].clip(lo, hi)
    X_test[col] = X_test[col].clip(lo, hi)

w = (y_train == 0).sum() / (y_train == 1).sum()     # class weight for defaulters
print(f"Training: {len(X_train)} clients | Test: {len(X_test)} clients | Defaulter weight: {w:.2f}")


def correct(p, weight=w):
    """Undo the effect of class weighting on probabilities (divide the odds by the weight)."""
    odds = p / (1 - p + 1e-12)
    odds = odds / weight
    return odds / (1 + odds)


# =====================================================================
# 3. Models and 5-fold cross-validated tuning (AUC)
# =====================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
searches = {
    "Logistic regression": GridSearchCV(
        Pipeline([("scale", StandardScaler()),
                  ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        {"model__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=cv, n_jobs=-1),
    "Random forest": GridSearchCV(
        RandomForestClassifier(max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"n_estimators": [300, 500], "min_samples_leaf": [1, 5, 10]}, scoring="roc_auc", cv=cv),
    "XGBoost": GridSearchCV(
        XGBClassifier(scale_pos_weight=w, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 4, 5, 6], "n_estimators": [200, 400, 800]},
        scoring="roc_auc", cv=cv),
}

models, probs = {}, {}
for name, search in searches.items():
    print(f"Tuning {name} ...")
    search.fit(X_train, y_train)
    models[name] = search.best_estimator_
    probs[name] = correct(search.predict_proba(X_test)[:, 1])
    print(f"   best settings: {search.best_params_} | cross-validated AUC: {search.best_score_:.4f}")

# =====================================================================
# 4. Evaluation on the test set
# =====================================================================
t_star = COST_FP / (COST_FP + COST_FN)               # Elkan's cost-minimising cut-off
rows = []
for name, p in probs.items():
    auc = roc_auc_score(y_test, p)
    fpr, tpr, _ = roc_curve(y_test, p)
    pred = (p >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({"Model": name, "AUC": auc, "Gini": 2 * auc - 1, "KS": np.max(tpr - fpr),
                 "Brier": brier_score_loss(y_test, p),
                 "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                 "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                 "Cost": COST_FN * fn + COST_FP * fp})
results = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
results.to_csv("results/results.csv", index=False)
pd.set_option("display.width", 200)
print(f"\nTest-set results (cut-off t* = {t_star:.3f}, cost ratio {COST_FN}:{COST_FP}):")
print(results.round(4).to_string(index=False))

best = results.loc[0, "Model"]
gap = results.loc[0, "AUC"] - results.loc[results["Model"] == "Logistic regression", "AUC"].iloc[0]
print(f"\nHighest AUC: {best}. Gain over logistic regression: {gap:.4f}")
print("Recommendation:", "logistic regression (gain under 0.02, easier to explain)"
      if gap < 0.02 else f"{best} (gain of 0.02 or more)")

# =====================================================================
# 5. Charts: ROC, calibration, cost curve
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("8. ROC curves (test set)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/08_roc_curves.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
for name, p in probs.items():
    frac, mean_p = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, marker="o", ms=3, label=name)
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed default rate")
ax.set_title("9. Calibration (test set, 10 groups)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/09_calibration.png"); plt.close(fig)

cuts = np.round(np.arange(0.05, 0.951, 0.01), 2)
p_best = probs[best]
cost_table = pd.DataFrame({"cut_off": cuts})
fig, ax = plt.subplots(figsize=(7, 3.8))
for ratio in (3, 5, 10):
    cost = [ratio * ((p_best < c) & (y_test == 1)).sum() + ((p_best >= c) & (y_test == 0)).sum() for c in cuts]
    cost_table[f"cost_{ratio}to1"] = cost
    c_min = cuts[int(np.argmin(cost))]
    ax.plot(cuts, np.array(cost) / 1000, label=f"{ratio}:1 (lowest at {c_min:.2f}; formula {1/(1+ratio):.2f})")
    print(f"Cost ratio {ratio}:1 -> formula cut-off {1/(1+ratio):.3f}, lowest cost on test set at {c_min:.2f}")
cost_table.to_csv("results/cost_by_cutoff.csv", index=False)
ax.set_xlabel("Cut-off probability"); ax.set_ylabel("Total cost (thousands of units)")
ax.set_title(f"10. Total cost against cut-off ({best})"); ax.legend(frameon=False, fontsize=8, title="Cost ratio")
fig.tight_layout(); fig.savefig("figures/10_cost_curve.png"); plt.close(fig)

# =====================================================================
# 6. Interpretation: odds ratios and feature importance
# =====================================================================
lr = models["Logistic regression"].named_steps["model"]
odds = pd.Series(np.exp(lr.coef_[0]), index=features, name="odds_ratio")
odds.sort_values(ascending=False).to_csv("results/odds_ratios.csv")
print("\nLogistic regression odds ratios (per standard deviation), furthest from 1 first:")
print(odds.iloc[np.argsort(-np.abs(np.log(odds.to_numpy())))].head(10).round(3).to_string())

top = odds.iloc[np.argsort(-np.abs(np.log(odds.to_numpy())))].head(10).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(top.index, top, color=["tab:red" if v > 1 else "tab:blue" for v in top])
ax.axvline(1, color="grey", lw=1)
ax.set_xlabel("Odds ratio per standard deviation"); ax.set_title("11. Logistic regression: largest odds ratios")
fig.tight_layout(); fig.savefig("figures/11_odds_ratios.png"); plt.close(fig)

tree_name = "XGBoost" if results.set_index("Model").loc["XGBoost", "AUC"] >= \
            results.set_index("Model").loc["Random forest", "AUC"] else "Random forest"
imp = pd.Series(models[tree_name].feature_importances_, index=features).sort_values().tail(10)
print(f"\n{tree_name} feature importance (top 10):")
print((imp[::-1] * 100).round(1).to_string())
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(imp.index, imp * 100)
ax.set_xlabel("Importance (%)"); ax.set_title(f"12. {tree_name}: feature importance")
fig.tight_layout(); fig.savefig("figures/12_feature_importance.png"); plt.close(fig)

# =====================================================================
# 7. Fairness: retrain the best model without sex, age and marital status
# =====================================================================
reduced = [f for f in features if f not in protected]
model_fair = searches[best].best_estimator_.__class__(**searches[best].best_estimator_.get_params())
model_fair.fit(X_train[reduced], y_train)
p_fair = correct(model_fair.predict_proba(X_test[reduced])[:, 1])
auc_fair = roc_auc_score(y_test, p_fair)
print(f"\nFairness check ({best}):")
print(f"   AUC with sex, age and marital status: {results.loc[0, 'AUC']:.4f}")
print(f"   AUC without them:                      {auc_fair:.4f} (change {auc_fair - results.loc[0, 'AUC']:+.4f})")

groups = pd.DataFrame({
    "Sex": X_test["SEX"].map({1: "Male", 2: "Female"}),
    "Age group": pd.cut(X_test["AGE"], [20, 30, 40, 50, 100], labels=["21-30", "31-40", "41-50", "51+"]),
    "default": y_test, "approved": p_fair < t_star})
fair_rows = []
for col in ["Sex", "Age group"]:
    for g, part in groups.groupby(col, observed=True):
        defaulters = part[part["default"] == 1]
        fair_rows.append({"Characteristic": col, "Group": g, "Clients": len(part),
                          "Default rate": part["default"].mean(),
                          "Approval rate": part["approved"].mean(),
                          "Recall": (~defaulters["approved"]).mean()})
fairness = pd.DataFrame(fair_rows)
fairness.to_csv("results/fairness.csv", index=False)
print(fairness.round(3).to_string(index=False))

print("\nSaved results/ (4 tables) and figures 08 to 12.")
