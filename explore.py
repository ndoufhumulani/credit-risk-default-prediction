"""Exploratory analysis: default rate by group and correlations."""
import matplotlib
matplotlib.use("Agg")                      # save charts without opening windows
import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("credit_risk.csv")
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
overall = df["default"].mean() * 100
print(f"Clients: {len(df)} | Default rate: {overall:.1f}%")


def rate_chart(groups, title, xlabel, filename, labels=None):
    """Bar chart of the default rate (%) for each group, with the overall rate as a line."""
    table = df.groupby(groups, observed=True)["default"].agg(["mean", "count"])
    table["rate"] = table["mean"] * 100
    if labels:
        table.index = [labels.get(i, i) for i in table.index]
    print(f"\n{title}")
    print(table[["count", "rate"]].round(1).to_string())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([str(i) for i in table.index], table["rate"])
    ax.axhline(overall, color="grey", ls="--", lw=1, label=f"Overall ({overall:.1f}%)")
    for i, v in enumerate(table["rate"]):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("Default rate (%)"); ax.set_xlabel(xlabel)
    ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"figures/{filename}"); plt.close(fig)


# 1. Most recent repayment status (the strongest predictor)
rate_chart("PAY_1", "1. Default rate by repayment status (September 2005)",
           "Repayment status (-2 no use, -1 paid in full, 0 minimum, 1-8 months late)",
           "01_default_by_pay1.png")

# 2. Credit limit bands (NT$)
df["limit_band"] = pd.cut(df["LIMIT_BAL"], [0, 50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000],
                          labels=["<=50k", "50-100k", "100-200k", "200-300k", "300-500k", ">500k"])
rate_chart("limit_band", "2. Default rate by credit limit", "Credit limit (NT$)", "02_default_by_limit.png")

# 3. Age bands
df["age_band"] = pd.cut(df["AGE"], [20, 25, 30, 35, 40, 50, 60, 80],
                        labels=["21-25", "26-30", "31-35", "36-40", "41-50", "51-60", "61+"])
rate_chart("age_band", "3. Default rate by age", "Age (years)", "03_default_by_age.png")

# 4-6. Demographic groups
rate_chart("SEX", "4. Default rate by sex", "Sex", "04_default_by_sex.png", {1: "Male", 2: "Female"})
rate_chart("EDUCATION", "5. Default rate by education", "Education", "05_default_by_education.png",
           {1: "Graduate", 2: "University", 3: "High school", 4: "Other"})
rate_chart("MARRIAGE", "6. Default rate by marital status", "Marital status", "06_default_by_marriage.png",
           {1: "Married", 2: "Single", 3: "Other"})

# 7. Correlation of the numerical variables
num = ["LIMIT_BAL", "AGE"] + [f"PAY_{i}" for i in range(1, 7)] + \
      [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] + ["default"]
corr = df[num].corr()
print("\nCorrelation with default (strongest first):")
print(corr["default"].drop("default").sort_values(key=abs, ascending=False).round(3).head(10).to_string())
print("\nCorrelation between BILL_AMT1 and BILL_AMT2:", round(corr.loc["BILL_AMT1", "BILL_AMT2"], 3))

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=90, fontsize=7)
ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
ax.set_title("7. Correlation between numerical variables")
fig.tight_layout(); fig.savefig("figures/07_correlation.png"); plt.close(fig)

print("\nSeven charts saved in the 'figures' folder.")
