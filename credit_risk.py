"""Clean the UCI Default of Credit Card Clients data and save credit_risk.csv."""
import pandas as pd

FILE = "default of credit card clients.xls"     # download from UCI or Kaggle (not in the repo)

df = pd.read_excel(FILE, header=1)              # row 1 holds the real column names
df = df.drop(columns="ID").rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"})

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicated rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape:", df.shape)

print("\nEDUCATION before:", df["EDUCATION"].value_counts().to_dict())
print("MARRIAGE before:", df["MARRIAGE"].value_counts().to_dict())
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # undocumented codes -> other
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                 # undocumented code -> other
print("EDUCATION after:", df["EDUCATION"].value_counts().to_dict())
print("MARRIAGE after:", df["MARRIAGE"].value_counts().to_dict())

print("\nPAY_1 values:", sorted(df["PAY_1"].unique().tolist()))
print("Default rate:", round(df["default"].mean(), 3))

df.to_csv("credit_risk.csv", index=False)
print("\nData saved to credit_risk.csv")
