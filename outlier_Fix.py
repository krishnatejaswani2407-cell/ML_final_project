import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


file_path = r"D:\ML PPTS\Project\WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

feature = "MonthlyIncome"

print("Original statistics:")
print(df[feature].describe())

Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)

IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print("\nQ1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("lower_fence =", lower_fence)
print("upper_fence =", upper_fence)

outliers = df[
    (df[feature] < lower_fence) |
    (df[feature] > upper_fence)
]

print("\nNumber of Outliers:", len(outliers))

df["MonthlyIncome_Clipped"] = df[feature].clip(
    lower=lower_fence,
    upper=upper_fence
)

print("\nMinimum BEFORE Clipping:")
print(df[feature].min())

print("\nMinimum AFTER Clipping:")
print(df["MonthlyIncome_Clipped"].min())

print("\nMaximum BEFORE Clipping:")
print(df[feature].max())

print("\nMaximum AFTER Clipping:")
print(df["MonthlyIncome_Clipped"].max())

scaler = MinMaxScaler()

df["MonthlyIncome_Clipped"] = scaler.fit_transform(
    df[["MonthlyIncome_Clipped"]]
)

print("\nAfter Min-Max Scaling:")
print(df["MonthlyIncome_Clipped"].head())