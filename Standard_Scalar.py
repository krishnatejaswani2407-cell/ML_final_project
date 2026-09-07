import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

file_path = r"D:\ML PPTS\Project\WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

num_cols = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["Attrition"]
)

print("\nTraining Data before scaling:")
print(train_df[num_cols].head())

print("\nTesting Data before scaling:")
print(test_df[num_cols].head())

scaler = StandardScaler()

train_scaled = scaler.fit_transform(
    train_df[num_cols]
)

test_scaled = scaler.transform(
    test_df[num_cols]
)

print("\nStandard Scaled Training Data:")
print(train_scaled)

print("\nStandard Scaled Testing Data:")
print(test_scaled)