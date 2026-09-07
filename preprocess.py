import pandas as pd


def get_preprocessed_data():

    df = pd.read_csv(
        r"D:\ML PPTS\Project\WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    # Remove unnecessary columns
    df = df.drop(
        columns=[
            "EmployeeCount",
            "Over18",
            "StandardHours",
            "EmployeeNumber"
        ],
        errors="ignore"
    )

    # Handle outliers using IQR
    Q1 = df["MonthlyIncome"].quantile(0.25)
    Q3 = df["MonthlyIncome"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["MonthlyIncome"] = df["MonthlyIncome"].clip(
        lower=lower,
        upper=upper
    )

    # Feature Engineering

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 40, 50, 100],
        labels=False
    )

    df["ExperienceGroup"] = pd.cut(
        df["TotalWorkingYears"],
        bins=[-1, 5, 10, 20, 100],
        labels=False
    )

    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 2, 5, 10, 100],
        labels=False
    )

    df["OverallSatisfaction"] = (
        df["JobSatisfaction"]
        + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"]
        + df["WorkLifeBalance"]
    )

    # Convert Attrition into 0 and 1
    df["Attrition"] = df["Attrition"].map({
        "No": 0,
        "Yes": 1
    })

    # Remove missing values
    df = df.dropna()

    return df