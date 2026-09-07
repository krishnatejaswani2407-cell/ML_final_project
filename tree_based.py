from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess import get_preprocessed_data


def run_tree_algorithm(algorithm):

    # Get only preprocessed data
    df = get_preprocessed_data()

    features = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "AgeGroup",
        "ExperienceGroup",
        "TenureGroup",
        "OverallSatisfaction"
    ]

    X = df[features]
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Select algorithm

    if algorithm == "decision_tree":

        model = DecisionTreeClassifier(
            random_state=42
        )

        model_name = "Decision Tree"

    elif algorithm == "random_forest":

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model_name = "Random Forest"

    elif algorithm == "extra_trees":

        model = ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        )

        model_name = "Extra Trees"

    elif algorithm == "gradient_boosting":

        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )

        model_name = "Gradient Boosting"

    elif algorithm == "adaboost":

        model = AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )

        model_name = "AdaBoost"

    elif algorithm == "xgboost":

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )

        model_name = "Extreme Gradient Boosting (XGBoost)"

    elif algorithm == "lightgbm":

        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )

        model_name = "LightGBM"

    else:

        return {
            "error": "Invalid algorithm selected"
        }

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return {
        "algorithm": model_name,
        "target": "Attrition",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }