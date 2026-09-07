from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import get_preprocessed_data


def run_logistic_regression(method):

    # ==========================================
    # GET PREPROCESSED DATA
    # ==========================================

    df = get_preprocessed_data()


    # ==========================================
    # FEATURES
    # ==========================================

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


    # ==========================================
    # INPUT AND TARGET
    # ==========================================

    X = df[features]

    # Predict employee Attrition
    y = df["Attrition"]


    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=42,

        stratify=y

    )


    # ==========================================
    # STANDARD SCALING
    # ==========================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)


    # ==========================================
    # SELECT ALGORITHM
    # ==========================================

    if method == "logistic":

        # Normal Logistic Regression
        # No regularisation

        model = LogisticRegression(
            penalty=None,
            max_iter=1000
        )

        model_name = (
            "Logistic Regression "
            "(Without Regularisation)"
        )


    elif method == "ridge":

        # L2 Regularisation

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000
        )

        model_name = (
            "Logistic Regression "
            "(L2 Regularisation)"
        )


    elif method == "lasso":

        # L1 Regularisation

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=1000
        )

        model_name = (
            "Logistic Regression "
            "(L1 Regularisation)"
        )


    else:

        return {
            "error": "Invalid logistic regression method selected"
        }


    # ==========================================
    # TRAIN MODEL
    # ==========================================

    model.fit(
        X_train,
        y_train
    )


    # ==========================================
    # PREDICTION
    # ==========================================

    y_pred = model.predict(
        X_test
    )


    # ==========================================
    # EVALUATION
    # ==========================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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


    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return {

        "model": model_name,

        "target": "Attrition",

        "accuracy": round(accuracy, 4),

        "precision": round(precision, 4),

        "recall": round(recall, 4),

        "f1": round(f1, 4)

    }