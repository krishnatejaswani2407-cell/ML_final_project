from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from math import sqrt

from preprocess import get_preprocessed_data


def run_linear_regression(method):

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

    # Linear Regression predicts Monthly Income
    y = df["MonthlyIncome"]


    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=42

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

    if method == "linear":

        model = LinearRegression()

        model_name = (
            "Linear Regression "
            "(Without Regularisation)"
        )


    elif method == "ridge":

        model = Ridge(
            alpha=1.0
        )

        model_name = (
            "Ridge Regression "
            "(L2 Regularisation)"
        )


    elif method == "lasso":

        model = Lasso(
            alpha=1.0,
            max_iter=10000
        )

        model_name = (
            "Lasso Regression "
            "(L1 Regularisation)"
        )


    else:

        return {
            "error": "Invalid regression method selected"
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

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )


    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return {

        "model": model_name,

        "target": "MonthlyIncome",

        "mse": round(mse, 2),

        "rmse": round(rmse, 2),

        "r2": round(r2, 4)

    }