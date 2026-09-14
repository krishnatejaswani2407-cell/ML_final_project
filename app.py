from flask import Flask, render_template, request

from load_data import get_dataset_info

from eda import (
    generate_all_charts,
    get_eda_summary
)

from preprocess import run_preprocessing

from linear_regression import (
    run_linear_regression
)

from logistic_regression import (
    run_logistic_regression
)

from tree_based import (
    run_tree_algorithm
)


app = Flask(__name__)


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    error = None
    data = None

    try:

        data = get_dataset_info()

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "index.html",
        data=data,
        error=error
    )


# =====================================================
# EDA
# =====================================================

@app.route("/eda")
def eda():

    error = None
    summary = None

    try:

        generate_all_charts()

        summary = get_eda_summary()

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "eda.html",
        summary=summary,
        error=error
    )


# =====================================================
# PREPROCESSING
# =====================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    results = None

    try:

        results = run_preprocessing()

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "preprocessing.html",
        results=results,
        error=error
    )


# =====================================================
# LINEAR REGRESSION PAGE
# =====================================================

@app.route("/linear-regression")
def linear_regression_page():

    return render_template(
        "linear_regression.html",
        results=None,
        selected_model=None,
        error=None
    )


# =====================================================
# LINEAR REGRESSION
# WITHOUT REGULARIZATION
# =====================================================

@app.route(
    "/linear-regression/without-regularization"
)
def linear_regression_without():

    error = None
    results = None

    try:

        results = run_linear_regression(
            "none"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        selected_model="none",
        error=error
    )


# =====================================================
# LINEAR REGRESSION
# RIDGE REGULARIZATION
# =====================================================

@app.route(
    "/linear-regression/with-regularization"
)
def linear_regression_with():

    error = None
    results = None

    try:

        results = run_linear_regression(
            "ridge"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        selected_model="ridge",
        error=error
    )


# =====================================================
# LINEAR REGRESSION
# LASSO REGULARIZATION
# =====================================================

@app.route(
    "/linear-regression/lasso"
)
def linear_regression_lasso():

    error = None
    results = None

    try:

        results = run_linear_regression(
            "lasso"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        selected_model="lasso",
        error=error
    )


# =====================================================
# LOGISTIC REGRESSION PAGE
# =====================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    return render_template(
        "logistic_regression.html",
        results=None,
        selected_model=None,
        error=None
    )


# =====================================================
# LOGISTIC REGRESSION
# WITHOUT REGULARIZATION
# =====================================================

@app.route(
    "/logistic-regression/without-regularization"
)
def logistic_regression_without():

    error = None
    results = None

    try:

        results = run_logistic_regression(
            "none"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        selected_model="none",
        error=error
    )


# =====================================================
# LOGISTIC REGRESSION
# L2 REGULARIZATION
# =====================================================

@app.route(
    "/logistic-regression/with-regularization"
)
def logistic_regression_with():

    error = None
    results = None

    try:

        results = run_logistic_regression(
            "l2"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        selected_model="l2",
        error=error
    )


# =====================================================
# LOGISTIC REGRESSION
# L1 / LASSO REGULARIZATION
# =====================================================

@app.route(
    "/logistic-regression/lasso"
)
def logistic_regression_lasso():

    error = None
    results = None

    try:

        results = run_logistic_regression(
            "lasso"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        selected_model="lasso",
        error=error
    )


# =====================================================
# TREE BASED MODELS PAGE
# =====================================================

@app.route("/tree-based")
def tree_based_page():

    return render_template(
        "tree_based.html",
        results=None,
        selected_model=None,
        error=None
    )


# =====================================================
# ID3
# =====================================================

@app.route(
    "/tree-based-models/id3"
)
def id3_model():

    try:

        results = run_tree_algorithm(
            "id3"
        )

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="id3",
            error=None
        )

    except Exception as e:

        return render_template(
            "tree_based.html",
            results=None,
            selected_model="id3",
            error=str(e)
        )


# =====================================================
# RANDOM FOREST
# =====================================================

@app.route(
    "/tree-based-models/random-forest"
)
def random_forest_model():

    try:

        results = run_tree_algorithm(
            "random_forest"
        )

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="random_forest",
            error=None
        )

    except Exception as e:

        return render_template(
            "tree_based.html",
            results=None,
            selected_model="random_forest",
            error=str(e)
        )


# =====================================================
# ADABOOST
# =====================================================

@app.route(
    "/tree-based-models/adaboost"
)
def adaboost_model():

    try:

        results = run_tree_algorithm(
            "adaboost"
        )

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="adaboost",
            error=None
        )

    except Exception as e:

        return render_template(
            "tree_based.html",
            results=None,
            selected_model="adaboost",
            error=str(e)
        )


# =====================================================
# GRADIENT BOOSTING
# =====================================================

@app.route(
    "/tree-based-models/gradient-boosting"
)
def gradient_boosting_model():

    try:

        results = run_tree_algorithm(
            "gradient_boosting"
        )

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="gradient_boosting",
            error=None
        )

    except Exception as e:

        return render_template(
            "tree_based.html",
            results=None,
            selected_model="gradient_boosting",
            error=str(e)
        )


# =====================================================
# XGBOOST
# =====================================================

@app.route(
    "/tree-based-models/xgboost"
)
def xgboost_model():

    try:

        results = run_tree_algorithm(
            "xgboost"
        )

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="xgboost",
            error=None
        )

    except Exception as e:

        return render_template(
            "tree_based.html",
            results=None,
            selected_model="xgboost",
            error=str(e)
        )


# =====================================================
# LIGHTGBM
# =====================================================

@app.route(
    "/tree-based-models/lightgbm"
)
def lightgbm_model():

    try:

        results = run_tree_algorithm(
            "lightgbm"
        )

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="lightgbm",
            error=None
        )

    except Exception as e:

        return render_template(
            "tree_based.html",
            results=None,
            selected_model="lightgbm",
            error=str(e)
        )


# =====================================================
# TREE BASED MODELS MAIN PAGE
# =====================================================

@app.route(
    "/tree-based-models"
)
def tree_based_models_page():

    return render_template(
        "tree_based.html",
        results=None,
        selected_model=None,
        error=None
    )


# =====================================================
# START APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )