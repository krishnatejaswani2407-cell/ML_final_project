from flask import Flask, render_template, request, jsonify

from load_data import get_dataset_info

from eda import (
    generate_all_charts,
    get_eda_summary
)

from linear_regression import run_linear_regression

from logistic_regression import run_logistic_regression

from tree_based import run_tree_algorithm


app = Flask(__name__)


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    data = get_dataset_info()

    return render_template(
        "index.html",
        data=data
    )


# =====================================================
# EDA
# =====================================================

@app.route("/eda")
def eda():

    generate_all_charts()

    summary = get_eda_summary()

    return render_template(
        "eda.html",
        summary=summary
    )


# =====================================================
# PREPROCESSING
# =====================================================

@app.route("/preprocessing")
def preprocessing():

    summary = {
        "rows": 1470,
        "columns": 35,
        "missing": 0,
        "duplicates": 0,

        "train_rows": 1029,
        "test_rows": 441,

        "train_percentage": 70,
        "test_percentage": 30
    }


    outlier = {

        "Q1": 3211,
        "Q3": 8834,
        "IQR": 5623,

        "lower_fence": -5223.5,
        "upper_fence": 17268.5,

        "outlier_count": 114,

        "original_min": 1009,
        "original_max": 19999,

        "clipped_min": 1009,
        "clipped_max": 17268.5
    }


    return render_template(

        "preprocessing.html",

        summary=summary,

        outlier=outlier,

        feature_preview=(
            "AgeGroup, ExperienceGroup, TenureGroup "
            "and OverallSatisfaction are created "
            "during preprocessing."
        ),

        standard_train=(
            "StandardScaler is applied to the training "
            "features before Linear and Logistic Regression."
        ),

        standard_test=(
            "The same fitted StandardScaler is applied "
            "to the testing features."
        )

    )


# =====================================================
# LINEAR REGRESSION PAGE
# =====================================================

@app.route("/linear-regression")
def linear_regression_page():

    return render_template(
        "linear_regression.html"
    )


# =====================================================
# RUN LINEAR REGRESSION
# =====================================================

@app.route("/run-linear-regression")
def run_linear_regression_model():

    try:

        method = request.args.get(
            "method",
            "without"
        )

        result = run_linear_regression(
            method
        )

        return jsonify(result)

    except Exception as e:

        print(
            "Linear Regression Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =====================================================
# LOGISTIC REGRESSION PAGE
# =====================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    return render_template(
        "logistic_regression.html"
    )


# =====================================================
# RUN LOGISTIC REGRESSION
# =====================================================

@app.route("/run-logistic-regression")
def run_logistic_regression_model():

    try:

        method = request.args.get(
            "method",
            "logistic"
        )

        result = run_logistic_regression(
            method
        )

        return jsonify(result)

    except Exception as e:

        print(
            "Logistic Regression Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =====================================================
# TREE BASED ALGORITHMS PAGE
# =====================================================

@app.route("/tree-based")
def tree_based_page():

    return render_template(
        "tree_based.html"
    )


# =====================================================
# RUN TREE BASED ALGORITHM
# =====================================================

@app.route("/run-tree-algorithm")
def run_tree_algorithm_model():

    try:

        algorithm = request.args.get(
            "algorithm",
            "decision_tree"
        )

        result = run_tree_algorithm(
            algorithm
        )

        return jsonify(result)

    except Exception as e:

        print(
            "Tree Based Algorithm Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =====================================================
# START APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )