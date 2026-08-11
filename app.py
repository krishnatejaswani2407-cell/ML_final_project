from flask import Flask, render_template

from load_data import get_dataset_info

from eda import (
    generate_all_charts,
    get_eda_summary
)


app = Flask(__name__)


# =========================================================
# Data Loading
# =========================================================

@app.route("/")
def home():

    data = get_dataset_info()

    return render_template(
        "index.html",
        data=data
    )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    generate_all_charts()

    summary = get_eda_summary()

    return render_template(
        "eda.html",
        summary=summary
    )


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )