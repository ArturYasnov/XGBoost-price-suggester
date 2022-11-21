import json
import time
from datetime import date, datetime
from os import path

import joblib
import pandas as pd
from bottle import post, request, response, route, run

product_categories = pd.read_csv(path.realpath(path.curdir) + "/resources/...csv")
product_brand_categories = pd.read_csv(
    path.realpath(path.curdir) + "/resources/product_brand_categories.csv"
)["index"]

request_date = datetime.strptime("2018-01-01", "%Y-%m-%d").date()
exrate_value = 2.0


def get_request(request, n_try=0, max_n_try=7):
    result = request
    return result.json()


def prediction_data_clear_and_transform(data):
    data = data.replace({"..": "<1"}, 0)
    data = data.replace({"..": ">4"}, 5)
    data[["..", ".."]] = data[["..", ".."]].astype("float64")
    data[".."] = data[".."].astype(str)

    data = data[data.product_subcategory.isin(["..", ".."])]
    data = data[data.subcategory == 10200]
    data = data[data.condition == 1]
    data = data[data.amount >= 50]

    data[".."] = data.product_brand + " " + data.product_model
    data = data.drop([".."], axis=1)

    data = data[data.product.isin(product_categories)]
    data = data[["..", "..", ".."]]
    data = data.reset_index(drop=True)
    return data


def dummy_encode(sample):
    product_type_categories = [
        "food",
        "clothes",
        "shoes",
        "real estate",
        "auto",
    ]
    payment_categories = ["one_pay", "installment plan"]
    product_subcategorie = ["cat_a", "cat_b"]

    sample.product_type = pd.Categorical(sample.product_type, categories=product_type_categories)
    sample.product_gearbox = pd.Categorical(
        sample.product_gearbox, categories=payment_categories
    )
    sample.product_stats = pd.Categorical(
        sample.product_stats, categories=product_subcategorie
    )

    sample = pd.get_dummies(sample)
    return sample



@post("..")
def predict_product_price():
    try:
        o = json.load(request.body)

        keys = [".."]
        for i in keys:
            if (
                (i not in o)
                or (i is None)
                or (o[i] in ["", "null", []])
                or (o[i] is None)
            ):
                return bottle.HTTPResponse(
                    status=500, body="Not all fields are filled. Please fill in " + i
                )

        data = pd.read_json(json.dumps([o]))
        data = data.dropna()

        for i in range(len(data)):
            data.loc[i, ".."] = product_type_map[data.loc[i, ".."]]
            data.loc[i, ".."] = product_stats_map[data.loc[i, ".."]]
            data.loc[i, ".."] = product_subcategory_map[data.loc[i, ".."]]

        data = prediction_data_clear_and_transform(data)
        data = dummy_encode(data)

        if len(data) == 0:
            return bottle.HTTPResponse(
                status=500, body="Product does not fit the algorithm criteria"
            )

        xgb_model = joblib.load(path.realpath(path.curdir) + "/models/predictor")

        predict = xgb_model.predict(data)

        response.content_type = "application/json"

        global request_date, exrate_value
        if request_date != date.today():
            exrate_value = cur_proc("225", date.today())
            request_date = date.today()

        o["predicted_price_usd"] = str(predict[0])
        o["predicted_price_currency"] = str(predict[0] * exrate_value)
        return json.dumps(o)
    except Exception:
        return bottle.HTTPResponse(status=500, body="Something wrong with the data")


@route("/health")
def health_check():
    return "OK"


run(host="0.0.0.0", port=8080, reloader=True)
