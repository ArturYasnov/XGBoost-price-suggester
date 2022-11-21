import sys
from os import path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
)

repository_path = path.expanduser("~")
sys.path.append(repository_path + "/modules")


# Reading data
data = pd.read_sql(
    """
select 
    ..
from ..
left join .. using (..)
where 
    ..
""",
    connection,
)


# Clear missed values
def missing_values_clean(data):
    data.active_stats_of_user = data.active_stats_of_user.fillna(0)
    data = data.dropna()
    return data


# Features transform
def feature_transform_and_selection(data):
    data = data.replace({"..": "<1"}, 0)
    data = data.replace({"..": ">4"}, 5)
    data[["..", ".."]] = data[["..", ".."]].astype("float64")

    data = data[data.param.isin(["..", ".."])]
    data = data[data.param >= 50]

    data = data[["..", "..", ".."]]
    data = data.reset_index(drop=True)
    return data


# Dummy encode
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
    sample.product_stats = pd.Categorical(
        sample.product_stats, categories=payment_categories
    )
    sample.product_engine = pd.Categorical(
        sample.product_engine, categories=product_subcategorie
    )

    sample = pd.get_dummies(sample)
    return sample


# Transforms
data = missing_values_clean(data)
data = feature_transforn_and_selection(data)
data = dummy_encode(data)


# Fit-predict base
X = data.drop(["price"], axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, shuffle=True)


params = {
    "objective": "binary:logistic",
    "max_depth": 8,
    "learning_rate": 0.03,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "min_child_weight": 4,
    "grow_policy": "lossguide",
    "tree_method": "hist",
    "n_estimators": 1500,
}
xgb_model = xgb.XGBRegressor(**params)
xgb_model.fit(X_train, y_train)
pred = xgb_model.predict(X_test)

# save model
joblib.dump(xgb_model, "xgb_model.pth")
