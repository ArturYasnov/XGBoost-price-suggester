import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys

from sklearn.model_selection import cross_val_predict, cross_val_score, \
    GridSearchCV, RandomizedSearchCV, train_test_split, KFold, validation_curve
from import_credentials import import_credentials
from os import path

repository_path = path.expanduser('~')
sys.path.append(repository_path + '/modules')


# Reading data
data = pd.read_sql("""
select 
    ..
from ..
left join .. using (..)
where 
    ..
""", connection)


# Clear missed values
def missing_values_clean(data):
    data.active_ads_of_user = data.active_ads_of_user.fillna(0)
    data = data.dropna()
    return data


# Features transform
def feature_transform_and_selection(data):
    data = data.replace({"..": '<1'}, 0)
    data = data.replace({"..": '>4'}, 5)
    data = data.replace({"..": '..'}, '1980')
    data[['..', '..']] = data[['..', '..']].astype('float64')

    data = data[data.param.isin(['..', '..'])]
    data = data[data.param >= 50]

    data = data[['..', '..', ..]]
    data = data.reset_index(drop=True)
    return data


# Dummy encode
def dummy_encode(sample):
    cars_type_categories = ['sedan', 'minivan', 'off-road', 'hatchback', 'estate',
                        'minibus', 'coupe', 'convertible', 'van', 'pickup']
    cars_gearbox_categories = ['automatic', 'mechanics']
    cars_engine_categories = ['petrol', 'diesel']

    sample.cars_type = pd.Categorical(sample.cars_type, categories = cars_type_categories)
    sample.cars_gearbox = pd.Categorical(sample.cars_gearbox, categories = cars_gearbox_categories)
    sample.cars_engine = pd.Categorical(sample.cars_engine, categories = cars_engine_categories)

    sample = pd.get_dummies(sample)
    return sample


# Transforms
data = missing_values_clean(data)
data = feature_transforn_and_selection(data)
data = dummy_encode(data)


# Fit-predict base
X = data.drop(['price'], axis = 1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle = True)


params = {
    "objective": "binary:logistic",
    "max_depth":10,
    "learning_rate": 0.1,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    'min_child_weight': 4,
    "grow_policy": "lossguide",
    "tree_method": "hist",
    "n_estimators": 1000
}
xgb_model = xgb.XGBRegressor(**params)
xgb_model.fit(X_train, y_train)
pred = xgb_model.predict(X_test)

#save model
joblib.dump(xgb_model, 'base_xgb_model.pth')
