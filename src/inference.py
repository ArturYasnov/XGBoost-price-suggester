from bottle import post, run, request, response, route
import pandas as pd
import joblib
import json
from os import path

import requests
from datetime import date, datetime
import time


car_categories = pd.read_csv(path.realpath(path.curdir) + '/resources/...csv')
car_brand_categories = pd.read_csv(path.realpath(path.curdir) + '/resources/car_brand_categories.csv')['index']

request_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
exrate_value = 2.40


def cur_proc(cur_id, date_):
    response_ = get_request(requests.get('http://www.nbrb.by/API/ExRates/Rates/' + cur_id + '?onDate=' + str(date_)))
    return response_['Cur_OfficialRate']


def get_request(request, n_try=0, max_n_try=7):
    try:
        result = request
        return result.json()
    except Exception:
        time.sleep(1.5)
        if n_try < max_n_try:
            n_try += 1
            get_request(request, n_try=n_try)


def prediction_data_clear_and_transform(data):
    data = data.replace({"..": '<1'}, 0)
    data = data.replace({"..": '>4'}, 5)
    data = data.replace({"..": '..'}, '1980')
    data[['..', '..']] = data[['..', '..']].astype('float64')
    data['..'] = data['..'].astype(str)

    data = data[data.cars_engine.isin(['..', '..'])]  # -2%
    data = data[data.subcategory == 2010]  # -0.2%
    data = data[data.condition == 1]  # -0.15%
    data = data[data.mileage >= 50]  # -1%
    data = data[data.regdate >= 2002]  # -0%

    data['..'] = data.cars_brand + ' ' + data.cars_model
    data = data.drop(['..'], axis=1)

    data = data[data.car.isin(car_categories)]
    data = data[['..', '..', '..', ..]]
    data = data.reset_index(drop=True)
    return data


def dummy_encode(data):
    data.cars_type = pd.Categorical(data.cars_type, categories=cars_type_categories)
    data.cars_gearbox = pd.Categorical(data.cars_gearbox, categories=cars_gearbox_categories)
    data.cars_engine = pd.Categorical(data.cars_engine, categories=cars_engine_categories)
    data.car = pd.Categorical(data.car, categories=car_categories)
    data.cars_brand = pd.Categorical(data.cars_brand, categories=car_brand_categories)

    data = pd.get_dummies(data)
    return data


@post('..') 
def predict_auto():
    try:
        o = json.load(request.body)

        keys = [..]
        for i in keys:
            if (i not in o) or (i is None) or (o[i] in ['', "null", []]) or (o[i] is None):
                return bottle.HTTPResponse(status=500, body='Not all fields are filled. Please fill in ' + i)

        data = pd.read_json(json.dumps([o]))
        data = data.dropna()
        
        for i in range(len(data)):
            data.loc[i, '..'] = cars_type_map[data.loc[i, '..']]
            data.loc[i, '..'] = cars_engine_map[data.loc[i, '..']]
            data.loc[i, '..'] = cars_gearbox_map[data.loc[i, '..']]

        data = prediction_data_clear_and_transform(data)
        data = dummy_encode(data)

        if len(data) == 0:
            return bottle.HTTPResponse(status=500, body='Сar does not fit the algorithm criteria')

        xgb_model = joblib.load(path.realpath(path.curdir) + '/models/predictor')

        predict = xgb_model.predict(data)

        response.content_type = 'application/json'

        global request_date, exrate_value
        if request_date != date.today():
            exrate_value = cur_proc('145', date.today())
            request_date = date.today()

        o['predicted_price_usd'] = str(predict[0])
        o['predicted_price_byn'] = str(predict[0] * exrate_value)
        return json.dumps(o)
    except Exception:
        return bottle.HTTPResponse(status=500, body='Something wrong with data')



@route('/health')
def health_check():
    return 'OK'

run(host='0.0.0.0', port=8080, reloader=True)
