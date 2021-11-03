import pandas as pd
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('xgbpredict')


dv, xgbmodel = pd.read_pickle('xgb_model.bin')

profile = {"age": 40,
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "oct",
    "day_of_week": "fri",
    "campaign": 3,
    "previous": 1,
    "emp.var.rate": 1,
    "cons.price.idx": 93.45,
    "euribor3m": 2.567,
    "nr.employed": 5567
    }
@app.route('/predict',methods =['POST'])

def predict():
    profile = request.get_json()

    X = dv.transform([profile])
    dval = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = xgbmodel.predict(dval)
    term = y_pred < 0.5
    result = {
        'Subscribe': bool(term),
        'Probability': float(y_pred)
    }
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host= 'localhost', port=9696)
