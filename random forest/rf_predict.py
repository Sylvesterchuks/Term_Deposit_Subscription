import flask
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction import DictVectorizer

# from sklearn.model_selection import KFold


dv,rf_model = pd.read_pickle('rf_model.bin')

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

app = Flask('term')

@app.route('/predict',methods =['POST'])

def predict():
    profile = request.get_json()

    X = dv.transform([profile])
    y_pred = rf_model.predict_proba(X)[0,1]
    term = y_pred < 0.5

    result = {
        'Subscribe to Term Deposit': float(y_pred),
        'Term Deposit': bool(term)
    }
    return jsonify(result)
#subscribe a term deposit

if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=9696)