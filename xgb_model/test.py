#!/usr/bin/env python
# coding: utf-8

import requests

profile = {"age": 40,
    "marital": "married",
    "education": "university.degree",
    "default": "yes",
    "housing": "yes",
    "loan": "no",
    "contact": "telephone",
    "month": "Aug",
    "day_of_week": "fri",
    "campaign": 3,
    "previous": 1,
    "emp.var.rate": 3,
    "cons.price.idx": 93.45,
    "euribor3m": 2.567,
    "nr.employed": 5567
    }

url = 'http://localhost:9696/predict'


response = requests.post(url,json=profile).json()
print(response)
