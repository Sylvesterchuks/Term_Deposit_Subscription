
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

from sklearn.model_selection import KFold

df = pd.read_csv('../bank-additional-full.csv',delimiter=';')

features = ['age','marital','education','default','housing','loan','contact','month','day_of_week','campaign','previous','emp.var.rate','cons.price.idx','euribor3m' , 'nr.employed', 'y']
new_df = df[features]



df_full_train, df_test = train_test_split(new_df,test_size=0.2,random_state=1)
df_train, df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = (df_train['y'] == 'no').astype('int').values
y_val = (df_val['y'] == 'no').astype('int').values
y_test = (df_test['y'] == 'no').astype('int').values


df_train = df_train.drop('y',axis=1)
df_val = df_val.drop('y',axis=1)
df_test = df_test.drop('y',axis=1)


df_train = df_train.replace('unknown',np.nan)
df_val = df_val.replace('unknown',np.nan)
df_test = df_test.replace('unknown',np.nan)


df_train = df_train.fillna(method='bfill',limit=20)

train_dict = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)


def dict_transform(df):
    df = df.fillna(method='bfill',limit=20)
    df_dict = df.to_dict(orient='records')
    new_df = dv.transform(df_dict)
    return new_df


X_val = dict_transform(df_val)


feature_names = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train,feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val,feature_names=feature_names)

xgb_params = {
    'eta':0.3,
    'max_depth':6,
    'min_child_weight' : 1,

    'objective' : 'binary:logistic',
    'nthread': 8,

    'seed':1,
    'verbosity' : 1
}

model = xgb.train(xgb_params,dtrain,num_boost_round=10)

y_pred = model.predict(dval)
auc = roc_auc_score(y_val,y_pred)
print(auc)

features.pop()
new_f = features
print(new_f)

def train(df_train,y_train):
    train_dict = df_train[new_f].to_dict(orient='records')
    dv = DictVectorizer(sparse = False)
    x_train = dv.fit_transform(train_dict)
    
    feature_names = dv.get_feature_names_out()
    dtrain = xgb.DMatrix(x_train, label=y_train,feature_names=feature_names)
    
    xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight' : 1,

    'objective' : 'binary:logistic',
    'eval_metric' : 'auc',
    
    'nthread': 8,
    'seed':1,
    'verbosity' : 1
    }
    
    xgbmodel = xgb.train(xgb_params,dtrain,num_boost_round=50)
    return dv, xgbmodel


def predict(df,dv,model):
    df_dict = df[new_f].to_dict(orient='records')
    X = dv.transform(df_dict)
    
    feature_names = dv.get_feature_names_out()
    dval = xgb.DMatrix(X, feature_names=feature_names)
    
    y_pred = model.predict(dval)
    return y_pred

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

fold = 0
xgb_scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = (df_train.y == 'no').astype('int').values
    y_val = (df_val.y == 'no').astype('int').values
    
    dv, xgbmodel = train(df_train,y_train)
    y_pred = predict(df_val,dv,xgbmodel)
    auc = roc_auc_score(y_val, y_pred)
    xgb_scores.append(auc)  
    fold += 1  
    print(f'fold-{fold}:{auc}')
print(np.mean(xgb_scores))

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train['y'] == 'no').astype('int').values

df_full_train = df_full_train.drop('y',axis=1)
df_full_train = df_full_train.replace('unknown',np.nan)
df_full_train = df_full_train.fillna(method='bfill',limit=20)
df_full_train = df_full_train.fillna(method='ffill',limit=20)


dv, xgbmodel = train(df_full_train,y_full_train)
y_pred = predict(df_test,dv,xgbmodel)
auc = roc_auc_score(y_test, y_pred)
print('Final model: ',auc)

pd.to_pickle((dv,xgbmodel),'xgb_model.bin')
print('model saved')
