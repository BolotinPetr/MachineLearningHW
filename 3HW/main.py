import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from mlxtend.regressor import StackingRegressor
from sklearn.svm import SVR
import math


def SMAPE(estimator, X_test, y_test):
    forecast = estimator.predict(X_test)
    non_zero = lambda x: x if x > 0 else 0.0
    func = np.vectorize(non_zero)
    forecast = func(forecast)
    actual = y_test
    # return mean_absolute_error(forecast, actual)
    return 100 * np.average((2 * abs(forecast - actual) / ((abs(forecast) + abs(actual)))))


def feature_transform(train_t, test_t):
    def transform(element):
        f = element.columns[34:]
        element = element.drop(f, axis=1)
        element = pd.get_dummies(element, columns=['year', 'week']).to_dense()
        element = pd.get_dummies(element, columns=['item_id']).to_dense()
        return element

    target_t = train_t['y']
    print target_t[:10]
    train_t = train_t.drop(['y'], axis=1)
    train_t = transform(train_t)
    test_t = transform(test_t)
    return train_t, target_t, test_t

def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d

train = pd.DataFrame.from_csv('train.tsv')
train = train.sample(frac=0.1, random_state=13)
test = pd.DataFrame.from_csv('test.tsv')
sample_submission = pd.read_csv("sample_submission.tsv")

train, target, test = feature_transform(train, test)
test = add_missing_dummy_columns(test, train.columns)
train = train[test.columns]
print train.columns

model_xgb = XGBRegressor(max_depth=9, n_estimators=369, learning_rate=0.01,
                         min_child_weight=1, gamma=0, colsample_bytree = 0.9,
                         subsample = 0.98, reg_alpha=0.1)
model_rf = RandomForestRegressor(n_estimators=200, max_features=0.26326530612244903, criterion='mse')
model_extra_tree = ExtraTreesRegressor(n_estimators=200, criterion='mse')
model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=43)
model_lr = LinearRegression()
svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
ridge = Ridge()
model_xgb2 = XGBRegressor(max_depth=10, n_estimators=100)
model_vote = VotingClassifier(estimators=[('xgb', model_xgb), ('rf', model_rf), ('gb', model_gb)])
sclf = StackingRegressor(regressors=[model_extra_tree, model_xgb2, model_rf],
                         meta_regressor=model_lr)


time_split = TimeSeriesSplit(n_splits=5)
print cross_val_score(sclf, X=train.as_matrix(), y=target.as_matrix(),
                      scoring=SMAPE, cv=time_split).mean()



sclf.fit(X=train, y=target)
preds = sclf.predict(test)
sample_submission['y'] = preds
print sample_submission[sample_submission['y'] < 0]
sample_submission['y'] = sample_submission['y'].map(lambda x: x if x > 0 else 0.0)
sample_submission.to_csv("my_submission_24_2.tsv", sep=',', index=False)