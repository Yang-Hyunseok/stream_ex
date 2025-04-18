import streamlit as st
import numpy as np
import pandas as pd
import datetime
import scipy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# RandomForestRegressor를 임포트합니다.
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform

@st.cache_data
def load_data():
    loae_df = pd.read_csv("SN_total.csv")
    load_df = load_df.set_index("시간")
    load_df.index = pd.DatetimeIndex(load_df.index)
    raw_df = load_df.copy()
    return raw_df

@st.cache_resource
def run_model(data, max_depth, n_estimators, eta, subsample, min_child_weight):
    data["로그 원수 탁도"] = np.log10(data["원수 탁도"])
    data["로그 응집제 주입률"] = np.log10(data["3단계 1계열 응집제 주입률"])
    X = data[['로그 원수 탁도', '원수 pH', '원수 알칼리도', '원수 전기전도도', '원수 수은', '3단계 원수 유입 유량', '3단계 침전지 체류시간']]
    y = data['로그 응집제 주입률']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    st.write('max_depth:', max_depth, ', n_estimators:', n_estimators, ', eta:', eta, ', subsample:', subsample, ', min_child_weight:', min_child_weight)

    random_search = {'max_depth': [max_depth],
                     'n_estimators': [n_estimators],
                     'eta': [eta],
                     'subsample': [subsample],
                     'min_child_weight': [min_child_weight]}
    regressor = XGBRegressor(random_state=2, n_jobs=-1)
    model = RandomizedSearchCV(estimator=regressor, param_distributions=random_search, n_iter=30)
    return model.fit(X_train, y_train), X_test, y_test

def prediction(model, X_train, y_train, X_test, y_test):
    yt_pred = model.predict(X_train)
    yts_pred = model.predict(X_test)

    mse_train = mean_squared_error(10**y_train, 10**yt_pred)
    mse_test = mean_squared_error(10**y_test, 10**yts_pred)
    st.write(f"학습 데이터 MSE: {mse_train}")
    st.write(f"테스트 데이터 MSE: {mse_test}")
    r2_train = r2_score(10**y_train, 10**yt_pred)
    r2_test = r2_score(10**y_test, 10**yts_pred)
    st.write(f"학습 데이터 R2: {r2_train}")
    st.write(f"테스트 데이터 R2: {r2_test}")

    return yt_pred, mse_train, r2_train, mse_test, r2_test

def prediction_plot(X_train, y_train, X_test, y_test, yt_pred, yts_pred, mse_train, r2_train, mse_test, r2_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(X_train["로그 원수 탁도"], y_train, s=3, label="학습 데이터 (실제)")
    ax.scatter(X_train["로그 원수 탁도"], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    ax.set_xlabel("로그 원수 탁도")
    ax.set_ylabel("로그 응집제 주입률")
    ax.set_title(
        rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
        fontsize=18)
    
    ax = axes[1]
    ax.scatter(X_test["로그 원수 탁도"], y_test, s=3, label="테스트 데이터 (실제)")
    ax.scatter(X_test["로그 원수 탁도"], yts_pred, s=3, label="테스트 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    ax.set_xlabel("로그 원수 탁도")
    ax.set_ylabel("로그 응집제 주입률")
    ax.set_title(
        rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
        fontsize=18)
    
def main():
    max_depth = st.slider("max_depth", 0, 20)
    n_estimators = st.slider("n_estimators", 0, 500)
    eta = st.slider("eta", 0, 1)
    subsample = st.slider("subsample", 0, 30)
    min_child_weight = st.slider("min_child_weight", 0, 30)

    data = load_data()
    model, X_train, y_train, X_test, y_test = run_model(data, max_depth, n_estimators, eta, subsample, min_child_weight)
    yt_pred, yts_pred, mse_train, r2_train, mse_test, r2_test = prediction(model, X_train, y_train, X_test, y_test)
    prediction_plot(X_train, y_train, X_test, y_test, yt_pred, yts_pred, mse_train, r2_train, mse_test, r2_test)

if __name__=="__name__":
    main()
