from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

app = FastAPI()

@app.get("/check")
def health():
    return "It's still on!"

def load_data(file):
    data = pd.read_csv(StringIO(file))
    return data

async def decode_data(file):
    contents = await file.read()
    decoded_data = load_data(contents.decode('utf-8'))
    return decoded_data

@app.post("/load_data")
async def load_data_endpoint(file: UploadFile = File(...)):
    data = await decode_data(file)
    return {"columns": data.columns.tolist(), "shape": data.shape}

@app.post("/dataset_info")
async def dataset_info_endpoint(file: UploadFile = File(...)):
    data = await decode_data(file)
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    return {"info": s}

@app.post("/stat_analysis")
async def stat_analysis_endpoint(file: UploadFile = File(...)):
    data = await decode_data(file)
    return data.describe().T.to_dict()

@app.post("/non_numeric_cols")
async def non_numeric_columns_endpoint(file: UploadFile = File(...)):
    data = await decode_data(file)
    non_numeric_col = data.select_dtypes(include=['object']).columns.tolist()
    return {"non_numeric_columns": non_numeric_col}

@app.post("/unique_values")
async def unique_values_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    non_numeric_col = data.select_dtypes(include=['object']).columns.tolist()
    unique_vals = {col: data[col].unique().tolist() for col in non_numeric_col}
    return unique_vals

@app.post("/correlation_matrix")
async def correlation_matrix_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    cor_matrix = data.corr()
    return cor_matrix.to_dict()

@app.post("/high_covariance_features")
async def high_covariance_features_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    core_num = data.select_dtypes(include=[np.number]).dropna(axis='columns')
    vif_data = pd.DataFrame()
    vif_data['Feature'] = core_num.columns
    vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]
    return vif_data.to_dict()

@app.post("/feature_importance")
async def feature_importance_endpoint(target, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    df = data.copy()
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    cor_matrix = df.corr()
    feature_imp = cor_matrix[target].drop(target).abs().sort_values(ascending=False)
    imp_df = pd.DataFrame({
        'Feature': feature_imp.index,
        'Imp Score': feature_imp.values
    })
    return imp_df.to_dict()

@app.post("/group_features")
async def group_features_endpoint(target, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    cor_matrix = data.corr()
    target_cor = cor_matrix[target].drop(target)
    groups = target_cor[target_cor.abs() > 0.5].index.tolist()
    return {"groups": groups}

@app.post("/evaluate_group_performance")
async def evaluate_group_performance_endpoint(target, groups: List[str], file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    elements = groups[0]
    group_list = elements.split(",")

    missing_columns = [col for col in group_list if col not in data.columns]
    if missing_columns:
        return {"error": f"Columns {missing_columns} are not in the DataFrame"}

    X = data[group_list]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {"accuracy": accuracy}

@app.post("/binning")
async def binning_endpoint(column, bins: int, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    data[f'{column}_binned'] = pd.cut(data[column], bins=bins)
    data[f'{column}_binned'] = data[f'{column}_binned'].apply(lambda x: x.mid)
    return data[[column, f'{column}_binned']].head().to_dict()

@app.post("/sampling")
async def sampling_endpoint(sample_size: int, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    sample = data.sample(n=sample_size, random_state=42)
    return sample.head().to_dict()

@app.post("/feature_selection")
async def feature_selection_endpoint(target, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    X = data.drop(columns=[target])
    y = data[target]
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
        elif pd.api.types.is_interval_dtype(X[column]):
            X[column] = X[column].apply(lambda x: x.mid)
    selector = SelectKBest(score_func=f_classif, k=5)
    selected_features = selector.fit_transform(X, y)
    return {"selected_features": selected_features.tolist()}

@app.post("/handle_imbalanced_data")
async def handle_imbalanced_data_endpoint(target, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    majority_class = data[data[target] == data[target].mode()[0]]
    minority_class = data[data[target] != data[target].mode()[0]]
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=1)
    balanced_data = pd.concat([majority_class, minority_upsampled])
    return balanced_data.to_dict()

def execute_model(model_name, X_train, X_test, y_train, y_test):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor()
    elif model_name == "Support Vector Classifier":
        model = SVC()
    elif model_name == "Support Vector Regressor":
        model = SVR()
    elif model_name == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
    elif model_name == "XGBoost Classifier":
        model = XGBClassifier()
    elif model_name == "XGBoost Regressor":
        model = XGBRegressor()
    elif model_name == "LightGBM Classifier":
        model = LGBMClassifier()
    elif model_name == "LightGBM Regressor":
        model = LGBMRegressor()
    elif model_name == "CatBoost Classifier":
        model = CatBoostClassifier(verbose=0)
    elif model_name == "CatBoost Regressor":
        model = CatBoostRegressor(verbose=0)
    elif model_name == "K-Nearest Neighbors Classifier":
        model = KNeighborsClassifier()
    elif model_name == "K-Nearest Neighbors Regressor":
        model = KNeighborsRegressor()
    elif model_name == "Neural Network Classifier (MLPClassifier)":
        model = MLPClassifier(max_iter=1000)
    elif model_name == "Neural Network Regressor (MLPRegressor)":
        model = MLPRegressor(max_iter=1000)
    else:
        return "Model not recognized."

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if model_name in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor", "K-Nearest Neighbors Regressor", "Neural Network Regressor (MLPRegressor)"]:
        mse = mean_squared_error(y_test, predictions)
        return {"Mean Squared Error": mse}
    else:
        accuracy = accuracy_score(y_test, predictions)
        return {"Accuracy": accuracy}

@app.post("/execute_model")
async def execute_model_endpoint(target, model, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    X = data.drop(columns=[target])
    y = data[target]
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metric = execute_model(model, X_train, X_test, y_train, y_test)
    return {"status": "Model executed", "Performance": metric}

def hyperparameter_tuning(model_name, X_train, y_train):
    param_grid = {}
    if model_name == "Linear Regression":
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
    elif model_name == "Logistic Regression":
        model = LogisticRegression()
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    elif model_name == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor()
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    elif model_name == "Support Vector Classifier":
        model = SVC()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == "Support Vector Regressor":
        model = SVR()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "XGBoost Classifier":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "XGBoost Regressor":
        model = XGBRegressor()
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "LightGBM Classifier":
        model = LGBMClassifier()
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "LightGBM Regressor":
        model = LGBMRegressor()
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "CatBoost Classifier":
        model = CatBoostClassifier(verbose=0)
        param_grid = {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "CatBoost Regressor":
        model = CatBoostRegressor(verbose=0)
        param_grid = {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == "K-Nearest Neighbors Classifier":
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif model_name == "K-Nearest Neighbors Regressor":
        model = KNeighborsRegressor()
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif model_name == "Neural Network Classifier (MLPClassifier)":
        model = MLPClassifier(max_iter=1000)
        param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
    elif model_name == "Neural Network Regressor (MLPRegressor)":
        model = MLPRegressor(max_iter=1000)
        param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
    else:
        return {"Model not recognized."}

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy' if 'Classifier' in model_name else 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # return {"Best Parameters": grid_search.best_params_}
    return grid_search.best_estimator_

def execute_model_with_hyperparameters(model_name, X_train, X_test, y_train, y_test):
    model = hyperparameter_tuning(model_name, X_train, y_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if model_name in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor", "K-Nearest Neighbors Regressor", "Neural Network Regressor (MLPRegressor)"]:
        mse = mean_squared_error(y_test, predictions)
        return {"Mean Squared Error": mse}
    else:
        accuracy = accuracy_score(y_test, predictions)
        return {"Accuracy": accuracy}


@app.post("/execute_model_w_hyperparams")
async def execute_model_endpoint(target, model, file: UploadFile = File(...)):
    contents = await file.read()
    data = load_data(contents.decode('utf-8'))
    X = data.drop(columns=[target])
    y = data[target]
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metric = execute_model_with_hyperparameters(model, X_train, X_test, y_train, y_test)
    return {"status": "Model executed", "Performance": metric}