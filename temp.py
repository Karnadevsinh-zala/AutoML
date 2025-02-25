from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
from typing import List, Dict, Any

app = FastAPI()

# Define Pydantic models for request bodies
class ColumnBins(BaseModel):
    column: str
    bins: int

class SampleSize(BaseModel):
    sample_size: int

class TokenizationColumn(BaseModel):
    column: str

class LemmatizationColumn(BaseModel):
    column: str

class TargetVariable(BaseModel):
    target_variable: str

class ModelExecution(BaseModel):
    model_name: str
    target_variable: str

class HyperparameterModelExecution(BaseModel):
    model_name: str
    target_variable: str

# Helper function to load data from uploaded file
def load_data(file: UploadFile):
    data = pd.read_csv(file.file)
    return data

@app.post("/load_data/")
async def load_data_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    return {"columns": data.columns.tolist(), "shape": data.shape}

@app.post("/dataset_info/")
async def dataset_info_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    return {"info": s}

@app.post("/stat_analysis/")
async def stat_analysis_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    return data.describe().T.to_dict()

@app.post("/non_numeric_columns/")
async def non_numeric_columns_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    non_numeric_col = data.select_dtypes(include=['object']).columns.tolist()
    return {"non_numeric_columns": non_numeric_col}

@app.post("/unique_values/")
async def unique_values_endpoint(file: UploadFile = File(...), columns: List[str]):
    data = load_data(file)
    unique_vals = {col: data[col].unique().tolist() for col in columns}
    return unique_vals

@app.post("/correlation_matrix/")
async def correlation_matrix_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    cor_matrix = data.corr()
    return cor_matrix.to_dict()

@app.post("/high_covariance_features/")
async def high_covariance_features_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    core_num = data.select_dtypes(include=[np.number]).dropna(axis='columns')
    vif_data = pd.DataFrame()
    vif_data['Feature'] = core_num.columns
    vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]
    return vif_data.to_dict()

@app.post("/feature_importance/")
async def feature_importance_endpoint(file: UploadFile = File(...), target_variable: TargetVariable):
    data = load_data(file)
    df = data.copy()
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    cor_matrix = df.corr()
    feature_imp = cor_matrix[target_variable.target_variable].drop(target_variable.target_variable).abs().sort_values(ascending=False)
    imp_df = pd.DataFrame({
        'Feature': feature_imp.index,
        'Imp Score': feature_imp.values
    })
    return imp_df.to_dict()

@app.post("/group_features/")
async def group_features_endpoint(file: UploadFile = File(...), target_variable: TargetVariable):
    data = load_data(file)
    cor_matrix = data.corr()
    target_cor = cor_matrix[target_variable.target_variable].drop(target_variable.target_variable)
    groups = target_cor[target_cor.abs() > 0.5].index.tolist()
    return {"groups": groups}

@app.post("/evaluate_group_performance/")
async def evaluate_group_performance_endpoint(file: UploadFile = File(...), target_variable: TargetVariable, groups: List[str]):
    data = load_data(file)
    X = data[groups]
    y = data[target_variable.target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {"accuracy": accuracy}

@app.post("/binning/")
async def binning_endpoint(file: UploadFile = File(...), column_bins: ColumnBins):
    data = load_data(file)
    data[f'{column_bins.column}_binned'] = pd.cut(data[column_bins.column], bins=column_bins.bins)
    data[f'{column_bins.column}_binned'] = data[f'{column_bins.column}_binned'].apply(lambda x: x.mid)
    return data[[column_bins.column, f'{column_bins.column}_binned']].head().to_dict()

@app.post("/sampling/")
async def sampling_endpoint(file: UploadFile = File(...), sample_size: SampleSize):
    data = load_data(file)
    sample = data.sample(n=sample_size.sample_size, random_state=42)
    return sample.head().to_dict()

@app.post("/tokenization/")
async def tokenization_endpoint(file: UploadFile = File(...), column: TokenizationColumn):
    data = load_data(file)
    vectorizer = CountVectorizer()
    tokenized_data = vectorizer.fit_transform(data[column.column])
    return {"tokenized_data": tokenized_data.toarray().tolist()}

@app.post("/lemmatization/")
async def lemmatization_endpoint(file: UploadFile = File(...), column: LemmatizationColumn):
    data = load_data(file)
    lemmatizer = WordNetLemmatizer()
    lemmatized_data = data[column.column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return lemmatized_data.tolist()

@app.post("/scaling_normalization/")
async def scaling_normalization_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    return {"scaled_data": scaled_data.tolist()}

@app.post("/dimensionality_reduction/")
async def dimensionality_reduction_endpoint(file: UploadFile = File(...)):
    data = load_data(file)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data.select_dtypes(include=[np.number]))
    return {"reduced_data": reduced_data.tolist()}

@app.post("/feature_selection/")
async def feature_selection_endpoint(file: UploadFile = File(...), target_variable: TargetVariable):
    data = load_data(file)
    X = data.drop(columns=[target_variable.target_variable])
    y = data[target_variable.target_variable]
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

@app.post("/handle_imbalanced_data/")
async def handle_imbalanced_data_endpoint(file: UploadFile = File(...), target_variable: TargetVariable):
    data = load_data(file)
    majority_class = data[data[target_variable.target_variable] == data[target_variable.target_variable].mode()[0]]
    minority_class = data[data[target_variable.target_variable] != data[target_variable.target_variable].mode()[0]]
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=1)
    balanced_data = pd.concat([majority_class, minority_upsampled])
    return balanced_data.to_dict()

@app.post("/execute_model/")
async def execute_model_endpoint(file: UploadFile = File(...), model_execution: ModelExecution):
    data = load_data(file)
    X = data.drop(columns=[model_execution.target_variable])
    y = data[model_execution.target_variable]
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    execute_model(model_execution.model_name, X_train, X_test, y_train, y_test)
    return {"status": "Model executed"}

@app.post("/execute_model_with_hyperparameters/")
async def execute_model_with_hyperparameters_endpoint(file: UploadFile = File(...), model_execution: HyperparameterModelExecution):
    data = load_data(file)
    X = data.drop(columns=[model_execution.target_variable])
    y = data[model_execution.target_variable]
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random