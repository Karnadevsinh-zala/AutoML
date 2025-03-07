import base64
import os
import uuid
import optuna
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import seaborn as sns

from starlette.responses import JSONResponse, StreamingResponse
from io import BytesIO
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

from pydantic import BaseModel

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

def get_file_path(file_id: str) -> str:
    return os.path.join(DATA_DIR, f"{file_id}.csv")

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        logging.info(f"File uploaded with ID: {file_id}")
        file_path = get_file_path(file_id)

        data = pd.read_csv(file.file)
        cleaned_data = data.dropna()
        cleaned_data.to_csv(file_path, index=False)

        return {"file_id": file_id, "columns": data.columns.tolist(), "message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload file: {e}")


@app.get("/get-file/")
async def get_file(file_id: str):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")

        # Return the file as a streaming response
        return StreamingResponse(open(file_path, "rb"), media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve file: {e}")


@app.post("/dataset-info/")
async def dataset_info(file_id: str):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)
        data_sample = data.to_dict(orient='records')
        return {"data": data_sample}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get dataset info: {e}")


@app.post("/split-dataset/")
async def split_dataset(file_id: str, test_size: float = 0.3):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")

        data = pd.read_csv(file_path)
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

        train_file_path = os.path.join(DATA_DIR, f"{file_id}_train.csv")
        test_file_path = os.path.join(DATA_DIR, f"{file_id}_test.csv")

        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)

        return {"train_file_id": f"{file_id}_train", "test_file_id": f"{file_id}_test",
                "message": "Dataset split successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to split dataset: {e}")


@app.post("/stat-analysis/")
async def stat_analysis(file_id: str):
    try:
        file_path = get_file_path(file_id)
        logging.info(f"Performing statistical analysis on file ID: {file_id}")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)
        description = data.describe(include='all').T

        description = description.replace([float('inf'), float('-inf'), float('nan')], None)
        description = description.to_dict()

        column_names = []
        data_types = []
        descriptions = []

        for column in data.columns:
            column_names.append(column)

            if pd.api.types.is_numeric_dtype(data[column]):
                data_types.append("Numeric")
                median = data[column].median()
                iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
                kurtosis = data[column].kurtosis()
                variance = data[column].var()

                description_text = ""

                if kurtosis > 3:
                    description_text += " The data has a high peak and heavy tails."
                elif kurtosis < 3:
                    description_text += " The data has a low peak and light tails."

                if variance > 1000:
                    description_text += " The variance is high, indicating a large spread in the data."
                elif variance < 100:
                    description_text += " The variance is low, indicating a small spread in the data."
                else:
                    description_text += " The variance is moderate, indicating a moderate spread in the data."

            else:
                data_types.append("Categorical")
                unique_values = data[column].nunique()
                top_value_freq = data[column].value_counts().iloc[0]

                if unique_values > 10:
                    description_text = f"It contains categorical data with many unique values."
                elif unique_values > 2:
                    description_text = f"It contains categorical data with several unique values."
                else:
                    description_text = f"It contains binary categorical data."

                if top_value_freq / len(data) > 0.5:
                    dominant_category = data[column].mode()[0]
                    description_text += f" There is a dominant category: '{dominant_category}'."
                else:
                    description_text += " There is no dominant category."

            descriptions.append(description_text)

        column_details = pd.DataFrame({
            "Column Name": column_names,
            "Data Type": data_types,
            "Description": descriptions
        }).to_dict()

        return {"description": description, "column_details": column_details}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to perform statistical analysis: {e}")


@app.post("/correlation-matrix/")
async def correlation_matrix(file_id: str):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            logging.error(f"File not found for file ID: {file_id}")
            raise HTTPException(status_code=404, detail="File not found.")

        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully for file ID: {file_id}")

        num_data = data.select_dtypes(include=[float, int])

        if num_data.empty:
            logging.error("No numeric data available for correlation matrix.")
            raise HTTPException(status_code=400, detail="No numeric data available for correlation matrix.")

        cor_matrix = num_data.corr()

        cor_matrix = cor_matrix.replace({float('nan'): None})

        results = pd.DataFrame(
            columns=['Column', 'Top Correlated Column 1', 'Top Correlated Column 2', 'Top Correlated Column 3']
        )

        k = 3
        for column in cor_matrix.columns:
            top_k = cor_matrix[column].dropna().nlargest(k + 1).iloc[1:]  # Exclude the column itself

            results = pd.concat([results, pd.DataFrame({
                'Column': [column],
                'Top Correlated Column 1': [f"{top_k.index[0]}: {top_k.iloc[0]:.2f}"] if len(top_k) > 0 else [None],
                'Top Correlated Column 2': [f"{top_k.index[1]}: {top_k.iloc[1]:.2f}"] if len(top_k) > 1 else [None],
                'Top Correlated Column 3': [f"{top_k.index[2]}: {top_k.iloc[2]:.2f}"] if len(top_k) > 2 else [None]
            })], ignore_index=True)

        return {
            "correlation_matrix": cor_matrix.to_dict(),
            "top_correlations": results.to_dict(orient='records')
        }

    except Exception as e:
        logging.error(f"Failed to compute correlation matrix: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to compute correlation matrix: {e}")


@app.post("/high-covariance-features/")
async def high_covariance_features(file_id: str, vif_threshold: float = 10, corr_threshold: float = 0.8):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)
        core_num = data.select_dtypes(include=[np.number]).dropna(axis='columns')
        vif_data = pd.DataFrame()
        vif_data['Feature'] = core_num.columns
        vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]

        multicollinear_pairs = []
        corr_matrix = core_num.corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > corr_threshold:
                    multicollinear_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        return {"vif_data": vif_data.to_dict(orient='records'), "multicollinear_pairs": multicollinear_pairs}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to identify high covariance features: {e}")

@app.post("/feature-importance/")
async def feature_importance(target_variable, file_id):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)
        df = data.copy()
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

        cor_matrix = df.corr()
        cor_matrix = cor_matrix.replace({float('nan'): None})
        feature_imp = cor_matrix[target_variable].drop(target_variable).abs().sort_values(ascending=False)

        imp_df = pd.DataFrame({
            'Feature': feature_imp.index,
            'Imp Score': feature_imp.values
        })

        return imp_df.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate feature importance: {e}")



def evaluate_group_performance(data, target_variable, grouped_features):
    results = {}

    if data[target_variable].dtype == 'object':
        label_encoder = LabelEncoder()
        data[target_variable] = label_encoder.fit_transform(data[target_variable])

    for threshold, groups in grouped_features.items():
        if not groups:
            continue

        X = data[groups]
        y = data[target_variable]

        # Encode categorical variables in the features
        for column in X.select_dtypes(include=['object']).columns:
            label_encoder = LabelEncoder()
            X[column] = label_encoder.fit_transform(X[column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[threshold] = (groups, accuracy)

    return results

async def select_best_group(data, target_variable, top_k=3):
    grouped_features = await group_features(target_variable, data)
    performance_results = evaluate_group_performance(data, target_variable, grouped_features)

    sorted_results = sorted(performance_results.items(), key=lambda x: x[1][1], reverse=True)[:top_k]

    best_groups = [{"Threshold": threshold, "Features": groups, "Accuracy": accuracy}
                   for threshold, (groups, accuracy) in sorted_results]

    best_groups_df = pd.DataFrame(best_groups)
    return best_groups_df

@app.post("/group-features/")
async def group_features(target_variable: str, file_id):
    file_path = get_file_path(file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    data = pd.read_csv(file_path)

    df = data.copy()
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    cor_matrix = df.corr()
    target_cor = cor_matrix[target_variable].drop(target_variable)

    grouped_features = {}
    for threshold in target_cor.abs().unique():
        groups = target_cor[target_cor.abs() > threshold].index.tolist()
        grouped_features[str(threshold)] = groups


    results = {}

    if data[target_variable].dtype == 'object':
        label_encoder = LabelEncoder()
        data[target_variable] = label_encoder.fit_transform(data[target_variable])

    for threshold, groups in grouped_features.items():
        if not groups:
            continue

        X = data[groups]
        y = data[target_variable]

        # Encode categorical variables in the features
        for column in X.select_dtypes(include=['object']).columns:
            label_encoder = LabelEncoder()
            X[column] = label_encoder.fit_transform(X[column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[threshold] = (groups, accuracy)

        # Select k best groups
        # Sort the results by accuracy in descending order and select the top k groups
        sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:3]

        best_groups = [{"Features": groups, "Accuracy": accuracy}
                       for threshold, (groups, accuracy) in sorted_results]

        best_groups_df = pd.DataFrame(best_groups)

    return best_groups_df


def determine_bins_for_column(data):
    #Freedman-Diaconis rule --> for optimal bins selection
    data = data.dropna()
    n = len(data)
    if n == 0:
        return 1
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr > 0:
        data_range = data.max() - data.min()
        fd_bins = np.ceil(data_range / (2 * iqr / np.cbrt(n)))
        return int(fd_bins)
    else:
        return 1

@app.post("/bin-numerical-columns/")
async def bin_numerical_columns_with_visualization(file_id):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)

        df = data.copy()
        visualizations = {}

        for column in df.select_dtypes(include=[np.number]).columns:
            unique_values = df[column].nunique()

            # Skip binary or categorical-like columns
            if unique_values <= 10:
                continue

            num_bins = determine_bins_for_column(df[column])

            if num_bins > 1:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                sns.histplot(df[column], bins=30, kde=True, color='blue', ax=axes[0])
                axes[0].set_title(f'Original Distribution of {column}')
                axes[0].set_xlabel(column)
                axes[0].set_ylabel('Frequency')

                df[column] = pd.cut(df[column], bins=num_bins, labels=False, include_lowest=True)

                sns.histplot(df[column], bins=num_bins, kde=False, color='green', ax=axes[1])
                axes[1].set_title(f'Binned Distribution of {column}')
                axes[1].set_xlabel(f'{column} Bins')
                axes[1].set_ylabel('Frequency')

                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                visualizations[column] = image_base64

        data.to_csv(file_path, index=False)
        return JSONResponse(content=visualizations)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to bin numerical columns: {e}")


@app.post("/sample-numerical-columns/")
async def sample_numerical_columns(file_id: str, sample_fraction: float = 0.1):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)

        df = data.copy()
        sampled_df = pd.DataFrame()
        visualizations = {}
        no_sampling_required = []
        sampled_columns = []

        for column in df.select_dtypes(include=[np.number]).columns:
            uniqueValues = df[column].nunique()
            total_values = len(df[column])

            if uniqueValues > 0.1 * total_values and total_values > 100:
                sampled_columns.append(column)

                sampled_df[column] = df[column].sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                sns.histplot(df[column], bins=30, kde=True, color='blue', ax=axes[0])
                axes[0].set_title(f'Original {column}')
                axes[0].set_xlabel(column)
                axes[0].set_ylabel('Frequency')

                sns.histplot(sampled_df[column], bins=30, kde=True, color='green', ax=axes[1])
                axes[1].set_title(f'Sampled {column}')
                axes[1].set_xlabel(column)
                axes[1].set_ylabel('Frequency')

                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                visualizations[column] = image_base64
            else:
                no_sampling_required.append(column)
                sampled_df[column] = df[column]

        data.to_csv(file_path, index=False)
        return JSONResponse(content={"visualizations": visualizations, "no_sampling_required": no_sampling_required})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to sample numerical columns: {e}")


@app.post("/apply-dimensionality-reduction/")
async def apply_dimensionality_reduction(file_id: str, target_column: str = '', explained_variance_threshold: float = 0.95):
    try:
        file_path = get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        data = pd.read_csv(file_path)

        df = data.copy()

        numerical_cols = df.select_dtypes(include=[np.number]).columns

        # Drop the target column if it is numeric
        if target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)

        if len(numerical_cols) < 2:
            return JSONResponse(content={"message": "Dimensionality reduction not applicable: Not enough numerical columns."})

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])

        pca = PCA()
        pca.fit(scaled_data)

        # Determine the number of components needed to explain the desired variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.searchsorted(cumulative_variance, explained_variance_threshold) + 1

        if num_components >= len(numerical_cols):
            return JSONResponse(content={"message": "Dimensionality reduction not necessary: All components needed to retain variance."})

        reduced_data = pca.transform(scaled_data)[:, :num_components]
        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i + 1}' for i in range(num_components)])

        fig, ax = plt.subplots()
        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        ax.axhline(y=explained_variance_threshold, color='r', linestyle='--')
        ax.set_title('Cumulative Explained Variance')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        cumulative_variance_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        data.to_csv(file_path, index=False)
        return JSONResponse(content={
            "message": f"Reduced dimensions from {len(numerical_cols)} to {num_components} to retain {explained_variance_threshold * 100}% variance.",
            "explained_variance": pca.explained_variance_ratio_[:num_components].tolist(),
            "cumulative_variance_image": cumulative_variance_image
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply dimensionality reduction: {e}")



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def create_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor

def preprocess_data(preprocessor, X):
    try:
        X_transformed = preprocessor.transform(X)
        return X_transformed
    except Exception as e:
        logger.error(f"Error during transformation: {e}")
        raise ValueError(f"Error during transformation: {e}")

def objective(trial, X_train, y_train, model_name):
    try:
        if model_name == "Logistic Regression":
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            model = LogisticRegression(C=C, max_iter=1000)

        elif model_name == "Decision Tree":
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

        elif model_name == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        elif model_name == "Gradient Boosting":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

        elif model_name == "SVM":
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
            model = SVC(C=C, kernel=kernel, probability=True)

        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)
        return accuracy_score(y_train, model.predict(X_train))
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise ValueError(f"Error during model training: {e}")

@app.post("/train-and-compare-models/")
async def train_and_compare_models(train_file_id: str, test_file_id: str, target_variable: str = ''):
    try:
        train_file_path = get_file_path(train_file_id)
        test_file_path = get_file_path(test_file_id)

        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            raise HTTPException(status_code=404, detail="Train or test file not found.")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        X_train = train_data.drop(columns=[target_variable])
        y_train = train_data[target_variable]

        X_test = test_data.drop(columns=[target_variable])
        y_test = test_data[target_variable]

        # Create preprocessor using training data
        preprocessor = create_preprocessor(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)

        # Use the same preprocessor for test data
        X_test_transformed = preprocess_data(preprocessor, X_test)

        # Encode target variable
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM", "KNN"]
        best_metrics = {model_name: {} for model_name in models}
        model_params = {model_name: {} for model_name in models}  # To store hyperparameters for each model
        best_model = None
        best_model_name = ""
        best_accuracy = 0.0
        best_params = {}

        for model_name in models:
            logger.info(f"Optimizing model: {model_name}")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X_train_transformed, y_train_encoded, model_name),
                           n_trials=50)

            best_trial = study.best_trial
            current_params = best_trial.params
            model_params[model_name] = current_params  # Store hyperparameters
            logger.info(f"Best parameters for {model_name}: {current_params}")

            if model_name == "Logistic Regression":
                model = LogisticRegression(**current_params, max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(**current_params)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(**current_params)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(**current_params)
            elif model_name == "SVM":
                model = SVC(**current_params, probability=True)
            elif model_name == "KNN":
                model = KNeighborsClassifier(**current_params)

            model.fit(X_train_transformed, y_train_encoded)
            y_pred = model.predict(X_test_transformed)

            accuracy = accuracy_score(y_test_encoded, y_pred)
            best_metrics[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision_score(y_test_encoded, y_pred, average='weighted'),
                "Recall": recall_score(y_test_encoded, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test_encoded, y_pred, average='weighted')
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
                best_params = current_params

        if best_model is not None:
            best_model_dir = "best_model"
            os.makedirs(best_model_dir, exist_ok=True)

            initial_type = [('float_input', FloatTensorType([None, X_train_transformed.shape[1]]))]
            onnx_model = convert_sklearn(best_model, initial_types=initial_type)
            onnx_file_path = os.path.join(best_model_dir, f"best_model_{best_model_name}.onnx")
            with open(onnx_file_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

        return JSONResponse(content={
            "best_metrics": best_metrics,
            "best_model_name": best_model_name,
            "best_params": best_params,
            "model_params": model_params  # Include all models' hyperparameters
        })

    except Exception as e:
        logger.error(f"Failed to train and compare models: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to train and compare models: {e}")


class RetrainRequest(BaseModel):
    hyperparameters: dict
    train_file_id: str
    test_file_id: str
    target_variable: str

@app.post("/retrain-models/")
async def retrain_models(request: RetrainRequest):
    try:
        # Load and preprocess data as before...

        train_file_path = get_file_path(request.train_file_id)
        test_file_path = get_file_path(request.test_file_id)

        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            raise HTTPException(status_code=404, detail="Train or test file not found.")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        X_train = train_data.drop(columns=[request.target_variable])
        y_train = train_data[request.target_variable]

        X_test = test_data.drop(columns=[request.target_variable])
        y_test = test_data[request.target_variable]

        preprocessor = create_preprocessor(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocess_data(preprocessor, X_test)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        best_model = None
        best_model_name = ""
        best_accuracy = 0.0

        best_metrics = {}
        for model_name, params in request.hyperparameters.items():
            if model_name == "Logistic Regression":
                model = LogisticRegression(**params, max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(**params)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(**params)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(**params)
            elif model_name == "SVM":
                model = SVC(**params, probability=True)
            elif model_name == "KNN":
                model = KNeighborsClassifier(**params)

            model.fit(X_train_transformed, y_train_encoded)
            y_pred = model.predict(X_test_transformed)

            accuracy = accuracy_score(y_test_encoded, y_pred)
            best_metrics[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision_score(y_test_encoded, y_pred, average='weighted'),
                "Recall": recall_score(y_test_encoded, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test_encoded, y_pred, average='weighted')
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
                model_param = params

        return JSONResponse(content={
            "best_metrics": best_metrics,
            "best_model_name": best_model_name,
            "param": model_param
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrain models: {e}")