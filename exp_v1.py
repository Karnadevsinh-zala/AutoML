import base64

import optuna
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import io
import seaborn as sns
from sklearn.svm import SVC
from starlette.responses import JSONResponse
from io import BytesIO
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

app = FastAPI()



@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        return {"columns": data.columns.tolist(), "message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload file: {e}")


@app.post("/dataset-info/")
async def dataset_info(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        buffer = io.StringIO()
        data.info(buf=buffer)
        return {"info": buffer.getvalue()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get dataset info: {e}")


@app.post("/stat-analysis/")
async def stat_analysis(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        description = data.describe(include='all').T.to_dict()

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
async def correlation_matrix(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into a DataFrame
        data = pd.read_csv(file.file)

        # Convert non-numeric columns to numeric, coercing errors to NaN
        num_data = data.copy()
        for column in num_data.select_dtypes(include=['object']).columns:
            num_data[column] = pd.to_numeric(num_data[column], errors='coerce')

        # Compute the correlation matrix
        cor_matrix = num_data.corr()

        # Prepare results with top 3 correlated columns for each column
        results = pd.DataFrame(
            columns=['Column', 'Top Correlated Column 1', 'Top Correlated Column 2', 'Top Correlated Column 3']
        )

        k = 3
        for column in cor_matrix.columns:
            top_k = cor_matrix[column].nlargest(k + 1).iloc[1:]  # Exclude the column itself

            results = pd.concat([results, pd.DataFrame({
                'Column': [column],
                'Top Correlated Column 1': [f"{top_k.index[0]}: {top_k.iloc[0]:.2f}"],
                'Top Correlated Column 2': [f"{top_k.index[1]}: {top_k.iloc[1]:.2f}"],
                'Top Correlated Column 3': [f"{top_k.index[2]}: {top_k.iloc[2]:.2f}"]
            })], ignore_index=True)

        return {
            "correlation_matrix": cor_matrix.to_dict(),
            "top_correlations": results.to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to compute correlation matrix: {e}")


@app.post("/high-covariance-features/")
async def high_covariance_features(file: UploadFile = File(...), vif_threshold: float = 10, corr_threshold: float = 0.8):
    try:
        data = pd.read_csv(file.file)
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
async def feature_importance(target_variable, file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        df = data.copy()
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

        cor_matrix = df.corr()
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

    # Sort the results by accuracy in descending order and select the top k groups
    sorted_results = sorted(performance_results.items(), key=lambda x: x[1][1], reverse=True)[:top_k]

    best_groups = [{"Threshold": threshold, "Features": groups, "Accuracy": accuracy}
                   for threshold, (groups, accuracy) in sorted_results]

    best_groups_df = pd.DataFrame(best_groups)
    return best_groups_df

@app.post("/group-features/")
async def group_features(target_variable: str, file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
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



    # evaluate group performance
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
    #Freedman-Diaconis rule
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
async def bin_numerical_columns_with_visualization(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
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

        return JSONResponse(content=visualizations)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to bin numerical columns: {e}")


@app.post("/sample-numerical-columns/")
async def sample_numerical_columns(file: UploadFile = File(...), sample_fraction: float = 0.1):
    try:
        data = pd.read_csv(file.file)
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

        return JSONResponse(content={"visualizations": visualizations, "no_sampling_required": no_sampling_required})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to sample numerical columns: {e}")


@app.post("/apply-dimensionality-reduction/")
async def apply_dimensionality_reduction(file: UploadFile = File(...), target_column: str = '', explained_variance_threshold: float = 0.95):
    try:
        data = pd.read_csv(file.file)
        df = data.copy()

        numerical_cols = df.select_dtypes(include=[np.number]).columns

        # Drop the target column if it is numeric
        if target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)

        if len(numerical_cols) < 2:
            return JSONResponse(content={"message": "Dimensionality reduction not applicable: Not enough numerical columns."})

        # Scale the data excluding the target column
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

        # Visualization
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

        return JSONResponse(content={
            "message": f"Reduced dimensions from {len(numerical_cols)} to {num_components} to retain {explained_variance_threshold * 100}% variance.",
            "explained_variance": pca.explained_variance_ratio_[:num_components].tolist(),
            "cumulative_variance_image": cumulative_variance_image
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply dimensionality reduction: {e}")



def preprocess_data(data, target_variable):
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['number']).columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_transformed = preprocessor.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_transformed, y_encoded, label_encoder


def objective(trial, X_train, y_train, model_name):
    if model_name == "Logistic Regression":
        C = trial.suggest_loguniform("C", 1e-3, 1e3)
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
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 0.1)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    elif model_name == "SVM":
        C = trial.suggest_loguniform("C", 1e-3, 1e3)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        model = SVC(C=C, kernel=kernel, probability=True)

    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)
    return accuracy_score(y_train, model.predict(X_train))


@app.post("/train-and-compare-models/")
async def train_and_compare_models(file: UploadFile = File(...), target_variable: str = ''):
    try:
        data = pd.read_csv(file.file)
        X_transformed, y_encoded, _ = preprocess_data(data, target_variable)

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.3, random_state=42)

        models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM", "KNN"]
        best_metrics = {model_name: {} for model_name in models}
        best_model = None
        best_model_name = ""
        best_accuracy = 0.0

        for model_name in models:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X_train, y_train, model_name), n_trials=50)

            best_trial = study.best_trial
            best_params = best_trial.params

            if model_name == "Logistic Regression":
                model = LogisticRegression(**best_params, max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(**best_params)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(**best_params)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(**best_params)
            elif model_name == "SVM":
                model = SVC(**best_params, probability=True)
            elif model_name == "KNN":
                model = KNeighborsClassifier(**best_params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Collect metrics
            accuracy = accuracy_score(y_test, y_pred)
            best_metrics[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted')
            }

            # Check for the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

        return JSONResponse(content={"best_metrics": best_metrics, "best_model_name": best_model_name})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to train and compare models: {e}")