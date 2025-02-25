import optuna
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import io
from statsmodels.stats.outliers_influence import variance_inflation_factor
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, precision_score, \
    recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import f1_score, silhouette_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# Download WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(file_path):
    return pd.read_csv(file_path)

def dataset_info(data):
    st.write("Dataset Info\n")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()

    # lines = s.split('\n')
    # info_dict = {
    #     "Class": lines[0],
    #     "RangeIndex": lines[1],
    #     "Data columns": lines[2],
    #     "Columns": [line.strip() for line in lines[3:-3]],
    #     "dtypes": lines[-3],
    #     "memory usage": lines[-2]
    # }
    #
    # st.subheader("Dataset Information")
    # st.text(info_dict["RangeIndex"])
    # st.text(info_dict["Data columns"])
    #
    # st.subheader("Columns Information")
    # columns_info = pd.DataFrame([col.split(maxsplit=4) for col in info_dict["Columns"]],
    #                             columns=["#", "Column", "Non-Null Count", "Dtype", "Extra"])
    # st.dataframe(columns_info.drop(columns=["Extra"]))
    #
    # st.text(info_dict["dtypes"])
    st.write(data)

def stat_analysis(data):
    st.write("STAT ANALYSIS: \n")
    description = data.describe(include='all').T
    st.write(description)

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

            description = ""

            if kurtosis > 3:
                description += " The data has a high peak and heavy tails."
            elif kurtosis < 3:
                description += " The data has a low peak and light tails."

            if variance > 1000:
                description += " The variance is high, indicating a large spread in the data."
            elif variance < 100:
                description += " The variance is low, indicating a small spread in the data."
            else:
                description += " The variance is moderate, indicating a moderate spread in the data."

        else:
            data_types.append("Categorical")
            unique_values = data[column].nunique()
            top_value_freq = data[column].value_counts().iloc[0]

            if unique_values > 10:
                description = f"It contains categorical data with many unique values."
            elif unique_values > 2:
                description = f"It contains categorical data with several unique values."
            else:
                description = f"It contains binary categorical data."

            if top_value_freq / len(data) > 0.5:
                dominant_category = data[column].mode()[0]
                description += f" There is a dominant category: '{dominant_category}'."
            else:
                description += " There is no dominant category."

        descriptions.append(description)

    column_details = pd.DataFrame({
        "Column Name": column_names,
        "Data Type": data_types,
        "Description": descriptions
    })

    st.write("COLUMN DETAILS:\n")
    # st.write(column_details)
    # st.dataframe(column_details.style.set_properties(**{'width': '900px'}))
    st.table(column_details)

def non_numeric_columns(data):
    st.write("NON NUMERIC COLS: \n")
    non_numeric_col = data.select_dtypes(include=['object']).columns
    st.write(non_numeric_col)
    return non_numeric_col

def unique_values(data, non_numeric_col):
    for column in non_numeric_col:
        st.write(f"Unique values in {column}:")
        st.write(data[column].unique())

def correlation_matrix(data):
    num_data = data.copy()
    for column in num_data.select_dtypes(include=['object']).columns:
        num_data[column] = pd.to_numeric(num_data[column], errors='coerce')
    cor_matrix = num_data.corr()
    # st.write(cor_matrix)

    # plot_correlation_matrix(cor_matrix)

    results = pd.DataFrame(
        columns=['Column', 'Top Correlated Column 1', 'Top Correlated Column 2',
                 'Top Correlated Column 3'])

    k = 3
    # Iterate over each column in the correlation matrix
    for column in cor_matrix.columns:
        # Get the top k correlated columns for the current column
        top_k = cor_matrix[column].nlargest(k + 1).iloc[1:]  # Exclude the column itself

        # Append the results to the DataFrame using pd.concat
        results = pd.concat([results, pd.DataFrame({
            'Column': [column],
            'Top Correlated Column 1': [f"{top_k.index[0]}: {top_k.iloc[0]:.2f}"],
            'Top Correlated Column 2': [f"{top_k.index[1]}: {top_k.iloc[1]:.2f}"],
            'Top Correlated Column 3': [f"{top_k.index[2]}: {top_k.iloc[2]:.2f}"]
        })], ignore_index=True)

    st.write(results)

    return cor_matrix


def high_covariance_features(data, vif_threshold=10, corr_threshold=0.8):
    core_num = data.select_dtypes(include=[np.number])
    core_num = core_num.dropna(axis='columns')
    vif_data = pd.DataFrame()
    vif_data['Feature'] = core_num.columns
    vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]
    st.write("VIF \n")
    st.write(vif_data)

    # Identify multicollinear features based on VIF
    multicollinear_features = vif_data[vif_data['VIF'] > vif_threshold]['Feature'].tolist()

    # Calculate correlation matrix
    corr_matrix = core_num.corr().abs()

    # Identify highly correlated pairs
    multicollinear_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > corr_threshold:
                multicollinear_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    st.write("Multicollinear Pairs \n")
    st.write(multicollinear_pairs)

    return vif_data, multicollinear_pairs

def plot_correlation_matrix(cor_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)

def feature_importance(data, target_variable):
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

    st.write("Feature Importance Scores:")
    st.write(imp_df)

    return imp_df


def group_features(data, target_variable):
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
        grouped_features[threshold] = groups

    return grouped_features

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

def select_best_group(data, target_variable, top_k=3):
    grouped_features = group_features(data, target_variable)
    performance_results = evaluate_group_performance(data, target_variable, grouped_features)

    # Sort the results by accuracy in descending order and select the top k groups
    sorted_results = sorted(performance_results.items(), key=lambda x: x[1][1], reverse=True)[:top_k]

    best_groups = [{"Threshold": threshold, "Features": groups, "Accuracy": accuracy}
                   for threshold, (groups, accuracy) in sorted_results]

    best_groups_df = pd.DataFrame(best_groups)
    return best_groups_df


def custom_binning(data, column):
    num_bins = int(np.ceil(np.log2(len(data[column])) + 1))

    binned_data, bin_edges = pd.qcut(data[column], q=num_bins, retbins=True, duplicates='drop')

    data[f'{column}_binned'] = binned_data.apply(lambda x: x.mid)

    return data, bin_edges


def determine_bins(dataset, column):
    data = dataset[column].dropna()

    n = len(data)
    data_range = data.max() - data.min()
    iqr = np.subtract(*np.percentile(data, [75, 25]))  # Interquartile range

    # Sturges' rule
    sturges_bins = np.ceil(np.log2(n) + 1)

    # Square-root choice
    sqrt_bins = np.ceil(np.sqrt(n))

    # Freedman-Diaconis rule
    if iqr > 0:
        fd_bins = np.ceil(data_range / (2 * iqr / np.cbrt(n)))
    else:
        fd_bins = sturges_bins  # Fallback if IQR is 0

    # Choose the method based on data size and distribution
    if n < 30:
        # For small datasets, use square-root choice
        return int(sqrt_bins)
    elif data_range == 0:
        # If all values are the same, only one bin is needed
        return 1
    else:
        # For larger datasets, use Freedman-Diaconis if IQR > 0, else Sturges
        return int(fd_bins if iqr > 0 else sturges_bins)


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


def bin_numerical_columns_with_visualization(dataset):
    df = dataset.copy()
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

            st.pyplot(fig)

    return df

def sampling(data, sample_size):
    sample = data.sample(n=sample_size, random_state=42)
    st.write(f"Random Sample of size {sample_size}:")
    st.write(sample.head())

    return sample


def sample_numerical_columns(dataset, sample_fraction=0.1):
    df = dataset.copy()
    sampled_df = pd.DataFrame()
    no_sampling_required = []
    sampled_columns = []

    for column in df.select_dtypes(include=[np.number]).columns:
        uniqueValues = df[column].nunique()
        total_values = len(df[column])

        if uniqueValues > 0.5 * total_values and total_values > 1000:
            sampled_columns.append(column)
            st.write(f"Sampling column: {column}")

            sampled_df[column] = df[column].sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

            st.write(f"Distribution of {column} before and after sampling:")
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(df[column], bins=30, kde=True, color='blue', ax=axes[0])
            axes[0].set_title(f'Original {column}')
            axes[0].set_xlabel(column)
            axes[0].set_ylabel('Frequency')

            sns.histplot(sampled_df[column], bins=30, kde=True, color='green', ax=axes[1])
            axes[1].set_title(f'Sampled {column}')
            axes[1].set_xlabel(column)
            axes[1].set_ylabel('Frequency')

            st.pyplot(fig)
        else:
            no_sampling_required.append(column)
            sampled_df[column] = df[column]

    if not sampled_columns:
        st.write("No sampling was required for any numerical columns in the dataset.")
    elif no_sampling_required:
        st.write(f"No sampling required for columns: {', '.join(no_sampling_required)}")
    else:
        st.write("Sampling was required for all numerical columns.")

    return sampled_df


def tokenization(data, column):
    st.write("Tokenization")
    vectorizer = CountVectorizer()
    tokenized_data = vectorizer.fit_transform(data[column])
    st.write(tokenized_data.toarray())
    return tokenized_data

def lemmatization(data, column):
    st.write("Lemmatization")
    lemmatizer = WordNetLemmatizer()
    lemmatized_data = data[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    st.write(lemmatized_data)
    return lemmatized_data

def scaling_normalization(data):
    st.write("Scaling and Normalization")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    st.write(scaled_data)
    return scaled_data

def dimensionality_reduction(data):
    st.write("Dimensionality Reduction")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data.select_dtypes(include=[np.number]))
    st.write(reduced_data)
    return reduced_data


def apply_dimensionality_reduction(dataset, target_column, explained_variance_threshold=0.95):
    df = dataset.copy()

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Drop the target column if it is numeric
    if target_column in numerical_cols:
        numerical_cols = numerical_cols.drop(target_column)

    if len(numerical_cols) < 2:
        st.write("Dimensionality reduction not applicable: Not enough numerical columns.")
        return df

    # Scale the data excluding the target column
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])

    pca = PCA()
    pca.fit(scaled_data)

    # Determine the number of components needed to explain the desired variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.searchsorted(cumulative_variance, explained_variance_threshold) + 1

    if num_components >= len(numerical_cols):
        st.write("Dimensionality reduction not necessary: All components needed to retain variance.")
        return df

    reduced_data = pca.transform(scaled_data)[:, :num_components]
    reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i + 1}' for i in range(num_components)])

    st.write(
        f"Reduced dimensions from {len(numerical_cols)} to {num_components} to retain {explained_variance_threshold * 100}% variance.")
    st.write("Explained variance ratio for each principal component:")
    st.bar_chart(pca.explained_variance_ratio_[:num_components])

    fig, ax = plt.subplots()
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    ax.axhline(y=explained_variance_threshold, color='r', linestyle='--')
    ax.set_title('Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    st.pyplot(fig)

    # Concatenate reduced data with non-numerical columns and the target column
    result_df = pd.concat([df.drop(columns=numerical_cols), reduced_df], axis=1)

    return result_df

def feature_selection(data, target_variable):
    st.write("Feature Selection")
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

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
    st.write(selected_features)
    return selected_features

def handle_imbalanced_data(data, target_variable):
    st.write("Handling Imbalanced Data")
    majority_class = data[data[target_variable] == data[target_variable].mode()[0]]
    minority_class = data[data[target_variable] != data[target_variable].mode()[0]]
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=1)
    balanced_data = pd.concat([majority_class, minority_upsampled])
    st.write(balanced_data)
    return balanced_data

def feature_contribution(model, X_train):
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        st.write("Model does not have feature importance attribute.")
        return

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    st.write("Feature Contribution:")
    st.write(feature_importance)


def predictors_insight(model, X_test, y_test):
    predictions = model.predict(X_test)
    residuals = y_test - predictions

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    st.pyplot(plt)


def calculate_scores(model_name, y_test, predictions):
    if model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                      "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor",
                      "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor",
                      "K-Nearest Neighbors Regressor", "Neural Network Regressor (MLPRegressor)"]:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R2 Score: {r2}")
    else:
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        report = classification_report(y_test, predictions, output_dict=True)

        st.write(f"Accuracy: {accuracy}")
        st.write(f"F1 Score: {f1}")

        st.write("Classification Report:")

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)


def model_catalog():
    models = {
        "Linear Regression": "A linear approach to modeling the relationship between a dependent variable and one or more independent variables.",
        "Logistic Regression": "A statistical model that in its basic form uses a logistic function to model a binary dependent variable.",
        "Decision Tree Classifier": "A decision support tool that uses a tree-like model of decisions and their possible consequences.",
        "Decision Tree Regressor": "A decision support tool that uses a tree-like model for regression tasks.",
        "Random Forest Classifier": "An ensemble learning method for classification that operates by constructing multiple decision trees.",
        "Random Forest Regressor": "An ensemble learning method for regression that operates by constructing multiple decision trees.",
        "Support Vector Classifier": "A supervised learning model that analyzes data for classification and regression analysis.",
        "Support Vector Regressor": "A type of Support Vector Machine that supports linear and non-linear regression.",
        "Gradient Boosting Classifier": "An ensemble learning method that builds multiple decision trees sequentially to minimize the loss function.",
        "Gradient Boosting Regressor": "An ensemble learning method that builds multiple decision trees sequentially for regression tasks.",
        "XGBoost Classifier": "An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.",
        "XGBoost Regressor": "An optimized distributed gradient boosting library for regression tasks.",
        "LightGBM Classifier": "A gradient boosting framework that uses tree-based learning algorithms, designed for efficiency and speed.",
        "LightGBM Regressor": "A gradient boosting framework for regression tasks.",
        "CatBoost Classifier": "A gradient boosting library that is fast and provides state-of-the-art results without extensive hyperparameter tuning.",
        "CatBoost Regressor": "A gradient boosting library for regression tasks.",
        "K-Nearest Neighbors Classifier": "A simple, instance-based learning algorithm that classifies based on the majority class among the k-nearest neighbors.",
        "K-Nearest Neighbors Regressor": "A simple, instance-based learning algorithm for regression tasks.",
        "Neural Network Classifier (MLPClassifier)": "A neural network model for classification tasks.",
        "Neural Network Regressor (MLPRegressor)": "A neural network model for regression tasks."
    }

    st.write("Model Catalog:")
    for model, description in models.items():
        st.write(f"**{model}**: {description}")


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
        st.write("Model not recognized.")
        return

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if model_name in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor", "K-Nearest Neighbors Regressor", "Neural Network Regressor (MLPRegressor)"]:
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Mean Squared Error: {mse}")
    else:
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy}")


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
        st.write("Model not recognized.")
        return

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy' if 'Classifier' in model_name else 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    st.write(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def execute_model_with_hyperparameters(model_name, X_train, X_test, y_train, y_test):
    model = hyperparameter_tuning(model_name, X_train, y_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if model_name in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor", "K-Nearest Neighbors Regressor", "Neural Network Regressor (MLPRegressor)"]:
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Mean Squared Error: {mse}")
    else:
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy}")

    feature_contribution(model, X_train)
    predictors_insight(model, X_test, y_test)
    calculate_scores(model_name, y_test, predictions)


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


def train_and_compare_models(data, target_variable):
    X_transformed, y_encoded, _ = preprocess_data(data, target_variable)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.3, random_state=42)

    models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM", "KNN"]
    best_metrics = {model_name: {} for model_name in models}
    best_model = None
    best_model_name = ""
    best_accuracy = 0.0

    results = []

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

        results.append(best_metrics)

        st.write(f"Best Model: {model_name}")
        st.write("Best Hyperparameters:", best_params)
        st.write(best_metrics[model_name])

    df_results = pd.DataFrame(best_metrics).T
    df_results = df_results.transpose()

    st.balloons()
    st.write("\nModel Evaluation Results:")
    st.table(df_results)

    # Prepare data for plotting
    accuracy_df = pd.DataFrame({
        "Model": models,
        "Accuracy": [best_metrics[model]["Accuracy"] for model in models]
    }).set_index("Model")

    precision_df = pd.DataFrame({
        "Model": models,
        "Precision": [best_metrics[model]["Precision"] for model in models]
    }).set_index("Model")

    recall_df = pd.DataFrame({
        "Model": models,
        "Recall": [best_metrics[model]["Recall"] for model in models]
    }).set_index("Model")

    f1_df = pd.DataFrame({
        "Model": models,
        "F1 Score": [best_metrics[model]["F1 Score"] for model in models]
    }).set_index("Model")

    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Precision", "Recall", "F1 Score"])

    with tab1:
        st.write("Model Accuracy Comparison")
        st.bar_chart(accuracy_df)

    with tab2:
        st.write("Model Precision Comparison")
        st.bar_chart(precision_df)

    with tab3:
        st.write("Model Recall Comparison")
        st.bar_chart(recall_df)

    with tab4:
        st.write("Model F1 Score Comparison")
        st.bar_chart(f1_df)

    if best_model is not None:
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(best_model, initial_types=initial_type)
        with open(f"best_model_{best_model_name}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

    return best_metrics


################################################################################################################
# st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    @import url('style.css');
    </style>
    """,
    unsafe_allow_html=True
)

st.title("ML Pipeline")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = data.dropna()

    tab1, tab2, tab3, tab4 = st.tabs(["Info", "Feature Selection","Processing", "Model Execution"])

    with tab1:
        dataset_info(data)

        if st.checkbox("Statistical Analysis"):
            stat_analysis(data)

        # if st.checkbox("Non-Numeric Columns"):
        #     non_numeric_col = non_numeric_columns(data)
        #
        #     if st.checkbox("Unique Values for Non-Numeric Columns"):
        #         unique_values(data, non_numeric_col)

        if st.checkbox("Correlation"):
            cor_matrix = correlation_matrix(data)
            # plot_correlation_matrix(cor_matrix)


    with tab2:
        if st.checkbox("High Covariance Features"):
            high_covariance_features(data)

        if st.checkbox("Feature Importance"):
            target_variable = st.text_input("Enter the target variable:")
            if target_variable:
                feature_importance(data, target_variable)

        if st.checkbox("Group Features and Evaluate Performance"):
            target_variable = st.selectbox("Select Target Variable", ["Select an option"] + list(data.columns))

            if target_variable != "Select an option":
                groups = group_features(data, target_variable)
                evaluate_group_performance(data, target_variable, groups)
                st.write(select_best_group(data, target_variable))
            else:
                st.warning("Please select a target variable.")


    with tab3:
        if st.checkbox("Binning"):
            # column = st.selectbox("Select Column for Binning", data.select_dtypes(include=[np.number]).columns)
            # data, bin_edges = custom_binning(data, column)
            # st.write(f"Binned {column}:")
            # st.write(data[[column, f'{column}_binned']].head())
            # st.write(f"Bin edges for {column}: {bin_edges}")
            st.write(bin_numerical_columns_with_visualization(data))
            data = bin_numerical_columns_with_visualization(data)

        if st.checkbox("Sampling"):
            # sample_size = st.slider("Select Sample Size", min_value=10, max_value=len(data), value=100)
            # sampling(data, sample_size)
            sample_numerical_columns(data)
            # data = sample_numerical_columns(data)


        # if st.checkbox("Scaling and Normalization"):
        #     scaling_normalization(data)

        if st.checkbox("Dimensionality Reduction"):
            # dimensionality_reduction(data)
            target_variable = st.text_input("Enter the target variable:", key="dr")

            if target_variable:
                data = apply_dimensionality_reduction(data, target_variable)
                st.write(data)
            else:
                st.warning("Please enter a target variable.")

        # if st.checkbox("Feature Selection"):
        #     target_variable = st.text_input("Enter the target variable for feature selection:")
        #     if target_variable:
        #         feature_selection(data, target_variable)
        #
        # if st.checkbox("Handle Imbalanced Data"):
        #     target_variable = st.text_input("Enter the target variable for handling imbalanced data:")
        #     if target_variable:
        #         handle_imbalanced_data(data, target_variable)


    with tab4:
        if st.checkbox("Model Execution"):
            target_variable = st.selectbox("Select target variable", ["Select an option"] + list(data.columns),
                                           key="me")

            if target_variable != "Select an option":
                train_and_compare_models(data, target_variable)
            else:
                st.warning("Please select a target variable.")

        # if st.checkbox("Model Catalog"):
        #     model_catalog()
        #
        # if st.checkbox("Execute Model"):
        #     target_variable = st.selectbox("Select Target Variable", data.columns, key="tm")
        #     model_name = st.selectbox("Select Model", [
        #         "Linear Regression", "Logistic Regression", "Decision Tree Classifier",
        #         "Decision Tree Regressor", "Random Forest Classifier", "Random Forest Regressor",
        #         "Support Vector Classifier", "Support Vector Regressor",
        #         "Gradient Boosting Classifier", "Gradient Boosting Regressor",
        #         "XGBoost Classifier", "XGBoost Regressor",
        #         "LightGBM Classifier", "LightGBM Regressor",
        #         "CatBoost Classifier", "CatBoost Regressor",
        #         "K-Nearest Neighbors Classifier", "K-Nearest Neighbors Regressor",
        #         "Neural Network Classifier (MLPClassifier)", "Neural Network Regressor (MLPRegressor)"
        #     ], key="em")
        #
        #     X = data.drop(columns=[target_variable])
        #     y = data[target_variable]
        #
        #     for column in X.select_dtypes(include=['object']).columns:
        #         le = LabelEncoder()
        #         X[column] = le.fit_transform(X[column])
        #
        #     if y.dtype == 'object':
        #         le = LabelEncoder()
        #         y = le.fit_transform(y)
        #
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        #     execute_model(model_name, X_train, X_test, y_train, y_test)
        #
        # if st.checkbox("Execute Model w hyperparams"):
        #     target_variable = st.selectbox("Select Target Variable", data.columns, key="et")
        #     model_name = st.selectbox("Select Model", [
        #         "Linear Regression", "Logistic Regression", "Decision Tree Classifier",
        #         "Decision Tree Regressor", "Random Forest Classifier", "Random Forest Regressor",
        #         "Support Vector Classifier", "Support Vector Regressor",
        #         "Gradient Boosting Classifier", "Gradient Boosting Regressor",
        #         "XGBoost Classifier", "XGBoost Regressor",
        #         "LightGBM Classifier", "LightGBM Regressor",
        #         "CatBoost Classifier", "CatBoost Regressor",
        #         "K-Nearest Neighbors Classifier", "K-Nearest Neighbors Regressor",
        #         "Neural Network Classifier (MLPClassifier)", "Neural Network Regressor (MLPRegressor)"
        #     ], key="hp")
        #
        #     X = data.drop(columns=[target_variable])
        #     y = data[target_variable]
        #
        #     for column in X.select_dtypes(include=['object']).columns:
        #         le = LabelEncoder()
        #         X[column] = le.fit_transform(X[column])
        #
        #     if y.dtype == 'object':
        #         le = LabelEncoder()
        #         y = le.fit_transform(y)
        #
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        #     execute_model_with_hyperparameters(model_name, X_train, X_test, y_train, y_test)