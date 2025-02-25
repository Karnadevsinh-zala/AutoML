import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path)


def dataset_info(data):
    print("DATASET: \n")
    print(data.info(), "\n")


def stat_analysis(data):
    print("STAT ANALYSIS: \n")
    print(data.describe().T, "\n")


def non_numeric_columns(data):
    print("NON NUMERIC COLS: \n")
    non_numeric_col = data.select_dtypes(include=['object']).columns
    print(non_numeric_col, "\n")
    return non_numeric_col


def unique_values(data, non_numeric_col):
    for column in non_numeric_col:
        print(f"Unique values in {column}:")
        print(data[column].unique())


def correlation_matrix(data):
    num_data = data.copy()
    for column in num_data.select_dtypes(include=['object']).columns:
        num_data[column] = pd.to_numeric(num_data[column], errors='coerce')
    cor_matrix = num_data.corr()
    print("Correlated Matrix \n")
    print(cor_matrix)
    return cor_matrix


def high_covariance_features(data):
    core_num = data.select_dtypes(include=[np.number])
    core_num = core_num.dropna(axis='columns')
    vif_data = pd.DataFrame()
    vif_data['Feature'] = core_num.columns
    vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]
    print("VIF \n")
    print(vif_data)
    return vif_data


def plot_correlation_matrix(cor_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()


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

    print("Feature Importance Scores:")
    print(imp_df)

    return imp_df


file_path = "Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = load_data(file_path)

dataset_info(data)
stat_analysis(data)
non_numeric_col = non_numeric_columns(data)
unique_values(data, non_numeric_col)
cor_matrix = correlation_matrix(data)
vif_data = high_covariance_features(data)
plot_correlation_matrix(cor_matrix)
imp_df = feature_importance(data, 'tenure')
