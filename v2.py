import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import io


def load_data(file_path):
    return pd.read_csv(file_path)


def dataset_info(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()

    lines = s.split('\n')
    info_dict = {
        "Class": lines[0],
        "RangeIndex": lines[1],
        "Data columns": lines[2],
        "Columns": [line.strip() for line in lines[3:-3]],
        "dtypes": lines[-3],
        "memory usage": lines[-2]
    }

    st.subheader("Dataset Information")
    st.text(info_dict["RangeIndex"])
    st.text(info_dict["Data columns"])

    st.subheader("Columns Information")
    columns_info = pd.DataFrame([col.split(maxsplit=4) for col in info_dict["Columns"]],
                                columns=["#", "Column", "Non-Null Count", "Dtype", "Extra"])
    st.dataframe(columns_info.drop(columns=["Extra"]))

    st.text(info_dict["dtypes"])


def stat_analysis(data):
    st.write("STAT ANALYSIS: \n")
    st.write(data.describe().T)


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
    st.write("Correlated Matrix \n")
    st.write(cor_matrix)
    return cor_matrix


def high_covariance_features(data):
    core_num = data.select_dtypes(include=[np.number])
    core_num = core_num.dropna(axis='columns')
    vif_data = pd.DataFrame()
    vif_data['Feature'] = core_num.columns
    vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]
    st.write("VIF \n")
    st.write(vif_data)
    return vif_data


def plot_correlation_matrix(cor_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_matrix, annot=True,  cmap='coolwarm', linewidths=0.5)
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


st.title("Data Analysis and Feature Importance")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if st.checkbox("Show Dataset Info"):
        dataset_info(data)

    if st.checkbox("Show Statistical Analysis"):
        stat_analysis(data)

    if st.checkbox("Show Non-Numeric Columns"):
        non_numeric_col = non_numeric_columns(data)

        if st.checkbox("Show Unique Values for Non-Numeric Columns"):
            unique_values(data, non_numeric_col)

    if st.checkbox("Show Correlation Matrix"):
        cor_matrix = correlation_matrix(data)

        if st.checkbox("Show High Covariance Features (VIF)"):
            vif_data = high_covariance_features(data)

        if st.checkbox("Plot Correlation Matrix Heatmap"):
            plot_correlation_matrix(cor_matrix)

    target_variable = st.text_input("Enter the target variable for feature importance calculation:")

    if target_variable:
        imp_df = feature_importance(data, target_variable)
