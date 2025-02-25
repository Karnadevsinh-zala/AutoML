from statistics import correlation

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

### --- Dataset info
print("DATASET: \n")
print(data.info, "\n")

### --- Stat
print("STAT ANALYSIS: \n")
print(data.describe().T, "\n")

### --- Non numeric columns
print("NON NUMERIC COLS: \n")
non_numeric_col = data.select_dtypes(include=['object']).columns
print(non_numeric_col, "\n")

### --- Unique values
for column in non_numeric_col:
    print(f"Unique values in {column}:")
    print(data[column].unique())

numData = data

for column in non_numeric_col:
    numData[column] = pd.to_numeric(numData[column], errors='coerce')

cor_matrix = numData.corr()
print("Correlated Matrix \n")
print(cor_matrix)

### --- High covariance features
core_num = data.select_dtypes(include=[np.number])
core_num = core_num.dropna(axis='columns')
print(core_num)

vif_data = pd.DataFrame()
vif_data['Feature'] = core_num.columns
vif_data['VIF'] = [variance_inflation_factor(core_num.values, i) for i in range(core_num.shape[1])]

print("VIF \n")
print(vif_data)

plt.figure(figsize=(12,10))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

### --- Feature Importance
df = data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

cm = core_num.corr()
print(cm)
targetVar = 'tenure'
feature_imp = cm[targetVar].drop(targetVar).abs().sort_values(ascending=False)
# print(feature_imp)
imp_df = pd.DataFrame({
    'Feature': feature_imp.index,
    'Imp Score': feature_imp.values
})

print("Feature Importance Scores:")
print(imp_df)