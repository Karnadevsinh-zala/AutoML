## Setup Instructions

### Installation
Clone the repository:
```bash
git clone https://github.com/Karnadevsinh-zala/AutoML.git
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Run
Start the API server:
```
fastapi run exp_v1.py
```
Launch Streamlit driver:
```
streamlit run driver.py
```

## Technologies Used
- **FastAPI**: For building the RESTful API backend.
- **Streamlit**: For creating the interactive web application frontend.
- **Optuna**: For hyperparameter optimization.
- **scikit-learn**: For machine learning algorithms and preprocessing.
- **pandas**: For data manipulation and analysis.
- **seaborn** and **matplotlib**: For data visualization.
- **ONNX**: For exporting machine learning models in a portable format.

## Usage

1. **Upload a CSV File**: Use the Streamlit interface to upload your dataset.
2. **Explore Data**: Perform dataset inspection and statistical analysis.
3. **Feature Engineering**: Compute correlations, identify high covariance features, and evaluate feature importance.
4. **Process Data**: Bin and sample numerical columns, apply dimensionality reduction.
5. **Train Models**: Select a target variable and train multiple models. View the best-performing model and its hyperparameters.
6. **Visualize Results**: Compare model performances.

## API Endpoints

- **`/upload-file/`**: Upload a CSV file.
- **`/dataset-info/`**: Get dataset information.
- **`/stat-analysis/`**: Statistical analysis on the dataset.
- **`/correlation-matrix/`**: Compute correlation matrix of the dataset.
- **`/high-covariance-features/`**: Identifies features with high covariance.
- **`/feature-importance/`**: Calculates feature importance for a target variable.
- **`/group-features/`**: Groups features based on correlation with a target variable and evaluate model performance on these groups.
- **`/bin-numerical-columns/`**: Bin numerical columns and provides visualizations of the original and binned distributions.
- **`/sample-numerical-columns/`**: Samples numerical columns to reduce data size and provides visualizations of original and sampled distributions.
- **`/apply-dimensionality-reduction/`**: Applied PCA to reduce dimensionality of numerical data while retaining specified variance.
- **`/train-and-compare-models/`**: Trains and compares machine learning models.
