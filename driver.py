import base64

import streamlit as st
import requests
import pandas as pd

backend_url = "http://localhost:8000"

st.title("ML Pipeline with FastAPI Backend")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    tab1, tab2, tab3, tab4 = st.tabs(["Info", "Feature Selection", "Processing", "Model Execution"])

    with tab1:
        if st.checkbox("Statistical Analysis"):
            response = requests.post(f"{backend_url}/stat-analysis/", files=files)
            if response.status_code == 200:
                result = response.json()
                description = pd.DataFrame(result["description"]).T
                column_details = pd.DataFrame(result["column_details"])

                st.write("STAT ANALYSIS: \n")
                st.write(description)
                st.write("COLUMN DETAILS:\n")
                st.table(column_details)
            else:
                st.error("Failed to perform statistical analysis.")

        if st.checkbox("Compute Correlation"):
            response = requests.post(f"{backend_url}/correlation-matrix/", files=files)
            if response.status_code == 200:
                result = response.json()
                top_correlations = pd.DataFrame(result["top_correlations"])
                st.write("Top Correlated Columns:")
                st.table(top_correlations)
            else:
                st.error("Failed to compute correlation matrix.")


    with tab2:
        if st.checkbox("High Covariance Features"):
            vif_threshold = st.number_input("VIF Threshold", value=10.0)
            corr_threshold = st.number_input("Correlation Threshold", value=0.8)
            response = requests.post(f"{backend_url}/high-covariance-features/", files=files,
                                     params={"vif_threshold": vif_threshold, "corr_threshold": corr_threshold})
            if response.status_code == 200:
                result = response.json()
                vif_data = pd.DataFrame(result["vif_data"])
                multicollinear_pairs = result["multicollinear_pairs"]

                st.write("VIF Data:")
                st.dataframe(vif_data)

                st.write("Multicollinear Pairs:")
                st.write(multicollinear_pairs)
            else:
                st.error("Failed to identify high covariance features.")

        if st.checkbox("Feature Importance"):
            target_variable = st.text_input("Enter the target variable:")
            if target_variable:
                response = requests.post(f"{backend_url}/feature-importance/", files=files,
                                         params={"target_variable": target_variable})
                if response.status_code == 200:
                    feature_importance = pd.DataFrame(response.json())
                    st.write("Feature Importance Scores:")
                    st.dataframe(feature_importance)
                else:
                    st.error("Failed to calculate feature importance.")

        if st.checkbox("Group Features and Evaluate Performance"):
            target_variable = st.text_input("Enter the target variable for grouping:")
            if target_variable:
                response = requests.post(f"{backend_url}/group-features/", files=files,
                                         params={"target_variable": target_variable})
                if response.status_code == 200:
                    best_grouped_features = response.json()
                    st.write("Grouped Features:")
                    st.dataframe(best_grouped_features)
                else:
                    st.error("Failed to group features.")

    with tab3:
        if st.checkbox("Bin Numerical Columns"):
            response = requests.post(f"{backend_url}/bin-numerical-columns/", files=files)

            if response.status_code == 200:
                visualizations = response.json()
                st.write("Binned Distributions:")
                for column, image_base64 in visualizations.items():
                    st.write(f"Binned Distribution for {column}")
                    image_data = base64.b64decode(image_base64)
                    st.image(image_data, use_container_width=True)
            else:
                st.error("Failed to bin numerical columns.")

        if st.checkbox("Sample Numerical Columns"):
            response = requests.post(f"{backend_url}/sample-numerical-columns/", files=files)

            if response.status_code == 200:
                result = response.json()
                visualizations = result["visualizations"]
                no_sampling_required = result["no_sampling_required"]

                if visualizations:
                    st.write("Sampling Results:")
                    for column, image_base64 in visualizations.items():
                        st.write(f"Sampling for column: {column}")
                        image_data = base64.b64decode(image_base64)
                        st.image(image_data, use_container_width=True)

                if no_sampling_required:
                    st.write(f"No sampling required for columns: {', '.join(no_sampling_required)}")
            else:
                st.error("Failed to sample numerical columns.")


        if st.checkbox("Apply Dimensionality Reduction"):
            target_variable = st.text_input("Enter the target variable")
            if target_variable:
                response = requests.post(
                    f"{backend_url}/apply-dimensionality-reduction/",
                    files=files,
                    params={"target_column": target_variable}
                )

                if response.status_code == 200:
                    result = response.json()

                    st.write(result["message"])

                    if "explained_variance" in result:
                        st.write("Explained Variance Ratio for each Principal Component:")
                        st.bar_chart(result["explained_variance"])

                    if "cumulative_variance_image" in result:
                        image_data = base64.b64decode(result["cumulative_variance_image"])
                        st.image(image_data, use_container_width=True)
                else:
                    st.error("Failed to apply dimensionality reduction.")

    with tab4:
        if st.checkbox("Train and Compare Models"):
            target_variable = st.text_input("Enter the target variable-")
            if target_variable:
                response = requests.post(
                    f"{backend_url}/train-and-compare-models/",
                    files=files,
                    params={"target_variable": target_variable}
                )

                if response.status_code == 200:
                    result = response.json()
                    best_metrics = result["best_metrics"]
                    best_model_name = result["best_model_name"]

                    st.write(f"Best Model: {best_model_name}")

                    df_results = pd.DataFrame(best_metrics).T
                    st.write("Model Evaluation Results:")
                    st.table(df_results)

                    # Prepare data for plotting
                    accuracy_df = pd.DataFrame({
                        "Model": list(best_metrics.keys()),
                        "Accuracy": [best_metrics[model]["Accuracy"] for model in best_metrics]
                    }).set_index("Model")

                    precision_df = pd.DataFrame({
                        "Model": list(best_metrics.keys()),
                        "Precision": [best_metrics[model]["Precision"] for model in best_metrics]
                    }).set_index("Model")

                    recall_df = pd.DataFrame({
                        "Model": list(best_metrics.keys()),
                        "Recall": [best_metrics[model]["Recall"] for model in best_metrics]
                    }).set_index("Model")

                    f1_df = pd.DataFrame({
                        "Model": list(best_metrics.keys()),
                        "F1 Score": [best_metrics[model]["F1 Score"] for model in best_metrics]
                    }).set_index("Model")

                    # Create tabs for each metric
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

                else:
                    st.error("Failed to train and compare models.")