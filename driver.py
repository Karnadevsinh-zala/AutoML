import base64
import streamlit as st
import requests
import pandas as pd

backend_url = "http://localhost:8000"

st.title("ML Pipeline")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and "file_id" not in st.session_state:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{backend_url}/upload-file/", files=files)
    if response.status_code == 200:
        result = response.json()
        file_id = result["file_id"]
        st.session_state["file_id"] = file_id
        st.write("File uploaded successfully.")
    else:
        st.error("Failed to upload file.")

if "file_id" in st.session_state:
    file_id = st.session_state["file_id"]
    tab1, tab2, tab3, tab4 = st.tabs(["Info", "Feature Selection", "Processing", "Model Execution"])

    #Data Info
    with tab1:
        if "dataset_info" not in st.session_state:
            dataset_info_response = requests.post(f"{backend_url}/dataset-info/", params={"file_id": file_id})
            if dataset_info_response.status_code == 200:
                dataset_info_content = dataset_info_response.json()
                st.session_state["dataset_info"] = dataset_info_content["data"]
            else:
                st.error(f"Failed to get dataset info: {dataset_info_response.status_code} - {dataset_info_response.text}")

        if "dataset_info" in st.session_state:
            df_sample = pd.DataFrame(st.session_state["dataset_info"])
            st.write(df_sample)

        if st.checkbox("Statistical Analysis"):
            if "stat_analysis" not in st.session_state:
                response = requests.post(f"{backend_url}/stat-analysis/", params={"file_id": file_id})
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["stat_analysis"] = {
                        "description": result["description"],
                        "column_details": result["column_details"]
                    }
                else:
                    st.error("Failed to perform statistical analysis.")

            if "stat_analysis" in st.session_state:
                description = pd.DataFrame(st.session_state["stat_analysis"]["description"]).T
                column_details = pd.DataFrame(st.session_state["stat_analysis"]["column_details"])
                st.write("STAT ANALYSIS:")
                st.write(description)
                st.write("COLUMN DETAILS:")
                st.table(column_details)

        if st.checkbox("Compute Correlation"):
            if "correlation_matrix" not in st.session_state:
                response = requests.post(f"{backend_url}/correlation-matrix/", params={"file_id": file_id})
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["correlation_matrix"] = result["top_correlations"]
                else:
                    st.error("Failed to compute correlation matrix.")

            if "correlation_matrix" in st.session_state:
                top_correlations = pd.DataFrame(st.session_state["correlation_matrix"])
                st.write("Top Correlated Columns:")
                st.table(top_correlations)

    #Feature Selection
    with tab2:
        if st.checkbox("High Covariance Features"):
            vif_threshold = st.number_input("VIF Threshold", value=10.0)
            corr_threshold = st.number_input("Correlation Threshold", value=0.8)
            cache_key = f"high_covariance_{vif_threshold}_{corr_threshold}"

            if cache_key not in st.session_state:
                response = requests.post(
                    f"{backend_url}/high-covariance-features/",
                    params={"file_id": file_id, "vif_threshold": vif_threshold, "corr_threshold": corr_threshold}
                )
                if response.status_code == 200:
                    st.session_state[cache_key] = response.json()
                else:
                    st.error("Failed to identify high covariance features.")

            if cache_key in st.session_state:
                result = st.session_state[cache_key]
                vif_data = pd.DataFrame(result["vif_data"])
                multicollinear_pairs = result["multicollinear_pairs"]

                st.write("VIF Data:")
                st.dataframe(vif_data)

                st.write("Multicollinear Pairs:")
                st.write(multicollinear_pairs)

        if st.checkbox("Feature Importance"):
            target_variable = st.text_input("Enter the target variable:")
            if target_variable:
                cache_key = f"feature_importance_{target_variable}"
                if cache_key not in st.session_state:
                    response = requests.post(
                        f"{backend_url}/feature-importance/",
                        params={"file_id": file_id, "target_variable": target_variable}
                    )
                    if response.status_code == 200:
                        st.session_state[cache_key] = response.json()
                    else:
                        st.error("Failed to calculate feature importance.")

                if cache_key in st.session_state:
                    feature_importance = pd.DataFrame(st.session_state[cache_key])
                    st.write("Feature Importance Scores:")
                    st.dataframe(feature_importance)

        if st.checkbox("Group Features and Evaluate Performance"):
            target_variable = st.text_input("Enter the target variable for grouping:")
            if target_variable:
                cache_key = f"group_features_{target_variable}"
                if cache_key not in st.session_state:
                    response = requests.post(
                        f"{backend_url}/group-features/",
                        params={"file_id": file_id, "target_variable": target_variable}
                    )
                    if response.status_code == 200:
                        st.session_state[cache_key] = response.json()
                    else:
                        st.error("Failed to group features.")

                if cache_key in st.session_state:
                    best_grouped_features = st.session_state[cache_key]
                    st.write("Grouped Features:")
                    st.dataframe(best_grouped_features)

    #Processing
    with tab3:
        if st.checkbox("Bin Numerical Columns"):
            if "bin_numerical" not in st.session_state:
                response = requests.post(f"{backend_url}/bin-numerical-columns/", params={"file_id": file_id})
                if response.status_code == 200:
                    st.session_state["bin_numerical"] = response.json()
                else:
                    st.error("Failed to bin numerical columns.")

            if "bin_numerical" in st.session_state:
                visualizations = st.session_state["bin_numerical"]
                st.write("Binned Distributions:")
                for column, image_base64 in visualizations.items():
                    st.write(f"Binned Distribution for {column}")
                    image_data = base64.b64decode(image_base64)
                    st.image(image_data, use_container_width=True)

        if st.checkbox("Sample Numerical Columns"):
            if "sample_numerical" not in st.session_state:
                response = requests.post(f"{backend_url}/sample-numerical-columns/", params={"file_id": file_id})
                if response.status_code == 200:
                    st.session_state["sample_numerical"] = response.json()
                else:
                    st.error("Failed to sample numerical columns.")

            if "sample_numerical" in st.session_state:
                result = st.session_state["sample_numerical"]
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

        if st.checkbox("Apply Dimensionality Reduction"):
            target_variable = st.text_input("Enter the target variable")
            if target_variable:
                cache_key = f"dimensionality_reduction_{target_variable}"
                if cache_key not in st.session_state:
                    response = requests.post(
                        f"{backend_url}/apply-dimensionality-reduction/",
                        params={"file_id": file_id, "target_column": target_variable}
                    )
                    if response.status_code == 200:
                        st.session_state[cache_key] = response.json()
                    else:
                        st.error("Failed to apply dimensionality reduction.")

                if cache_key in st.session_state:
                    result = st.session_state[cache_key]
                    st.write(result["message"])

                    if "explained_variance" in result:
                        st.write("Explained Variance Ratio for each Principal Component:")
                        st.bar_chart(result["explained_variance"])

                    if "cumulative_variance_image" in result:
                        image_data = base64.b64decode(result["cumulative_variance_image"])
                        st.image(image_data, use_container_width=True)

    #Model Execution
    with tab4:
        if st.checkbox("Train and Compare Models"):
            target_variable = st.text_input("Enter the target variable:", key="tcm")
            if target_variable:
                cache_key = f"train_and_compare_{target_variable}"
                if cache_key not in st.session_state:
                    response = requests.post(
                        f"{backend_url}/train-and-compare-models/",
                        params={"file_id": file_id, "target_variable": target_variable}
                    )
                    if response.status_code == 200:
                        st.session_state[cache_key] = response.json()
                    else:
                        st.error("Failed to train and compare models.")

                if cache_key in st.session_state:
                    result = st.session_state[cache_key]
                    best_metrics = result["best_metrics"]
                    best_model_name = result["best_model_name"]
                    best_params = result["best_params"]

                    st.write(f"Best Model: {best_model_name}")
                    st.write("Best Hyperparameters:", best_params)

                    df_results = pd.DataFrame(best_metrics).T
                    st.write("Model Evaluation Results:")
                    st.table(df_results)

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