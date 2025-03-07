import base64
import streamlit as st
import requests
import pandas as pd

# backend_url_info = "http://localhost:30003"
# backend_url_fe = "http://localhost:30005"
# backend_url_mp = "http://localhost:30007"
# backend_url_me = "http://localhost:30009"

backend_url = "http://localhost:8000"

# selected_section = st.sidebar.radio("Select Section",
#                                         ["Info", "Feature Selection", "Processing", "Model Execution"])

# Function to set the query parameter
def set_page(page_name):
    st.query_params["page"] = page_name

# Function to get the current page from query parameters
def get_page():
    query_params = st.query_params
    return query_params.get("page", [None])

def display_hyperparameter_controls(model_name, current_params):
    # st.write(f"Adjust Hyperparameters for {model_name}")
    adjusted_params = {}

    for param_name, param_value in current_params.items():
        if isinstance(param_value, int):
            adjusted_params[param_name] = st.number_input(f"{model_name} - {param_name}", value=param_value, step=1)
        elif isinstance(param_value, float):
            adjusted_params[param_name] = st.number_input(f"{model_name} - {param_name}", value=param_value, step=0.01)
        elif isinstance(param_value, str):
            adjusted_params[param_name] = st.text_input(f"{model_name} - {param_name}", value=param_value)
        # Add more types if necessary

    return adjusted_params


def retrain_models(hyperparameters, train_file_id, test_file_id, target_variable):
    payload = {
        "hyperparameters": hyperparameters,
        "train_file_id": train_file_id,
        "test_file_id": test_file_id,
        "target_variable": target_variable
    }

    response = requests.post(f"{backend_url}/retrain-models/", json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to retrain models: {response.status_code} - {response.text}")
        return None

# with st.sidebar.expander("Info", expanded=True):
#     st.button("Choose data source", on_click=set_page, args=("source_selection",))
#     st.button("Stat Analysis", on_click=set_page, args=("stat_analysis",))
#     st.button("Compute Correlation", on_click=set_page, args=("compute_corr",))
#
# with st.sidebar.expander("Feature Selection", expanded=False):
#     st.button("High Covariance Features", on_click=set_page, args=("high_cov",))
#     st.button("Feature Importance", on_click=set_page, args=("feature_importance",))
#     st.button("Group Features and Evaluate Performance", on_click=set_page, args=("group_features",))
#
# with st.sidebar.expander("Data Engg", expanded=False):
#     st.button("Bin Numerical Columns", on_click=set_page, args=("binning",))
#     st.button("Sample Numerical Columns", on_click=set_page, args=("sampling",))
#     st.button("Apply Dimensionality Reduction", on_click=set_page, args=("dimensionality_reduction",))
#
# with st.sidebar.expander("Model Building", expanded=False):
#     st.button("Execute Model", on_click=set_page, args=("build_model",))

with st.sidebar.expander("Info", expanded=True):
    if st.checkbox("Choose data source", True):
        set_page("source_selection")
    if st.checkbox("Stat Analysis"):
        set_page("stat_analysis")
    if st.checkbox("Compute Correlation"):
        set_page("compute_corr")

with st.sidebar.expander("Feature Selection", expanded=False):
    if st.checkbox("High Covariance Features"):
        set_page("high_cov")
    if st.checkbox("Feature Importance"):
        set_page("feature_importance")
    if st.checkbox("Group Features and Evaluate Performance"):
        set_page("group_features")

with st.sidebar.expander("Data Engg", expanded=False):
    if st.checkbox("Bin Numerical Columns"):
        set_page("binning")
    if st.checkbox("Sample Numerical Columns"):
        set_page("sampling")
    if st.checkbox("Apply Dimensionality Reduction"):
        set_page("dimensionality_reduction")

with st.sidebar.expander("Model Building", expanded=False):
    if st.checkbox("Execute Model"):
        set_page("build_model")

# Determine which page to display
current_page = get_page()

# Main content area
if current_page == 'source_selection':
    st.title("Source Selection")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if "file_id" not in st.session_state:
        if uploaded_file is not None:
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
        if "dataset_info" not in st.session_state:
            dataset_info_response = requests.post(f"{backend_url}/dataset-info/", params={"file_id": file_id})
            if dataset_info_response.status_code == 200:
                dataset_info_content = dataset_info_response.json()
                st.session_state["dataset_info"] = dataset_info_content["data"]
            else:
                st.error(
                    f"Failed to get dataset info: {dataset_info_response.status_code} - {dataset_info_response.text}")

        if "dataset_info" in st.session_state:
            df_sample = pd.DataFrame(st.session_state["dataset_info"])
            st.write(df_sample)

        if st.checkbox("Split Dataset"):
            file_id = st.session_state["file_id"]
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.3)
            if st.button("Split"):
                response = requests.post(f"{backend_url}/split-dataset/", params={"file_id": file_id, "test_size": test_size})
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["train_file_id"] = result["train_file_id"]
                    st.session_state["test_file_id"] = result["test_file_id"]
                    st.write("Dataset split successfully.")
                else:
                    st.error("Failed to split dataset.")

        if st.checkbox("Select Target Feature"):
            df_sample = pd.DataFrame(st.session_state["dataset_info"])
            options = ["Select a column..."] + list(df_sample.columns)
            target_feature_selected = st.selectbox("Select Target Variable", options)
            if target_feature_selected != "Select a column...":
                st.session_state["target"] = target_feature_selected
                st.write("Target feature set successfully!")

elif current_page == 'stat_analysis':
    st.title("Stat Analysis")
    if "train_file_id" in st.session_state:
        if "stat_analysis" not in st.session_state:
            response = requests.post(f"{backend_url}/stat-analysis/", params={"file_id": st.session_state["train_file_id"]})
            if response.status_code == 200:
                st.session_state["stat_analysis"] = response.json()
            else:
                st.error("Failed to perform statistical analysis.")

        if "stat_analysis" in st.session_state:
            description = pd.DataFrame(st.session_state["stat_analysis"]["description"]).T
            column_details = pd.DataFrame(st.session_state["stat_analysis"]["column_details"])
            st.write("STAT ANALYSIS:")
            st.write(description)
            st.write("COLUMN DETAILS:")
            st.table(column_details)
    else:
        st.warning("Please split the dataset first.")

elif current_page == 'compute_corr':
    st.title("Compute Correlation")
    if "train_file_id" in st.session_state:
        if "correlation_matrix" not in st.session_state:
            response = requests.post(f"{backend_url}/correlation-matrix/", params={"file_id": st.session_state["train_file_id"]})
            if response.status_code == 200:
                st.session_state["correlation_matrix"] = response.json()["top_correlations"]
            else:
                st.error("Failed to compute correlation matrix.")

        if "correlation_matrix" in st.session_state:
            top_correlations = pd.DataFrame(st.session_state["correlation_matrix"])
            st.write("Top Correlated Columns:")
            st.table(top_correlations)
    else:
        st.warning("Please split the dataset first.")

elif current_page == 'high_cov':
    st.title("High Covariance Features")
    if "train_file_id" in st.session_state:
        vif_threshold = st.number_input("VIF Threshold", value=10.0)
        corr_threshold = st.number_input("Correlation Threshold", value=0.8)
        cache_key = f"high_covariance_{vif_threshold}_{corr_threshold}"

        if cache_key not in st.session_state:
            response = requests.post(f"{backend_url}/high-covariance-features/",
                                     params={"file_id": st.session_state["train_file_id"], "vif_threshold": vif_threshold,
                                             "corr_threshold": corr_threshold})
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
    else:
        st.warning("Please split the dataset first.")

elif current_page == 'feature_importance':
    st.title("Feature Importance")
    if "train_file_id" in st.session_state and "target" in st.session_state:
        target_variable = st.session_state["target"]
        cache_key = f"feature_importance_{target_variable}"
        if cache_key not in st.session_state:
            response = requests.post(f"{backend_url}/feature-importance/",
                                     params={"file_id": st.session_state["train_file_id"], "target_variable": target_variable})
            if response.status_code == 200:
                st.session_state[cache_key] = response.json()
            else:
                st.error("Failed to calculate feature importance.")

        if cache_key in st.session_state:
            feature_importance_data = pd.DataFrame(st.session_state[cache_key])
            st.write("Feature Importance Scores:")
            st.dataframe(feature_importance_data)
    else:
        st.warning("Please split the dataset and select a target feature first.")

elif current_page == 'group_features':
    st.title("Group Features and Evaluate Performance")
    if "train_file_id" in st.session_state and "target" in st.session_state:
        target_variable = st.session_state["target"]
        cache_key = f"group_features_{target_variable}"
        if cache_key not in st.session_state:
            response = requests.post(f"{backend_url}/group-features/",
                                     params={"file_id": st.session_state["train_file_id"], "target_variable": target_variable})
            if response.status_code == 200:
                st.session_state[cache_key] = response.json()
            else:
                st.error("Failed to group features.")

        if cache_key in st.session_state:
            best_grouped_features = st.session_state[cache_key]
            st.write("Grouped Features:")
            st.dataframe(best_grouped_features)
    else:
        st.warning("Please split the dataset and select a target feature first.")

elif current_page == 'binning':
    st.title("Binning")
    if "train_file_id" in st.session_state:
        if "bin_numerical" not in st.session_state:
            response = requests.post(f"{backend_url}/bin-numerical-columns/",
                                     params={"file_id": st.session_state["train_file_id"]})
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
    else:
        st.warning("Please split the dataset first.")

elif current_page == 'sampling':
    st.title("Sampling")
    if "train_file_id" in st.session_state:
        # Let users select the sample fraction
        sample_fraction = st.slider("Sample Fraction", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

        if st.button("Sample Data"):
            response = requests.post(f"{backend_url}/sample-numerical-columns/",
                                     params={"file_id": st.session_state["train_file_id"],
                                             "sample_fraction": sample_fraction})
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
    else:
        st.warning("Please split the dataset first.")

elif current_page == 'dimensionality_reduction':
    st.title("Dimensionality Reduction")
    if "train_file_id" in st.session_state and "target" in st.session_state:
        target_variable = st.session_state["target"]
        cache_key = f"dimensionality_reduction_{target_variable}"
        if cache_key not in st.session_state:
            response = requests.post(f"{backend_url}/apply-dimensionality-reduction/",
                                     params={"file_id": st.session_state["train_file_id"], "target_column": target_variable})
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
    else:
        st.warning("Please split the dataset and select a target feature first.")


elif current_page == 'build_model':
    st.title("Model Building")
    if "train_file_id" in st.session_state and "test_file_id" in st.session_state and "target" in st.session_state:
        target_variable = st.session_state["target"]
        cache_key = f"train_and_compare_{target_variable}"
        if cache_key not in st.session_state:
            response = requests.post(f"{backend_url}/train-and-compare-models/",
                                     params={"train_file_id": st.session_state["train_file_id"],
                                             "test_file_id": st.session_state["test_file_id"],
                                             "target_variable": target_variable})
            if response.status_code == 200:
                st.session_state[cache_key] = response.json()
                result = st.session_state[cache_key]
                st.session_state["best_metrics"] = result["best_metrics"]
                st.session_state["best_model_name"] = result["best_model_name"]
                st.session_state["best_params"] = result["best_params"]
                st.session_state["model_params"] = result["model_params"]  # Retrieve all models' hyperparameters

            else:
                st.error("Failed to train and compare models.")

        if cache_key in st.session_state:
            st.subheader("Best Model Information")
            st.markdown(f"**Best Model:** {st.session_state['best_model_name']}")
            st.markdown(f"**Best Hyperparameters:**")
            st.json(st.session_state["best_params"])

            # Display hyperparameters for each model
            st.markdown("**Hyperparameters for Each Model**")
            for model_name, params in st.session_state["model_params"].items():
                st.markdown(f"{model_name}:")
                st.json(params)

            best_metrics = st.session_state["best_metrics"]

            df_results = pd.DataFrame(best_metrics).T
            st.subheader("Model Evaluation Results")
            st.table(df_results)

            # Plotting metrics for all models
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

            st.write("Hyperparameter Tuning")
            adjusted_params = {}
            for model_name, params in st.session_state["model_params"].items():
                with st.expander(f"Adjust Hyperparameters for {model_name}", expanded=False):
                    adjusted_params[model_name] = display_hyperparameter_controls(model_name, params)

            if st.button("Retrain Models with Adjusted Hyperparameters"):
                new_result = retrain_models(adjusted_params, st.session_state["train_file_id"],
                                            st.session_state["test_file_id"], target_variable)
                if new_result:
                    st.session_state[cache_key] = new_result  # Update session state with new results

                    # Display updated results
                    st.write("Updated Model Evaluation Results:")
                    updated_metrics = new_result["best_metrics"]
                    df_results = pd.DataFrame(updated_metrics).T
                    st.table(df_results)

                    # Display updated hyperparameters for the best model
                    best_model_name = new_result["best_model_name"]
                    param = new_result["param"]
                    st.write(f"Best Model: {best_model_name}")
                    st.write("Best Hyperparameters:", param)
    else:
        st.warning("Please complete all previous steps before building the model.")