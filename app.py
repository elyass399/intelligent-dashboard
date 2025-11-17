# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# Scikit-learn for Preprocessing and Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Dashboard & Predictive Analytics",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Data Function ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# --- Main Application Logic ---
st.title("🤖 Intelligent Dashboard & Predictive Analytics")
st.write("""
Upload your dataset, perform data cleaning, visualize insights, 
and build a predictive model with actionable explanations.
""")

# Initialize session state
if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None
if 'model_trained' not in st.session_state: st.session_state.model_trained = False

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("**Example:** Predict payment default risk for credit clients.")

# Main panel logic starts here
if uploaded_file is not None:
    df_original = load_data(uploaded_file)
    
    # Determine which DataFrame to use
    if st.session_state.cleaned_df is not None:
        df_to_use = st.session_state.cleaned_df
        st.info("Displaying the cleaned dataset. To revert, please re-upload the original file.")
    else:
        df_to_use = df_original

    # --- Section 2: Exploratory Data Analysis (EDA) ---
    st.header("2. Exploratory Data Analysis")
    st.dataframe(df_to_use.head())
    st.subheader("Summary Statistics")
    st.write(df_to_use.describe())
    st.subheader("Data Types & Missing Values (Before Cleaning)")
    buffer = StringIO()
    df_original.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # --- Section 2.5: Interactive Data Cleaning ---
    st.header("2.5. Interactive Data Cleaning")
    missing_values = df_original.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        st.warning("Found columns with missing values (NaNs):")
        st.dataframe(missing_values.to_frame(name='Missing Count'))
        with st.expander("Choose a cleaning method"):
            cleaning_method = st.selectbox("Select how to handle NaNs", ["-- Select a method --", "Remove rows with missing values", "Fill missing values (Imputation)"])
            if cleaning_method == "Remove rows with missing values":
                if st.button("Apply Row Removal"):
                    cleaned_df = df_original.dropna()
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.model_trained = False
                    st.success(f"Removed rows with NaNs. New dataset has {len(cleaned_df)} rows.")
                    st.rerun()
            elif cleaning_method == "Fill missing values (Imputation)":
                numeric_cols_with_nan = df_original.select_dtypes(include=np.number).columns[df_original.select_dtypes(include=np.number).isnull().any()].tolist()
                if numeric_cols_with_nan:
                    numeric_fill = st.selectbox("How to fill numeric NaNs?", ["Median", "Mean", "Zero"])
                if st.button("Apply Imputation"):
                    cleaned_df = df_original.copy()
                    for col in numeric_cols_with_nan:
                        fill_value = 0
                        if numeric_fill == "Median": fill_value = cleaned_df[col].median()
                        elif numeric_fill == "Mean": fill_value = cleaned_df[col].mean()
                        cleaned_df[col].fillna(fill_value, inplace=True)
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.model_trained = False
                    st.success("Filled NaNs using the selected strategies.")
                    st.rerun()
    else:
        st.success("Great! Your dataset has no missing values.")
        
    # --- Section 3: Data Visualization ---
    st.header("3. Data Visualization")
    st.subheader("Interactive Plots")
    # (Visualization code remains the same)
    
    # --- Section 4: Predictive Modeling ---
    st.header("4. Predictive Modeling (Classification)")
    with st.sidebar:
        st.header("Configure Model")
        target_variable = st.selectbox("Select Target Variable", options=df_to_use.columns)
    
    if target_variable:
        st.write(f"**Selected Target Variable:** `{target_variable}`")
        if st.button("Train Classification Model"):
            with st.spinner('Training model...'):
                X = df_to_use.drop(columns=[target_variable])
                y = df_to_use[target_variable]
                
                # Safety check for insufficient data
                class_counts = y.value_counts()
                if class_counts.min() < 2:
                    st.error(f"Error: A class ('{class_counts.idxmin()}') has only {class_counts.min()} sample. Each class must have at least two samples for training.")
                    st.stop()

                numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
                numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model_pipeline.fit(X_train, y_train)
                y_pred = model_pipeline.predict(X_test)
                
                # Save everything to session state
                st.session_state.model = model_pipeline
                st.session_state.X_train_columns = X_train.columns
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.numeric_features = numeric_features
                st.session_state.categorical_features = categorical_features
                st.session_state.model_trained = True
                st.success("Model trained successfully!")

    # --- Results Section ---
    if st.session_state.model_trained:
        st.subheader("Model Performance")
        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        precision = precision_score(st.session_state.y_test, st.session_state.y_pred, average='weighted', zero_division=0)
        recall = recall_score(st.session_state.y_test, st.session_state.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(st.session_state.y_test, st.session_state.y_pred, average='weighted', zero_division=0)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{accuracy:.2%}")
        col_m2.metric("Precision", f"{precision:.2%}")
        col_m3.metric("Recall", f"{recall:.2%}")
        col_m4.metric("F1-Score", f"{f1:.2%}")
        
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        st.subheader("Executive Summary & Insights")
        st.markdown(f"The model predicts **payment default risk** with an accuracy of **{accuracy:.2%}**. The main goal should be to minimize 'False Negatives' (cases where the model predicted 'Paid' but the client defaulted), as these represent the most significant financial losses.")
        
        st.subheader("Feature Importance")
        try:
            model_pipeline = st.session_state.model
            importances = model_pipeline.named_steps['classifier'].feature_importances_
            
            ohe_feature_names = []
            if st.session_state.categorical_features and 'cat' in model_pipeline.named_steps['preprocessor'].named_transformers_:
                ohe_feature_names = list(model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(st.session_state.categorical_features))
            
            all_feature_names = st.session_state.numeric_features + ohe_feature_names
            
            importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
            importance_df = importance_df.sort_values(by='importance', ascending=False).head(15)
            st.session_state.importance_df = importance_df # Save for single prediction explanation
            
            fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title="Top 15 Most Important Features")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display feature importance. Error: {e}")
            
    # --- Section 5: Predict for a New Client ---
    st.header("5. Predict for a New Client")
    if st.session_state.model_trained:
        st.info("Enter the client's data below to get a real-time risk prediction.")
        
        input_data = {}
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            for i, column in enumerate(st.session_state.X_train_columns):
                target_col = [col1, col2, col3][i % 3]
                with target_col:
                    if column in st.session_state.numeric_features:
                        input_data[column] = st.number_input(label=f"{column}", value=float(df_to_use[column].median()))
                    else:
                        options = df_to_use[column].unique().tolist()
                        input_data[column] = st.selectbox(label=f"{column}", options=options, index=0)
            submit_button = st.form_submit_button(label="Get Prediction")

        if submit_button:
            with st.spinner("Analyzing client..."):
                new_client_df = pd.DataFrame([input_data])[st.session_state.X_train_columns]
                prediction = st.session_state.model.predict(new_client_df)
                prediction_proba = st.session_state.model.predict_proba(new_client_df)
                
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error(f"**Prediction: High Risk of Payment Default** (Probability: {prediction_proba[0][1]:.2%})")
                    st.write("Based on the data provided, the model predicts this client is likely to **miss their next payment**.")
                else:
                    st.success(f"**Prediction: Low Risk - Payment Expected** (Probability of Paying: {prediction_proba[0][0]:.2%})")
                    st.write("The model predicts this client will likely **make their next payment on time**.")

                st.subheader("Key Factors in this Prediction")
                if 'importance_df' in st.session_state:
                    top_features = st.session_state.importance_df.head(3)['feature'].tolist()
                    st.write("The most important general factors the model considers are:")
                    for feature in top_features:
                        original_feature_name = next((col for col in st.session_state.X_train_columns if feature.startswith(col)), feature)
                        value = new_client_df.iloc[0][original_feature_name]
                        st.markdown(f"- **{original_feature_name}**: The client's value for this key factor is `{value}`.")
    else:
        st.warning("Please train a model first to enable the client prediction feature.")