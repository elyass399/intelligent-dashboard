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
# Use caching to prevent re-loading data on every interaction
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# --- Main Application Logic ---
st.title("🤖 Intelligent Dashboard & Predictive Analytics")
st.write("""
Upload your dataset, and this app will automatically perform an initial analysis, 
allow you to visualize the data, and build a predictive classification model.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.markdown("""
    **Example Use Case:**
    -  Upload credit data and predict `default payment next month`.
    """)

# Main panel logic starts here, only if a file is uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # --- Section 2: Exploratory Data Analysis (EDA) ---
    st.header("2. Exploratory Data Analysis")
    
    # Display a preview of the dataframe
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Display data types and missing values
    st.subheader("Data Types & Missing Values")
    # Use StringIO to capture the output of df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # --- Section 3: Data Visualization ---
    st.header("3. Data Visualization")
    
    # Allow user to select visualization type
    st.subheader("Interactive Plots")
    col1, col2 = st.columns(2)
    
    with col1:
        # Univariate analysis
        st.markdown("#### Univariate Analysis")
        column_to_plot = st.selectbox("Select a column for histogram", df.columns)
        if column_to_plot:
            fig_hist = px.histogram(df, x=column_to_plot, title=f"Histogram of {column_to_plot}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
    with col2:
        # Bivariate analysis
        st.markdown("#### Bivariate Analysis")
        x_axis = st.selectbox("Select X-axis for scatter plot", df.columns, index=0)
        y_axis = st.selectbox("Select Y-axis for scatter plot", df.columns, index=1)
        if x_axis and y_axis:
            fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {x_axis} vs {y_axis}")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No numeric columns found to generate a correlation heatmap.")


    # --- Section 4: Predictive Modeling ---
    st.header("4. Predictive Modeling (Classification)")

    with st.sidebar:
        st.header("Configure Model")
        target_variable = st.selectbox("Select Target Variable (what you want to predict)", options=df.columns)
    
    if target_variable:
        st.write(f"**Selected Target Variable:** `{target_variable}`")
        
        # Check if the target is suitable for classification
        if df[target_variable].nunique() > 20 or pd.api.types.is_numeric_dtype(df[target_variable]) and df[target_variable].nunique() > 2:
            st.warning("The selected target variable seems to be continuous or has many unique values. This tool is currently configured for classification tasks (a few distinct outcomes). Results may not be meaningful.")
        
        if st.button("Train Classification Model"):
            with st.spinner('Training model... This may take a moment.'):
                try:
                    # --- Data Preprocessing ---
                    # Separate features (X) and target (y)
                    X = df.drop(columns=[target_variable])
                    y = df[target_variable]
                    
                    # Identify numeric and categorical features
                    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
                    
                    # Create preprocessing pipelines for both numeric and categorical data
                    numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
                    
                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                    
                    # Create a preprocessor object using ColumnTransformer
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numeric_features),
                            ('cat', categorical_transformer, categorical_features)])
                    
                    # --- Model Training ---
                    # Create a pipeline that first preprocesses the data and then trains the model
                    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                   ('classifier', RandomForestClassifier(random_state=42))])
                    
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Train the model
                    model_pipeline.fit(X_train, y_train)
                    
                    # --- Evaluation ---
                    y_pred = model_pipeline.predict(X_test)
                    
                    st.subheader("Model Performance")
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Display metrics
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Accuracy", f"{accuracy:.2%}")
                    col_m2.metric("Precision", f"{precision:.2%}")
                    col_m3.metric("Recall", f"{recall:.2%}")
                    col_m4.metric("F1-Score", f"{f1:.2%}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted Label", y="True Label"),
                                       title="Confusion Matrix")
                    st.plotly_chart(fig_cm)
                    
                    # --- Feature Importance ---
                    st.subheader("Feature Importance")
                    
                    # Get feature names after one-hot encoding
                    try:
                        feature_names_raw = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                        all_feature_names = numeric_features + list(feature_names_raw)
                        
                        # Get importance scores
                        importances = model_pipeline.named_steps['classifier'].feature_importances_
                        
                        # Create a DataFrame for visualization
                        importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
                        importance_df = importance_df.sort_values(by='importance', ascending=False).head(20) # Top 20 features
                        
                        fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', 
                                         title="Top 20 Most Important Features")
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"Could not display feature importance. Error: {e}")

                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")

else:
    st.info("Awaiting for a CSV file to be uploaded.")