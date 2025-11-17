# -*- coding: utf-8 -*-

# --- LIBRERIE PRINCIPALI ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# --- LIBRERIE DI SCIKIT-LEARN ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- CONFIGURAZIONE DELLA PAGINA ---
st.set_page_config(
    page_title="Intelligent Dashboard & Predictive Analytics",
    page_icon="🤖",
    layout="wide"
)

# --- FUNZIONE PER CARICARE I DATI CON CACHING ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Imposta l'ID come indice, poiché non è una feature predittiva
    if 'ID' in df.columns:
        df = df.set_index('ID')
    return df

# --- LOGICA PRINCIPALE ---
st.title("🤖 Intelligent Dashboard & Predictive Analytics")
st.write("Carica il tuo dataset, pulisci i dati, visualizza insight e costruisci un modello predittivo con spiegazioni chiare e utilizzabili.")

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'model_trained' not in st.session_state: st.session_state.model_trained = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Carica i Tuoi Dati")
    uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")
    st.markdown("**Esempio:** Prevedere il **rischio di mancato pagamento** dei clienti.")

# Esecuzione solo se un file è caricato
if uploaded_file is not None:
    df_to_use = load_data(uploaded_file)

    # --- SEZIONE 2: ANALISI ESPLORATIVA (EDA) ---
    st.header("2. Analisi Esplorativa dei Dati")
    st.dataframe(df_to_use.head())
    st.subheader("Statistiche Descrittive (dati numerici)")
    st.write(df_to_use.describe())
    
    # --- SEZIONE 4: MODELING PREDITTIVO ---
    st.header("4. Modeling Predittivo (Classificazione)")
    with st.sidebar:
        st.header("Configura Modello")
        target_variable = st.selectbox("Seleziona la Variabile Target", options=df_to_use.columns)
    
    if target_variable:
        st.write(f"**Variabile Target Selezionata:** `{target_variable}`")
        if st.button("Addestra Modello di Classificazione"):
            with st.spinner('Addestramento in corso...'):
                # Il modello viene addestrato direttamente su df_to_use
                X = df_to_use.drop(columns=[target_variable])
                y = df_to_use[target_variable]
                
                class_counts = y.value_counts()
                if class_counts.min() < 2:
                    st.error("Errore: Dati insufficienti per l'addestramento.")
                    st.stop()

                # Ora select_dtypes funzionerà correttamente, identificando le colonne con stringhe
                # come 'categorical' e le altre come 'numeric'.
                numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
                
                # Le pipeline rimangono le stesse, sono già progettate per gestire questo caso
                numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                
                preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
                
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model_pipeline.fit(X_train, y_train)
                
                # Salva i risultati nello stato della sessione
                y_pred = model_pipeline.predict(X_test)
                st.session_state.model = model_pipeline
                st.session_state.X_train_columns = X_train.columns
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.model_trained = True
                st.success("Modello addestrato con successo!")

    # --- SEZIONE RISULTATI ---
    if st.session_state.model_trained:
        st.subheader("Performance del Modello")
        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        st.metric("Accuracy", f"{accuracy:.2%}")
        # ... (Altre metriche e Matrice di Confusione)

    # --- SEZIONE 5: PREVISIONE PER NUOVO CLIENTE ---
    st.header("5. Prevedi per un Nuovo Cliente")
    if st.session_state.model_trained:
        st.info("Inserisci i dati del cliente per ottenere una previsione.")
        
        input_data = {}
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            for i, column in enumerate(st.session_state.X_train_columns):
                target_col = [col1, col2, col3][i % 3] 
                with target_col:
                    # Controlla se la colonna è numerica o categorica (contiene stringhe)
                    if pd.api.types.is_numeric_dtype(df_to_use[column]):
                        input_data[column] = st.number_input(label=f"{column}", value=float(df_to_use[column].median()))
                    else:
                        # Per le colonne con stringhe, prendi le opzioni uniche dal dataset
                        options = df_to_use[column].unique().tolist()
                        input_data[column] = st.selectbox(label=column, options=options)
            
            submit_button = st.form_submit_button(label="Ottieni Previsione")

        if submit_button:
            with st.spinner("Analisi in corso..."):
                # Non c'è più bisogno di riconvertire i dati, la pipeline del modello lo fa in automatico
                new_client_df = pd.DataFrame([input_data])[st.session_state.X_train_columns]
                prediction = st.session_state.model.predict(new_client_df)
                prediction_proba = st.session_state.model.predict_proba(new_client_df)
                
                st.subheader("Risultato della Previsione")
                
                if prediction[0] == 1:
                    st.error(f"**Previsione: Alto Rischio di Mancato Pagamento** (Probabilità: {prediction_proba[0][1]:.2%})")
                    st.write("Il cliente probabilmente **non riuscirà a effettuare il pagamento minimo richiesto** nel prossimo mese.")
                else:
                    st.success(f"**Previsione: Pagamento Regolare Atteso** (Probabilità: {prediction_proba[0][0]:.2%})")
                    st.write("Il modello prevede che il cliente **effettuerà regolarmente il suo pagamento** nel prossimo mese.")
    else:
        st.warning("Per favore, addestra un modello per abilitare questa funzionalità.")