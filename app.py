# -*- coding: utf-8 -*-

# --- LIBRERIE PRINCIPALI ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    page_title="Dashboard Predittiva Intelligente",
    page_icon="🤖",
    layout="wide"
)

# --- CSS PERSONALIZZATO ---
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONE PER CARICARE I DATI ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Cerca di identificare colonne ID comuni e metterle come indice per non usarle nel modello
    possible_ids = ['ID', 'id', 'Id', 'CLIENT_ID', 'Client_ID', 'Customer_ID']
    for col in possible_ids:
        if col in df.columns:
            df = df.set_index(col)
            break
    return df

# --- LOGICA PRINCIPALE ---
st.title("🤖 Dashboard Predittiva Intelligente")
st.markdown("""
Carica i tuoi dati storici, addestra l'intelligenza artificiale e ottieni **risposte chiare** 
su nuovi dati (es. "Pagherà o no?", "Rinnoverà?", "È frode?").
""")

# --- STATE MANAGEMENT ---
if 'model_trained' not in st.session_state: st.session_state.model_trained = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Carica i Tuoi Dati")
    uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")
    
    # SEZIONE PERSONALIZZAZIONE (Appare dopo il training)
    if st.session_state.model_trained:
        st.divider()
        st.header("🛠️ Personalizza Risposte")
        st.info("Definisci i nomi delle classi per il risultato finale.")
        
        label_mapping = {}
        for cls in st.session_state.classes:
            # Default intelligente
            default_text = "Positivo / Paga" if str(cls) in ['0', 'No', 'Paid', 'OK'] else "Negativo / Default"
            user_label = st.text_input(f"Significato classe '{cls}':", value=f"Classe {cls}")
            label_mapping[cls] = user_label
        
        st.session_state.label_mapping = label_mapping

if uploaded_file is not None:
    df_to_use = load_data(uploaded_file)

    # --- 2. ANALISI DATI (EDA) ---
    with st.expander("📊 Clicca per vedere i Dati e le Statistiche", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Anteprima")
            st.dataframe(df_to_use.head())
        with col2:
            st.subheader("Statistiche")
            st.dataframe(df_to_use.describe())

    # --- 4. ADDESTRAMENTO ---
    st.header("🧠 Addestramento Modello")
    col_tr1, col_tr2 = st.columns([1, 3])
    
    with col_tr1:
        target_variable = st.selectbox("Cosa vuoi prevedere?", options=df_to_use.columns)
        train_btn = st.button("🚀 Avvia Addestramento", type="primary")

    if train_btn and target_variable:
        with st.spinner('Analisi dei pattern in corso...'):
            X = df_to_use.drop(columns=[target_variable])
            y = df_to_use[target_variable]
            
            # Identifica tipi di colonne
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            
            # Pipeline
            num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_trans, numeric_features), 
                ('cat', cat_trans, categorical_features)
            ])
            
            model = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
            
            # Train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            
            # Save Session
            st.session_state.model = model
            st.session_state.X_train_columns = X_train.columns
            st.session_state.feature_types = X_train.dtypes # Salviamo i tipi per formattare l'input dopo
            st.session_state.y_test = y_test
            st.session_state.y_pred = model.predict(X_test)
            st.session_state.classes = model.classes_
            st.session_state.model_trained = True
            
            # Init mapping
            if 'label_mapping' not in st.session_state:
                st.session_state.label_mapping = {c: str(c) for c in model.classes_}
            
            st.success("✅ Modello pronto!")
            st.rerun()

    # --- 5. SIMULATORE (INPUT CORRETTO) ---
    st.divider()
    st.header("🔮 Simulatore Nuovi Dati")
    
    if st.session_state.model_trained:
        st.write("Inserisci i parametri. **Nota:** I campi interi (es. ID, Età) non hanno decimali.")
        
        input_data = {}
        with st.form("prediction_form"):
            cols = st.columns(3)
            for i, col_name in enumerate(st.session_state.X_train_columns):
                with cols[i % 3]:
                    col_type = st.session_state.feature_types[col_name]
                    
                    # LOGICA MIGLIORATA PER NUMERI
                    if pd.api.types.is_numeric_dtype(col_type):
                        # Se è un INTERO (int64) o sembra un ID/Età
                        if pd.api.types.is_integer_dtype(col_type) or (df_to_use[col_name].fillna(0) % 1 == 0).all():
                            # Usa median come default, ma converti a INT
                            default_val = int(df_to_use[col_name].median()) if not pd.isna(df_to_use[col_name].median()) else 0
                            # step=1 e format="%d" forzano l'assenza di decimali
                            input_data[col_name] = st.number_input(col_name, value=default_val, step=1, format="%d")
                        else:
                            # Se è un FLOAT (soldi, percentuali)
                            default_val = float(df_to_use[col_name].median())
                            input_data[col_name] = st.number_input(col_name, value=default_val, format="%.2f")
                    else:
                        # Se è Categorico (Testo)
                        opts = df_to_use[col_name].dropna().unique().tolist()
                        input_data[col_name] = st.selectbox(col_name, options=opts)
            
            predict_btn = st.form_submit_button("🔍 Analizza Caso", type="primary", use_container_width=True)

        if predict_btn:
            # Calcolo
            df_new = pd.DataFrame([input_data])[st.session_state.X_train_columns]
            pred_raw = st.session_state.model.predict(df_new)[0]
            probs = st.session_state.model.predict_proba(df_new)[0]
            
            class_idx = list(st.session_state.classes).index(pred_raw)
            confidence = probs[class_idx]
            human_label = st.session_state.label_mapping.get(pred_raw, str(pred_raw))
            
            # --- CONCLUSIONE BUSINESS ---
            st.markdown("---")
            st.subheader("📢 Verdetto Finale")
            
            # Colore dinamico in base alla confidenza
            color_border = "#4CAF50" if confidence > 0.7 else "#FFC107"
            if confidence < 0.55: color_border = "#F44336"

            st.markdown(f"""
            <div style="
                background-color: #ffffff; 
                padding: 25px; 
                border-radius: 15px; 
                border-left: 10px solid {color_border}; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            ">
                <h4 style="margin:0; color: #666;">Previsione IA:</h4>
                <p style="font-size: 32px; font-weight: bold; color: #2c3e50; margin: 10px 0;">
                    👉 {human_label}
                </p>
                <hr style="border: 0; border-top: 1px solid #eee;">
                <p style="margin-bottom:0; font-size: 16px;">
                    Affidabilità della previsione: <strong>{confidence:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Spiegazione per NON tecnici
            if confidence > 0.75:
                st.success(f"✅ **Alta Sicurezza:** L'IA è molto sicura che il risultato sia '{human_label}'. Puoi procedere con fiducia.")
            elif confidence > 0.55:
                st.warning(f"⚖️ **Sicurezza Moderata:** L'IA propende per '{human_label}', ma la situazione non è netta. Controlla altri fattori.")
            else:
                st.error(f"⚠️ **Incertezza:** L'IA non è sicura (circa 50/50). Si consiglia vivamente una revisione umana manuale.")

    else:
        st.info("☝️ Inizia caricando un file CSV dalla barra laterale.")