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

# --- CSS PERSONALIZZATO PER MIGLIORARE LA LEGGIBILITÀ ---
st.markdown("""
<style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONE PER CARICARE I DATI CON CACHING ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'ID' in df.columns:
        df = df.set_index('ID')
    return df

# --- LOGICA PRINCIPALE ---
st.title("🤖 Dashboard Predittiva Intelligente")
st.markdown("""
Carica i tuoi dati storici, addestra l'intelligenza artificiale e ottieni **risposte chiare** 
su nuovi clienti (es. "Pagherà o no?").
""")

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'model_trained' not in st.session_state: st.session_state.model_trained = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Carica i Tuoi Dati")
    uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")
    
    # SEZIONE DI PERSONALIZZAZIONE LABELS (appare solo dopo training)
    if st.session_state.model_trained:
        st.divider()
        st.header("🛠️ Personalizza Risposte")
        st.info("Assegna un nome comprensibile alle classi per il verdetto finale.")
        
        # Creiamo un dizionario per mappare le classi tecniche in nomi umani
        label_mapping = {}
        for cls in st.session_state.classes:
            # Valore di default
            default_text = "Paga Regolarmente" if str(cls) in ['0', 'No', 'Paid'] else "Non Paga / Default"
            user_label = st.text_input(f"Cosa significa la classe '{cls}'?", value=f"Classe {cls}")
            label_mapping[cls] = user_label
        
        st.session_state.label_mapping = label_mapping

# Esecuzione solo se un file è caricato
if uploaded_file is not None:
    df_to_use = load_data(uploaded_file)

    # --- SEZIONE 2: ANALISI ESPLORATIVA (EDA) ---
    with st.expander("📊 Clicca per vedere l'Analisi dei Dati (EDA)", expanded=False):
        col_eda_1, col_eda_2 = st.columns(2)
        with col_eda_1:
            st.subheader("Anteprima Dati")
            st.dataframe(df_to_use.head())
        with col_eda_2:
            st.subheader("Statistiche")
            st.dataframe(df_to_use.describe())
        
        # Correlazione
        numeric_df = df_to_use.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.subheader("Mappa delle Correlazioni")
            fig_corr = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- SEZIONE 4: MODELING PREDITTIVO ---
    st.header("🧠 Addestramento Modello")
    
    col_train_1, col_train_2 = st.columns([1, 3])
    
    with col_train_1:
        target_variable = st.selectbox("Qual è la colonna da prevedere?", options=df_to_use.columns)
        train_btn = st.button("🚀 Avvia Addestramento", use_container_width=True)

    if train_btn and target_variable:
        with st.spinner('L\'IA sta imparando dai tuoi dati...'):
            # Preparazione X e y
            X = df_to_use.drop(columns=[target_variable])
            y = df_to_use[target_variable]
            
            # Pipeline di Preprocessing e Modello
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features), 
                ('cat', categorical_transformer, categorical_features)
            ])
            
            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
            
            # Split e Fit
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)
            
            # Salvataggio stato
            st.session_state.model = model_pipeline
            st.session_state.X_train_columns = X_train.columns
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.classes = model_pipeline.classes_
            st.session_state.model_trained = True
            
            # Inizializza mapping di default se non esiste
            if 'label_mapping' not in st.session_state:
                st.session_state.label_mapping = {c: str(c) for c in model_pipeline.classes_}
            
            st.success("✅ Addestramento completato!")
            st.rerun()

    # --- SEZIONE RISULTATI TECNICI ---
    if st.session_state.model_trained:
        with st.expander("📈 Vedi Performance Tecniche (Accuratezza, Matrice Confusione)"):
            acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
            st.metric("Accuratezza del Modello", f"{acc:.1%}")
            
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred, labels=st.session_state.classes)
            fig_cm = px.imshow(cm, text_auto=True, title="Matrice di Confusione", labels=dict(x="Predetto", y="Reale"))
            st.plotly_chart(fig_cm, use_container_width=True)

    # --- SEZIONE 5: PREVISIONE PER NUOVO CLIENTE (SIMULATORE) ---
    st.divider()
    st.header("🔮 Simulatore: Il cliente pagherà?")
    
    if st.session_state.model_trained:
        st.write("Inserisci i dati del nuovo cliente qui sotto:")
        
        input_data = {}
        with st.form("prediction_form"):
            cols = st.columns(3)
            for i, column in enumerate(st.session_state.X_train_columns):
                with cols[i % 3]:
                    if pd.api.types.is_numeric_dtype(df_to_use[column]):
                        val = float(df_to_use[column].median()) if not pd.isna(df_to_use[column].median()) else 0.0
                        input_data[column] = st.number_input(column, value=val)
                    else:
                        opts = df_to_use[column].dropna().unique().tolist()
                        input_data[column] = st.selectbox(column, options=opts)
            
            predict_btn = st.form_submit_button("🔍 Analizza Cliente", type="primary", use_container_width=True)

        if predict_btn:
            # Elaborazione Previsione
            new_df = pd.DataFrame([input_data])[st.session_state.X_train_columns]
            prediction_raw = st.session_state.model.predict(new_df)[0]
            probs = st.session_state.model.predict_proba(new_df)[0]
            
            # Recupero indice e probabilità massima
            class_idx = list(st.session_state.classes).index(prediction_raw)
            confidence = probs[class_idx]
            
            # Recupero etichetta comprensibile (dal mapping)
            human_label = st.session_state.label_mapping.get(prediction_raw, str(prediction_raw))
            
            # --- CONCLUSIONE PER NON TECNICI ---
            st.markdown("---")
            st.subheader("📢 Verdetto Finale")
            
            # Logica colori
            bg_color = "#d4edda" if class_idx == 0 else "#f8d7da" # Verde se classe 0, Rosso se classe 1 (supposizione base)
            # Se l'utente ha rinominato le label, usiamo un colore neutro o basato sulla probabilità
            if confidence > 0.8:
                emoji_conf = "molto alta 🟢"
            elif confidence > 0.6:
                emoji_conf = "moderata 🟡"
            else:
                emoji_conf = "bassa (incerto) 🔴"

            # Box del risultato
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid #4CAF50; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                <h3 style="margin-top:0; color: #333;">Risultato dell'Analisi:</h3>
                <p style="font-size: 28px; font-weight: bold; color: #000; margin: 10px 0;">
                    👉 {human_label}
                </p>
                <p style="font-size: 16px; color: #555;">
                    L'intelligenza artificiale è sicura al <strong>{confidence:.1%}</strong> ({emoji_conf}) di questa previsione.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Spiegazione Testuale Semplice
            st.write("")
            st.markdown("### 📝 Cosa significa?")
            st.write(f"In base ai dati inseriti, il modello statistico ritiene che il comportamento più probabile di questo cliente corrisponda alla categoria **'{human_label}'**.")
            
            if confidence < 0.60:
                st.warning("⚠️ **Attenzione:** La confidenza del modello è bassa (< 60%). Si consiglia una verifica umana manuale prima di prendere decisioni critiche.")
            else:
                st.success("✅ **Nota:** Il modello mostra una confidenza solida. Puoi procedere secondo le procedure standard per questa categoria.")

    else:
        st.warning("⚠️ Carica un file CSV e clicca su 'Avvia Addestramento' per sbloccare il simulatore.")