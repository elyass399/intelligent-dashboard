# -*- coding: utf-8 -*-

# --- LIBRERIE PRINCIPALI ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from io import BytesIO
from fpdf import FPDF  # <--- NUOVA IMPORTAZIONE
import datetime

# --- LIBRERIE DI SCIKIT-LEARN ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURAZIONE DELLA PAGINA ---
st.set_page_config(page_title="AI Business Dashboard Pro", page_icon="🚀", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONI UTILITY ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    possible_ids = ['ID', 'id', 'Id', 'CLIENT_ID', 'Client_ID', 'Customer_ID']
    for col in possible_ids:
        if col in df.columns:
            df = df.set_index(col)
            break
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# --- NUOVA FUNZIONE GENERAZIONE PDF ---
def create_pdf(input_data, label, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Intestazione
    pdf.cell(0, 10, "Report Analisi Predittiva IA", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generato il: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    # Sezione Verdetto
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "RISULTATO ANALISI:", ln=True)
    pdf.set_font("Arial", '', 12)
    
    # Colore verdetto (Semplificato per PDF in scala di grigi/nero)
    res_text = f"Previsione: {label}"
    conf_text = f"Livello di Confidenza: {confidence:.1%}"
    
    pdf.cell(0, 10, res_text, ln=True)
    pdf.cell(0, 10, conf_text, ln=True)
    pdf.ln(10)
    
    # Sezione Dati Input
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Dati Cliente Analizzati:", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Tabella dati
    for key, value in input_data.items():
        # Pulisci stringhe lunghe o formati
        val_str = str(value)
        pdf.cell(90, 8, f"{key}", border=1)
        pdf.cell(100, 8, f"{val_str}", border=1, ln=True)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Report generato automaticamente da AI Business Dashboard.", ln=True, align='C')
    
    # Ritorna il contenuto come stringa binaria
    return pdf.output(dest='S').encode('latin-1')

# --- LOGICA PRINCIPALE ---
st.title("🚀 AI Business Dashboard Pro")

# --- STATE MANAGEMENT ---
if 'model_trained' not in st.session_state: st.session_state.model_trained = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Dati & Configurazione")
    uploaded_file = st.file_uploader("Carica CSV Storico", type="csv")
    
    if st.session_state.model_trained:
        st.divider()
        st.header("💾 Salva Modello")
        model_buffer = BytesIO()
        joblib.dump(st.session_state.model, model_buffer)
        st.download_button("Download Modello (.pkl)", model_buffer, "modello_allenato.pkl")
        
        st.divider()
        st.header("🛠️ Personalizza Label")
        label_mapping = {}
        for cls in st.session_state.classes:
            default = "Positivo / Paga" if str(cls) in ['0', 'No', 'Paid'] else "Negativo / Default"
            user_label = st.text_input(f"Classe '{cls}':", value=f"Classe {cls}")
            label_mapping[cls] = user_label
        st.session_state.label_mapping = label_mapping

if uploaded_file is not None:
    df_to_use = load_data(uploaded_file)

    # --- 2. EDA ---
    with st.expander("📊 Step 1: Esplora i Dati", expanded=False):
        st.dataframe(df_to_use.head())

    # --- 4. TRAINING ---
    st.header("🧠 Step 2: Addestramento IA")
    col_tr1, col_tr2 = st.columns([1, 3])
    with col_tr1:
        target_variable = st.selectbox("Target", options=df_to_use.columns)
        train_btn = st.button("🚀 Avvia Training", type="primary")

    if train_btn and target_variable:
        with st.spinner('Training in corso...'):
            X = df_to_use.drop(columns=[target_variable])
            y = df_to_use[target_variable]
            
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            
            num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer(transformers=[('num', num_trans, numeric_features), ('cat', cat_trans, categorical_features)])
            model = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            
            st.session_state.model = model
            st.session_state.X_train_columns = X_train.columns
            st.session_state.feature_types = X_train.dtypes
            st.session_state.y_test = y_test
            st.session_state.y_pred = model.predict(X_test)
            st.session_state.classes = model.classes_
            st.session_state.model_trained = True
            
            if 'label_mapping' not in st.session_state: st.session_state.label_mapping = {c: str(c) for c in model.classes_}
            st.success("✅ Modello Addestrato!")
            st.rerun()

    # --- DASHBOARD PERFORMANCE ---
    if st.session_state.model_trained:
        with st.expander("📈 Performance & Importanza Fattori", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
                st.metric("Accuracy", f"{acc:.1%}")
            with col2:
                try:
                    imps = st.session_state.model.named_steps['clf'].feature_importances_
                    feat_df = pd.DataFrame({'Feature': [f"F{i}" for i in range(len(imps))], 'Importanza': imps}).sort_values('Importanza', ascending=False).head(10)
                    st.plotly_chart(px.bar(feat_df, x='Importanza', y='Feature', orientation='h'), use_container_width=True)
                except: pass

    # --- 5. SIMULATORE ---
    st.divider()
    st.header("🔮 Step 3: Simulatore")
    
    if st.session_state.model_trained:
        tab1, tab2 = st.tabs(["👤 Cliente Singolo", "📂 Batch Upload"])
        
        # --- TAB 1: SINGOLO + PDF ---
        with tab1:
            st.write("Parametri simulazione:")
            input_data = {}
            with st.form("prediction_form"):
                cols = st.columns(3)
                for i, col in enumerate(st.session_state.X_train_columns):
                    with cols[i % 3]:
                        ctype = st.session_state.feature_types[col]
                        if pd.api.types.is_numeric_dtype(ctype):
                            if pd.api.types.is_integer_dtype(ctype):
                                val = int(df_to_use[col].median())
                                input_data[col] = st.number_input(col, value=val, step=1, format="%d")
                            else:
                                val = float(df_to_use[col].median())
                                input_data[col] = st.number_input(col, value=val, format="%.2f")
                        else:
                            opts = df_to_use[col].dropna().unique().tolist()
                            input_data[col] = st.selectbox(col, options=opts)
                
                predict_btn = st.form_submit_button("🔍 Analizza", type="primary")

            if predict_btn:
                df_new = pd.DataFrame([input_data])[st.session_state.X_train_columns]
                pred = st.session_state.model.predict(df_new)[0]
                prob = st.session_state.model.predict_proba(df_new)[0]
                
                idx = list(st.session_state.classes).index(pred)
                conf = prob[idx]
                label = st.session_state.label_mapping.get(pred, str(pred))
                
                color = "#4CAF50" if conf > 0.7 else "#F44336" if conf < 0.55 else "#FFC107"
                
                # Visualizzazione Risultato
                st.markdown(f"""
                <div style="background-color:white; padding:20px; border-left:10px solid {color}; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="color:#333; margin:0;">👉 {label}</h3>
                    <p>Confidenza: <strong>{conf:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # --- GENERAZIONE PDF ---
                st.write("")
                pdf_bytes = create_pdf(input_data, label, conf)
                st.download_button(
                    label="📄 Scarica Report PDF",
                    data=pdf_bytes,
                    file_name="report_cliente.pdf",
                    mime="application/pdf"
                )

        # --- TAB 2: BATCH ---
        with tab2:
            batch_file = st.file_uploader("Carica CSV Batch", type="csv", key="batch")
            if batch_file:
                df_batch = pd.read_csv(batch_file)
                if st.button("🚀 Elabora Tutto"):
                    preds = st.session_state.model.predict(df_batch[st.session_state.X_train_columns])
                    probs = st.session_state.model.predict_proba(df_batch[st.session_state.X_train_columns])
                    df_batch['Previsione'] = [st.session_state.label_mapping.get(p, str(p)) for p in preds]
                    df_batch['Confidenza'] = [max(p) for p in probs]
                    st.dataframe(df_batch.head())
                    st.download_button("📥 Scarica Risultati CSV", convert_df_to_csv(df_batch), "risultati_batch.csv", "text/csv")

    else:
        st.info("Addestra il modello per sbloccare.")