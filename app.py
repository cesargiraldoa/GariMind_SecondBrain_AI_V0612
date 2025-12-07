import streamlit as st
import pandas as pd
import requests
import json
import io
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind Directo", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN INTELIGENTE (MÉTODO DIRECTO / SIN LIBRERÍA) ---
def analizar_api_directa(df, pregunta, api_key):
    """
    Se conecta directamente a la API de Google sin usar la librería problemática.
    Evita errores 404 y conflictos de versiones.
    """
    try:
        # 1. Preparar datos (Esquema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()

        # 2. Prompt
        prompt = f"""
        Actúa como experto en Pandas Python.
        DataFrame 'df' tiene estas columnas:
        {info_cols}
        
        Muestra:
        {muestra}
        
        Pregunta: "{pregunta}"
        
        TAREA:
        1. Genera código Python para responder usando 'df'.
        2. Guarda la respuesta en variable 'resultado'.
        3. Solo código, sin markdown.
        """

        # 3. LLAMADA DIRECTA A LA URL (Bypassing la librería)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            return f"Error de Google: {response.text}", ""

        # 4. Extraer el código de la respuesta JSON
        respuesta_json = response.json()
        try:
            texto_generado = respuesta_json['candidates'][0]['content']['parts'][0]['text']
            codigo = texto_generado.replace("```python", "").replace("```", "").strip()
        exceptKeyError:
            return "La IA no devolvió código válido.", ""

        # 5. Ejecutar código localmente
        local_vars = {'df': df, 'pd': pd}
        exec(codigo, globals(), local_vars)
        
        return local_vars.get('resultado', "Sin resultado"), codigo

    except Exception as e:
        return f"Error técnico: {str(e)}", ""

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_simple():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# PÁGINA 1: CEREBRO
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🧠 Cerebro (Conexión Directa)")
    st.info("💡 Usando API REST directa para máxima compatibilidad.")

    # API KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    # DATOS
    df = cargar_datos_simple()
    
    if not df.empty:
        st.success(f"Datos listos: {len(df):,} filas.")
        
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Conectando con Google..."):
                    res, cod = analizar_api_directa(df, pregunta, api_key)
                    
                    st.divider()
                    st.write("### 💡 Resultado:")
                    st.write(res)
                    
                    with st.expander("Ver código"):
                        st.code(cod, language='python')
            else:
                st.error("Falta API Key")
    else:
        st.warning("No hay datos cargados (Revisa SQL).")

# ==========================================
# REPORTES Y MAPA (Igual que siempre)
# ==========================================
elif pagina == "📊 Reportes":
    st.title("Reportes")
    df = cargar_datos_simple()
    if not df.empty:
        if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
             df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        st.bar_chart(df.groupby('Mes')['Valor'].sum())

elif pagina == "🗺️ Mapa":
    st.title("Mapa")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES", ttl=0))
    except:
        st.error("Error SQL")
