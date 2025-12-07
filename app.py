import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind Directo", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN INTELIGENTE (LA SOLUCIÓN) ---
def analizar_con_agente(df, pregunta, api_key):
    """
    Usa el modelo ESTÁNDAR (gemini-pro) para evitar errores de compatibilidad.
    """
    try:
        # Configurar API
        genai.configure(api_key=api_key)
        
        # 1. Preparar la estructura (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. Prompt para generar Python
        prompt = f"""
        Actúa como experto en Pandas Python.
        Tengo un DataFrame 'df' con estas columnas:
        {info_cols}
        
        Ejemplo de datos:
        {muestra}
        
        Usuario pregunta: "{pregunta}"
        
        TU TAREA:
        1. Genera código Python para responder usando 'df'.
        2. Guarda la respuesta en una variable llamada 'resultado'.
        3. NO uses markdown. Solo código puro.
        """
        
        # --- EL CAMBIO CLAVE: USAMOS EL MODELO CLÁSICO ---
        model = genai.GenerativeModel('gemini-pro') 
        
        response = model.generate_content(prompt)
        codigo = response.text.replace("```python", "").replace("```", "").strip()
        
        # 3. Ejecutar código
        local_vars = {'df': df, 'pd': pd}
        exec(codigo, globals(), local_vars)
        
        return local_vars.get('resultado', "Sin resultado"), codigo

    except Exception as e:
        return f"Error: {str(e)}", ""

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
    st.title("🧠 Cerebro (Modo Compatible)")
    
    # API KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    # DATOS
    df = cargar_datos_simple()
    
    if not df.empty:
        st.success(f"Datos listos: {len(df)} filas.")
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando..."):
                    # Usamos la función corregida
                    res, cod = analizar_con_agente(df, pregunta, api_key)
                    st.write("### 💡 Resultado:")
                    st.write(res)
            else:
                st.error("Falta API Key")
    else:
        st.error("Error cargando SQL")

# ==========================================
# RESTO DE PÁGINAS (REPORTES Y MAPA)
# ==========================================
elif pagina == "📊 Reportes":
    st.title("Reportes")
    # (Tu código de reportes sigue igual aquí, no afecta)
    
elif pagina == "🗺️ Mapa":
    st.title("Mapa")
    # (Tu código de mapa sigue igual aquí)
