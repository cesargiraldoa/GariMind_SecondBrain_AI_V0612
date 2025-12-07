import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind Directo", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN INTELIGENTE (LA SOLUCIÓN A TUS ERRORES) ---
def analizar_con_agente(df, pregunta, api_key):
    """
    Genera código Python para responder preguntas sin enviar los datos brutos.
    Usa 'gemini-pro' para asegurar compatibilidad.
    """
    try:
        # Configuración de la API
        genai.configure(api_key=api_key)
        
        # 1. Sacamos la 'radiografía' de los datos (Solo estructura)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. Creamos el prompt para el Agente
        prompt = f"""
        Actúa como un experto en Python Pandas.
        Tengo un DataFrame llamado 'df' en memoria con esta estructura:
        {info_cols}
        
        Ejemplo de las primeras 3 filas:
        {muestra}
        
        Pregunta del usuario: "{pregunta}"
        
        TU TAREA:
        1. Escribe el código Python para calcular la respuesta usando la variable 'df'.
        2. Guarda el resultado final en una variable llamada 'resultado'.
        3. NO inventes datos. NO uses pd.read_csv.
        4. Devuelve SOLO el código, sin explicaciones ni markdown.
        """
        
        # USAREMOS EL MODELO ESTÁNDAR 'gemini-pro' PARA EVITAR ERRORES 404
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content(prompt)
        codigo = response.text.replace("```python", "").replace("```", "").strip()
        
        # 3. Ejecutamos el código en un entorno local
        local_vars = {'df': df, 'pd': pd}
        exec(codigo, globals(), local_vars)
        
        return local_vars.get('resultado', "No se generó respuesta."), codigo

    except Exception as e:
        return f"Error en el análisis: {str(e)}", ""

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_simple():
    try:
        conn = st.connection("sql", type="sql")
        # Traemos los datos
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza básica
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error conectando a SQL: {e}")
        return pd.DataFrame()

# ==========================================
# PÁGINA 1: CEREBRO (CORREGIDO)
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🧠 Cerebro (Modo Agente)")
    st.info("💡 Estrategia: Análisis mediante código Python (Sin gastar límite de tokens).")

    # 1. API KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    
    if not api_key:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    # 2. Obtener Datos
    df = cargar_datos_simple()
    
    if df.empty:
        st.warning("Esperando datos... (Revisa tu conexión SQL)")
    else:
        st.success(f"✅ Datos listos: {len(df):,} registros.")
        
        with st.expander("Ver muestra de datos"):
            st.dataframe(df.head())

        # 3. Pregunta
        pregunta = st.text_input("Consulta:", "Dime cuál fue la sucursal con más ingresos y el total.")
        
        if st.button("Analizar con IA"):
            if not api_key:
                st.error("⛔ Falta la API Key.")
            else:
                with st.spinner("🤖 Pensando la solución..."):
                    respuesta, codigo_usado = analizar_con_agente(df, pregunta, api_key)
                    
                    st.divider()
                    st.subheader("📊 Resultado:")
                    st.write(respuesta)
                    
                    with st.expander("🔍 Ver código generado"):
                        st.code(codigo_usado, language='python')

# ==========================================
# PÁGINA 2: REPORTES
# ==========================================
elif pagina == "📊 Reportes":
    st.title("📊 Reportes")
    df = cargar_datos_simple()
    if not df.empty:
        if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
             df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')

        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        
        # Filtros
        sucursal = st.sidebar.selectbox("Sucursal", ["Todas"] + list(df['Sucursal'].unique()))
        if sucursal != "Todas": df = df[df['Sucursal'] == sucursal]
        
        col1, col2 = st.columns(2)
        col1.metric("Total Ingresos", f"${df['Valor'].sum():,.0f}")
        col2.metric("Transacciones", len(df))
        
        st.bar_chart(df.groupby('Mes')['Valor'].sum())

# ==========================================
# PÁGINA 3: MAPA
# ==========================================
elif pagina == "🗺️ Mapa":
    st.title("🗺️ Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES", ttl=0))
    except:
        st.error("Error de conexión SQL")
