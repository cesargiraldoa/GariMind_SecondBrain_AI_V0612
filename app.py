import streamlit as st
import pandas as pd
import os
from google import genai

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind Directo", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_simple():
    try:
        conn = st.connection("sql", type="sql")
        # Traemos los datos
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza básica para que se entienda bien
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        # Formato fecha corto para ahorrar espacio
        df['Fecha'] = df['Fecha'].dt.strftime('%d/%m/%Y')
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# PÁGINA 1: CEREBRO (MÉTODO DIRECTO)
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🧠 Cerebro (Análisis Directo)")
    st.info("💡 Estrategia: Enviar los datos directamente a la IA para evitar errores de SQL.")

    # 1. API KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            st.error("⛔ Falta la GEMINI_API_KEY.")
            st.stop()
            
    client = genai.Client(api_key=api_key)

    # 2. Obtener Datos
    df = cargar_datos_simple()
    
    if df.empty:
        st.error("No se pudieron cargar los datos de SQL Server.")
    else:
        st.success(f"✅ Datos cargados en memoria: {len(df)} registros.")
        with st.expander("Ver los datos que analizará la IA"):
            st.dataframe(df)

        # 3. Pregunta
        pregunta = st.text_input("Consulta:", "Dime cuál fue la sucursal con más ingresos y el total.")
        
        if st.button("Analizar con IA"):
            with st.spinner("La IA está leyendo tus datos..."):
                try:
                    # Convertimos los datos a texto (CSV) para que la IA los lea
                    # Limitamos a 200 filas por seguridad de tamaño, si tienes más, avísame.
                    datos_txt = df.to_csv(index=False)
                    
                    prompt = f"""
                    Actúa como un experto analista de datos.
                    Responde la siguiente pregunta basándote ÚNICAMENTE en los datos que te proporciono abajo.
                    
                    PREGUNTA: {pregunta}
                    
                    DATOS (Formato CSV):
                    {datos_txt}
                    
                    Instrucciones:
                    - Responde de forma directa y ejecutiva.
                    - Si calculas totales, menciona la cifra exacta.
                    - Da una recomendación breve al final.
                    """
                    
                    # Llamada Directa
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt]
                    )
                    
                    # Mostrar respuesta SIN FILTROS
                    st.subheader("🤖 Respuesta:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

# ==========================================
# PÁGINA 2: REPORTES
# ==========================================
elif pagina == "📊 Reportes":
    st.title("📊 Reportes")
    df = cargar_datos_simple()
    if not df.empty:
        df['Mes'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y').dt.strftime('%Y-%m')
        
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
