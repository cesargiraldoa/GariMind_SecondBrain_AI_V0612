import streamlit as st
import pandas as pd
import openai
import io
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind - GPT", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN INTELIGENTE (MOTOR OPENAI) ---
def analizar_con_gpt(df, pregunta, api_key):
    """
    Usa OpenAI (GPT-4o) para generar código Python de análisis.
    Envía solo la estructura (metadata), no los datos brutos.
    """
    try:
        # Configurar cliente
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Sacar 'radiografía' de los datos (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. Prompt de Ingeniería para Data Science
        prompt_system = "Eres un experto Data Scientist en Python. Tu trabajo es escribir código Pandas para responder preguntas."
        
        prompt_user = f"""
        Tengo un DataFrame 'df' en memoria.
        
        Columnas y Tipos:
        {info_cols}
        
        Muestra de datos:
        {muestra}
        
        Pregunta del usuario: "{pregunta}"
        
        TU TAREA:
        1. Escribe el código Python para responder usando la variable 'df'.
        2. El resultado final debe quedar en una variable llamada 'resultado'.
        3. IMPORTANTE: Devuelve SOLO el código limpio. Sin markdown (```) y sin explicaciones.
        """

        # 3. Llamada a la API (GPT-4o)
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0 # Temperatura 0 para máxima precisión matemática
        )
        
        codigo = response.choices[0].message.content
        
        # Limpieza de seguridad
        codigo = codigo.replace("```python", "").replace("```", "").strip()
        
        # 4. Ejecución Segura Local
        local_vars = {'df': df, 'pd': pd}
        exec(codigo, globals(), local_vars)
        
        return local_vars.get('resultado', "El código se ejecutó pero no generó la variable 'resultado'."), codigo

    except Exception as e:
        return f"Error OpenAI: {str(e)}", ""

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_simple():
    try:
        conn = st.connection("sql", type="sql")
        # Ajusta esta query a tu tabla real si es necesario
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza básica
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        # Asegurar formato fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# PÁGINA 1: CEREBRO
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🧠 Cerebro (Motor GPT-4o)")
    st.info("💡 Análisis potenciado por OpenAI.")

    # 1. GESTIÓN DE API KEY (SECRETS O MANUAL)
    # Primero buscamos en los secrets de Streamlit
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        usando_secrets = True
    else:
        # Si no está en secrets, pedimos manual
        api_key = st.text_input("Ingresa tu OpenAI API Key (sk-...):", type="password")
        usando_secrets = False

    # 2. CARGA DE DATOS
    df = cargar_datos_simple()
    
    if df.empty:
        st.warning("⚠️ No hay datos. Revisa la conexión SQL.")
    else:
        st.success(f"✅ Datos cargados: {len(df):,} registros.")
        
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if not api_key:
                st.error("⛔ Necesitas una API Key para continuar.")
            else:
                with st.spinner("🧠 GPT-4o está programando la respuesta..."):
                    res, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    st.subheader("📊 Resultado:")
                    st.write(res)
                    
                    with st.expander("🔍 Ver código generado (Python)"):
                        st.code(cod, language='python')

# ==========================================
# REPORTES Y MAPA (TU CÓDIGO ESTÁNDAR)
# ==========================================
elif pagina == "📊 Reportes":
    st.title("Reportes")
    df = cargar_datos_simple()
    if not df.empty:
        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        st.bar_chart(df.groupby('Mes')['Valor'].sum())

elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES", ttl=0))
    except:
        st.error("Error SQL")
