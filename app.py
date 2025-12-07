import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind - GPT", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN DE ANÁLISIS ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Preparar muestra
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT "ANTIBALAS" (Para que no use FechaCargue)
        prompt_system = """
        Eres un experto Data Scientist.
        REGLAS DE ORO:
        1. Para analizar ventas/tiempo, USA ÚNICAMENTE LA COLUMNA 'Fecha'.
        2. IGNORA COMPLETAMENTE la columna 'FechaCargue' (es administrativa).
        3. La columna 'Fecha' tiene formato DIA-MES-AÑO.
        4. Si te piden el 'mejor mes', devuelve el NOMBRE del mes.
        5. Si te piden GRÁFICO, genera una figura 'fig' usando matplotlib.
        """
        
        prompt_user = f"""
        DataFrame 'df':
        {info_cols}
        
        Muestra:
        {muestra}
        
        Pregunta: "{pregunta}"
        
        TAREA:
        Genera código Python para responder. Devuelve solo el código.
        """

        # 3. Consultar GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        # 4. Ejecutar
        local_vars = {'df': df, 'pd': pd, 'plt': plt}
        exec(codigo, globals(), local_vars)
        
        return local_vars.get('resultado', None), local_vars.get('fig', None), codigo

    except Exception as e:
        return f"Error: {str(e)}", None, ""

# --- CARGA DE DATOS SQL ---
# IMPORTANTE: ttl=0 evita que guarde datos viejos en caché
@st.cache_data(ttl=0) 
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        # Traemos toda la tabla
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados")
        
        # Limpieza
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        
        # AJUSTE DE FECHA (Fundamental)
        # dayfirst=True obliga a leer DD/MM/AAAA
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error de conexión SQL: {e}")
        return pd.DataFrame()

# ==========================================
# PÁGINA CEREBRO
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🧠 Cerebro Logístico")
    
    # API KEY
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("OpenAI API Key:", type="password")

    # Cargar Datos
    with st.spinner("Conectando a SQL Server..."):
        df = cargar_datos_sql()
    
    if not df.empty:
        # --- DIAGNÓSTICO VISUAL (¡ESTO ES CLAVE!) ---
        st.info(f"✅ Conexión Exitosa: {len(df):,} registros cargados.")
        
        with st.expander("🔍 INSPECTOR DE DATOS (Clic para verificar fechas)", expanded=True):
            fecha_min = df['Fecha'].min()
            fecha_max = df['Fecha'].max()
            registros_2025 = df[df['Fecha'].dt.year == 2025].shape[0]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Fecha más antigua", str(fecha_min.date()) if pd.notnull(fecha_min) else "N/A")
            col2.metric("Fecha más reciente", str(fecha_max.date()) if pd.notnull(fecha_max) else "N/A")
            col3.metric("Registros del 2025", f"{registros_2025:,}")
            
            if registros_2025 == 0:
                st.error("⚠️ ALERTA: SQL no está enviando filas con fecha de venta 2025. Revisa la base de datos.")
            else:
                st.success("✅ Datos del 2025 detectados correctamente.")

        # --- ÁREA DE CONSULTA ---
        pregunta = st.text_input("Pregunta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                res_txt, res_fig, cod = analizar_con_gpt(df, pregunta, api_key)
                
                st.divider()
                if res_txt:
                    st.write("### 📊 Respuesta:")
                    st.write(res_txt)
                
                if res_fig:
                    st.write("### 📈 Gráfico:")
                    st.pyplot(res_fig)
                
                with st.expander("Ver código"):
                    st.code(cod, language='python')
            else:
                st.warning("Falta la API Key")
