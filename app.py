import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", layout="wide")

# --- FUNCIÓN CEREBRO (GPT-4o) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT ESTRICTO
        prompt_system = """
        Eres Gari, el segundo cerebro extendido.
        
        REGLAS PARA EL CÓDIGO PYTHON:
        1. La única fecha válida para ventas es la columna 'Fecha'.
        2. IGNORA la columna 'FechaCargue'.
        3. Si piden el mejor mes, agrupa por mes (usando 'Fecha') y suma el 'Valor'.
        4. Devuelve el nombre del mes en Español.
        5. IMPORTANTE: Primero verifica si hay datos para el año solicitado. Si el dataframe filtrado está vacío, imprime: "No hay datos registrados en la columna Fecha para este año".
        6. Si piden gráfico, usa matplotlib y guarda la figura en 'fig'.
        """
        
        prompt_user = f"""
        Estructura de la tabla:
        {info_cols}
        
        Muestra de datos (primeras filas):
        {muestra}
        
        Pregunta: "{pregunta}"
        
        TAREA: Genera SOLO el código Python.
        """

        # 3. Llamada a GPT
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        # 4. Ejecución
        local_vars = {'df': df, 'pd': pd, 'plt': plt}
        exec(codigo, globals(), local_vars)
        
        return local_vars.get('resultado', None), local_vars.get('fig', None), codigo

    except Exception as e:
        return f"Error técnico: {str(e)}", None, ""

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        # Traemos toda la tabla
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        # Forzar formato fecha Día-Mes-Año
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error conectando a SQL: {e}")
        return pd.DataFrame()

# --- INTERFAZ LIMPIA ---

# Título simple (Texto)
st.title("Hola soy Gari tu segundo cerebro extendido")
st.write("¿Cómo te puedo ayudar hoy?")

# Menú lateral simple
pagina = st.sidebar.radio("Menú", ["Chat", "Reportes", "Mapa"])

if pagina == "Chat":
    
    # API KEY
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    # Carga de datos
    df = cargar_datos_sql()
    
    if not df.empty:
        # DIAGNÓSTICO DE FECHAS (Para que sepas la verdad de tus datos)
        with st.expander("🔍 Verificar Fechas disponibles"):
            fecha_max = df['Fecha'].max()
            st.write(f"La fecha de venta más reciente que veo es: **{fecha_max}**")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando..."):
                    res_txt, res_fig, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # Mostrar respuesta
                    if res_txt:
                        st.write("### Respuesta:")
                        st.write(res_txt)
                    elif not res_fig:
                        st.warning("El código se ejecutó pero no devolvió un resultado. Revisa si hay datos para esa fecha.")

                    # Mostrar gráfico
                    if res_fig:
                        st.write("### Gráfico:")
                        st.pyplot(res_fig)
                    
                    # Mostrar código
                    with st.expander("Ver código Python"):
                        st.code(cod, language='python')
            else:
                st.error("Falta la API Key")

elif pagina == "Reportes":
    st.title("Reportes")
    df = cargar_datos_sql()
    if not df.empty:
        st.dataframe(df.head())

elif pagina == "Mapa":
    st.title("Mapa de Tablas")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except:
        st.write("Error SQL")
