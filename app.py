import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA (Identidad Gari) ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- FUNCIÓN CEREBRO (GPT-4o) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT DE GARI (Instrucciones estrictas)
        prompt_system = """
        Eres Gari, el segundo cerebro extendido.
        
        REGLAS PARA EL CÓDIGO PYTHON:
        1. Trabaja ÚNICAMENTE con la columna 'Fecha' (formato datetime).
        2. IGNORA la columna 'FechaCargue'.
        3. Filtra primero por el año solicitado. IMPORTANTE: Si el filtro queda vacío, asigna None a 'resultado' y no intentes graficar.
        4. Si hay datos: Agrupa por mes, suma el 'Valor' y guarda el nombre del mes ganador en la variable 'resultado'.
        5. Genera un gráfico de barras con matplotlib y guarda la figura en la variable 'fig'.
        """
        
        prompt_user = f"""
        Estructura de la tabla:
        {info_cols}
        
        Muestra de datos:
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

# --- CARGA DE DATOS SQL (SIN CACHÉ) ---
# ttl=0 OBLIGA a consultar la base de datos real cada vez
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        # Traemos toda la tabla
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        
        # CLAVE: Forzar formato Día-Mes-Año
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error conectando a SQL: {e}")
        return pd.DataFrame()

# --- INTERFAZ GARI (LIMPIA) ---

# Título y Saludo
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=100)
with col2:
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### ¿Cómo te puedo ayudar hoy?")

# Menú lateral simple
pagina = st.sidebar.radio("Menú", ["Chat", "Reportes", "Mapa"])
st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=150, caption="Gari 🐹")


if pagina == "Chat":
    
    # API KEY
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    # Carga de datos
    with st.spinner("Conectando con la Base de Datos..."):
        df = cargar_datos_sql()
    
    if not df.empty:
        # --- VERIFICADOR DE VERDAD ---
        fecha_max = df['Fecha'].max()
        
        # Caja de información visual
        st.info(f"📅 Datos cargados hasta: **{fecha_max.strftime('%d de %B de %Y')}**")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Gari está pensando..."):
                    res_txt, res_fig, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # Mostrar respuesta
                    if res_txt:
                        st.success(f"🐹 Respuesta: {res_txt}")
                    else:
                        st.warning("Gari revisó los datos pero no encontró registros para esa fecha específica.")

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
