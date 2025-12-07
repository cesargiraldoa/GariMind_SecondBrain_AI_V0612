import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN DE PÁGINA ---
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
        
        # 2. PROMPT DE GARI (Con lógica de fecha estricta)
        prompt_system = """
        Eres Gari, el segundo cerebro extendido.
        
        REGLAS DE ORO PARA EL CÓDIGO:
        1. La única fecha válida es la columna 'Fecha'.
        2. El formato de 'Fecha' es DÍA-MES-AÑO.
        3. IGNORA la columna 'FechaCargue'.
        4. Si preguntan por el mejor mes, agrupa las ventas por mes usando 'Fecha' y devuelve el nombre del mes en ESPAÑOL.
        5. Si no hay datos para el año pedido, imprime "No hay datos registrados para ese año".
        6. Si piden gráfico, usa matplotlib y guarda la figura en la variable 'fig'.
        """
        
        prompt_user = f"""
        Estructura de la tabla (SQL Server):
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

# --- CARGA DE DATOS SQL DIRECTA ---
@st.cache_data(ttl=0) # ttl=0 para que NO guarde caché y traiga datos frescos siempre
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        # Traemos toda la tabla
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza y Formato
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        
        # CLAVE: Forzar formato Día-Mes-Año
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error conectando a la BD: {e}")
        return pd.DataFrame()

# --- INTERFAZ DE USUARIO ---

# 1. Menú limpio (Sin íconos raros)
menu = st.sidebar.radio("Navegación", ["Chat con Gari", "Reportes", "Mapa"])

# 2. Imagen del Hámster (Gari)
st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=150, caption="Gari 🐹")

if menu == "Chat con Gari":
    
    # SALUDO PERSONALIZADO
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=80)
    with col2:
        st.title("Hola soy Gari tu segundo cerebro extendido")
        st.write("¿Cómo te puedo ayudar hoy?")

    # GESTIÓN API KEY
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Ingresa tu API Key de OpenAI:", type="password")

    # CARGA SQL (Invisible al usuario)
    df = cargar_datos_sql()
    
    if not df.empty:
        # Input de pregunta
        pregunta = st.text_input("Escribe tu consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Gari está pensando... 🐹"):
                    res_txt, res_fig, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    if res_txt:
                        st.success("Respuesta:")
                        st.write(res_txt)
                    
                    if res_fig:
                        st.write("Gráfico:")
                        st.pyplot(res_fig)
                    
                    with st.expander("Ver código Python"):
                        st.code(cod, language='python')
            else:
                st.warning("Falta la API Key")
    else:
        st.error("No se pudieron cargar datos desde SQL Server.")

elif menu == "Reportes":
    st.title("Reportes")
    df = cargar_datos_sql()
    if not df.empty:
        st.dataframe(df.head())

elif menu == "Mapa":
    st.title("Mapa de Tablas")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except:
        st.write("Error SQL")
