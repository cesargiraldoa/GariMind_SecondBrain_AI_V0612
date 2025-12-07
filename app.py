import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari - Tu Segundo Cerebro", page_icon="🐹", layout="wide")

# --- BARRA LATERAL ---
st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=150, caption="Soy Gari 🐹")
st.sidebar.title("Menú")
pagina = st.sidebar.radio("Ir a:", ["🧠 Gari Chat", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN CEREBRO (GPT-4o) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto de datos
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT PERSONALIZADO (Identidad Gari)
        prompt_system = """
        Eres Gari, un asistente de datos inteligente y amigable.
        REGLAS DE ORO PARA EL CÓDIGO PYTHON:
        1. La columna 'Fecha' tiene formato DIA-MES-AÑO. Úsala para filtrar.
        2. IGNORA la columna 'FechaCargue'.
        3. Si te piden el "mejor mes" o "mes más alto", calcula la suma por mes y devuelve el NOMBRE del mes (ej: Enero).
        4. Si al filtrar por año (ej: 2025) el DataFrame queda vacío, imprime: "No encontré datos para ese año en la base de datos".
        5. Si te piden gráfico, usa matplotlib y guarda la figura en 'fig'.
        """
        
        prompt_user = f"""
        Estructura de datos:
        {info_cols}
        
        Muestra:
        {muestra}
        
        Pregunta: "{pregunta}"
        
        TAREA: Genera solo el código Python para resolver esto.
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
        return f"Ocurrió un error: {str(e)}", None, ""

# --- CARGA DE DATOS ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados")
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        # Forzamos formato DD/MM/AAAA
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce') 
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# PÁGINA PRINCIPAL: GARI CHAT
# ==========================================
if pagina == "🧠 Gari Chat":
    
    # --- ENCABEZADO CON IMAGEN DEL HÁMSTER ---
    col_img, col_txt = st.columns([1, 5])
    
    with col_img:
        # Usamos una imagen de internet de un hamster. Si tienes la foto de tu hamster, dime y te enseño a subirla.
        st.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=120)
    
    with col_txt:
        st.title("Hola, soy Gari, tu segundo cerebro extendido 🐹")
        st.write("### ¿Cómo te puedo ayudar hoy?")

    # API KEY
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Dame mi llave (API Key):", type="password")

    # Cargar Datos
    with st.spinner("Olfateando datos... 🐹"):
        df = cargar_datos_sql()
    
    if not df.empty:
        # DIAGNÓSTICO RÁPIDO (Oculto en un expander para no molestar)
        with st.expander("🕵️‍♂️ Ver lo que Gari está viendo (Fechas)"):
            fecha_max = df['Fecha'].max()
            total_2025 = df[df['Fecha'].dt.year == 2025].shape[0]
            st.write(f"📅 Fecha más reciente en la base de datos: **{fecha_max.date()}**")
            st.write(f"🔢 Cantidad de ventas encontradas del 2025: **{total_2025}**")
            if total_2025 == 0:
                st.warning("⚠️ OJO: No veo ninguna fila del 2025. ¿Seguro que SQL las está trayendo?")

        pregunta = st.text_input("Pregúntame lo que quieras:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                res_txt, res_fig, cod = analizar_con_gpt(df, pregunta, api_key)
                
                st.divider()
                
                if res_txt:
                    st.success("🐹 Gari dice:")
                    st.write(res_txt)
                elif not res_fig:
                    st.warning("No pude calcular el resultado. Revisa si hay datos del 2025 en el 'Inspector' de arriba.")
                
                if res_fig:
                    st.write("### 🎨 Aquí tienes tu gráfico:")
                    st.pyplot(res_fig)
                
                with st.expander("Ver cómo lo pensé (Código)"):
                    st.code(cod, language='python')
            else:
                st.error("Necesito la llave (API Key) para funcionar.")

# ==========================================
# OTRAS PÁGINAS (Reportes y Mapa)
# ==========================================
elif pagina == "📊 Reportes":
    st.title("📊 Reportes Clásicos")
    df = cargar_datos_sql()
    if not df.empty:
        st.dataframe(df.head())

elif pagina == "🗺️ Mapa":
    st.title("🗺️ Mapa de Datos")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except:
        st.error("Error SQL")
