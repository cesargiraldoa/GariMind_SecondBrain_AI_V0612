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

# --- FUNCIÓN INTELIGENTE (MEJORADA) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Schema
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT CON REGLAS DE NEGOCIO (Aquí está la corrección)
        prompt_system = """
        Eres un experto Data Scientist en Logística.
        REGLAS CLAVE SOBRE LOS DATOS:
        1. La columna 'Fecha' es la FECHA DE VENTA real. Úsala siempre para análisis de tiempo.
        2. La columna 'FechaCargue' es técnica (auditoría). IGNÓRALA para análisis de negocio.
        3. Si el resultado es un mes, devuelve el NOMBRE del mes (ej: 'Enero'), no el número.
        """
        
        prompt_user = f"""
        Tengo un DataFrame 'df'. 
        Columnas:
        {info_cols}
        
        Muestra:
        {muestra}
        
        Pregunta del usuario: "{pregunta}"
        
        TU TAREA:
        1. Escribe código Python para responder.
        2. Si el usuario pide un dato, guárdalo en la variable 'resultado'.
        3. Si pide un GRÁFICO, crea una variable 'fig' con matplotlib (sin plt.show()).
        4. Devuelve SOLO código limpio.
        """

        # 3. GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        
        codigo = response.choices[0].message.content
        codigo = codigo.replace("```python", "").replace("```", "").strip()
        
        # 4. Ejecución
        local_vars = {'df': df, 'pd': pd, 'plt': plt}
        exec(codigo, globals(), local_vars)
        
        resultado = local_vars.get('resultado', None)
        figura = local_vars.get('fig', None)
        
        return resultado, figura, codigo

    except Exception as e:
        return f"Error: {str(e)}", None, ""

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_simple():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# PÁGINA CEREBRO
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🧠 Cerebro (Lógica Corregida)")
    
    # API KEY
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("OpenAI API Key:", type="password")

    df = cargar_datos_simple()
    
    if not df.empty:
        st.success(f"Datos cargados: {len(df):,} filas.")
        
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando columna 'Fecha'..."):
                    res_txt, res_fig, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    if res_txt is not None:
                        st.subheader("📊 Respuesta:")
                        st.write(res_txt)
                    
                    if res_fig is not None:
                        st.subheader("📈 Gráfico:")
                        st.pyplot(res_fig)
                    
                    with st.expander("Ver código (Validar lógica)"):
                        st.code(cod, language='python')
            else:
                st.error("Falta API Key")
