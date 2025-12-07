import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- HERRAMIENTAS PRE-CARGADAS (Para que la IA no falle) ---
# Definimos el diccionario AQUÍ para que siempre exista y no dependa de la IA.
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

# --- FUNCIÓN CEREBRO (GPT-4o) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT ROBUSTO
        prompt_system = """
        Eres Gari, experto Data Scientist.
        
        TU ENTORNO DE EJECUCIÓN:
        - Tienes un DataFrame 'df' cargado.
        - Tienes un diccionario 'meses_es' cargado ({1: 'Enero'...}).
        - Tienes librerías: pd, plt, ticker.
        
        REGLAS DE ORO:
        1. Usa 'df'. NO cargues datos nuevos.
        2. Usa la columna 'Fecha' (datetime). Ignora 'FechaCargue'.
        3. Para traducir meses, usa df['Fecha'].dt.month.map(meses_es).
        
        VARIABLES DE SALIDA REQUERIDAS:
        A. 'resultado' (str): Texto con el mes ganador.
        B. 'tabla_resultados' (DataFrame): 
           - Columnas: ['Mes', 'Ventas'].
           - ORDEN: Cronológico (Enero, Febrero...), NO alfabético ni por ventas.
        C. 'fig' (matplotlib figure):
           - Gráfico de barras con etiquetas de datos encima (ax.bar_label).
           - Título y ejes claros.
        """
        
        prompt_user = f"""
        Estructura tabla: {info_cols}
        Muestra datos: {muestra}
        Pregunta: "{pregunta}"
        
        Genera SOLO código Python.
        """

        # 3. Llamada GPT
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        # 4. EJECUCIÓN BLINDADA
        # Aquí pasamos 'meses_es' para que el código de la IA lo encuentre
        local_vars = {
            'df': df, 
            'pd': pd, 
            'plt': plt, 
            'ticker': ticker, 
            'meses_es': meses_es 
        }
        
        exec(codigo, globals(), local_vars)
        
        return (local_vars.get('resultado', None), 
                local_vars.get('fig', None), 
                local_vars.get('tabla_resultados', None), 
                codigo)

    except Exception as e:
        return f"Error de ejecución: {str(e)}", None, None, ""

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza CRÍTICA para evitar errores después
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return pd.DataFrame()

# --- INTERFAZ GARI ---

col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=80)
with col2:
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### ¿Cómo te puedo ayudar hoy?")

pagina = st.sidebar.radio("Menú", ["Chat", "Reportes", "Mapa"])

if pagina == "Chat":
    
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    with st.spinner("Conectando con BD..."):
        df = cargar_datos_sql()
    
    if not df.empty:
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos disponibles hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Procesando consulta..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    if res_txt and "Error" not in str(res_txt):
                        st.success(f"📌 {res_txt}")
                    elif res_txt:
                        st.error(res_txt)
                    else:
                        st.warning("No encontré datos para responder.")

                    if res_tabla is not None:
                        st.write("### 📅 Detalle Mensual")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    if res_fig:
                        st.write("### 📊 Gráfico")
                        st.pyplot(res_fig)
                    
                    with st.expander("Ver código"):
                        st.code(cod, language='python')
            else:
                st.error("Falta API Key")

elif pagina == "Reportes":
    st.title("Reportes")
    df = cargar_datos_sql()
    if not df.empty:
        st.dataframe(df.head())

elif pagina == "Mapa":
    st.title("Mapa")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except:
        st.write("Error SQL")
