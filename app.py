import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🧠", layout="wide")

# --- FUNCIÓN CEREBRO (GPT-4o BLINDADO) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT ESTRICTO (CANDADO ANTI-INVENCIÓN)
        prompt_system = """
        Eres Gari, un experto en Python y Streamlit.
        
        TU ENTORNO:
        - Ya existe un DataFrame cargado en memoria llamado 'df'.
        - NO debes cargar datos nuevos. NO uses pd.read_csv. NO crees diccionarios de datos ficticios.
        - Usa 'df' exclusivamente.
        
        TU OBJETIVO:
        Generar código Python que cree 3 variables usando 'df':
        
        1. 'resultado' (str): El nombre del mes con más ventas (en Español).
        
        2. 'tabla_resultados' (DataFrame): 
           - Agrupado por Mes.
           - Columnas: ['Mes', 'Ventas'].
           - IMPORTANTE: Ordena los meses cronológicamente: ['Enero', 'Febrero', 'Marzo'...].
           
        3. 'fig' (matplotlib figure):
           - Gráfico de barras.
           - Título: 'Ventas Totales por Mes'.
           - ETIQUETAS: Muestra el valor encima de cada barra (ax.bar_label).
           - Formato de ejes legible (miles).
        """
        
        prompt_user = f"""
        Estructura de 'df' (SQL):
        {info_cols}
        
        Muestra real:
        {muestra}
        
        Pregunta: "{pregunta}"
        
        TAREA: Genera SOLO el código Python para manipular 'df'. No expliques nada.
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
        
        # 4. Ejecución (Pasamos 'df' explícitamente)
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker}
        exec(codigo, globals(), local_vars)
        
        return (local_vars.get('resultado', None), 
                local_vars.get('fig', None), 
                local_vars.get('tabla_resultados', None), 
                codigo)

    except Exception as e:
        return f"Error de ejecución: {str(e)}", None, None, ""

# --- CARGA DE DATOS SQL (SIN CACHÉ) ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza y Formato
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error crítico SQL: {e}")
        return pd.DataFrame()

# --- INTERFAZ GARI (LIMPIA) ---

st.title("Hola soy Gari tu segundo cerebro extendido")
st.write("### ¿Cómo te puedo ayudar hoy?")

pagina = st.sidebar.radio("Menú", ["Chat", "Reportes", "Mapa"])

if pagina == "Chat":
    
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Ingresa tu API Key:", type="password")

    # Carga de datos real
    with st.spinner("Conectando a la Base de Datos..."):
        df = cargar_datos_sql()
    
    if not df.empty:
        # Verificación de Fecha Real
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos reales actualizados hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando tus datos reales..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # 1. Respuesta Texto
                    if res_txt:
                        st.success(f"📌 El mes ganador es: **{res_txt}**")
                    else:
                        st.warning("No hay datos para responder a esa fecha (El filtro devolvió vacío).")

                    # 2. Tabla (Lo que pediste)
                    if res_tabla is not None:
                        st.write("### 📅 Tabla Mensual Ordenada")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    # 3. Gráfico (Lo que pediste)
                    if res_fig:
                        st.write("### 📊 Gráfico con Etiquetas")
                        st.pyplot(res_fig)
                    
                    # 4. Código (Para auditar que no invente datos)
                    with st.expander("Auditoría de Código (Verificar que usa 'df')"):
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
