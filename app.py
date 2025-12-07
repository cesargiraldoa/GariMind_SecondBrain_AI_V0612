import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import calendar  # <--- 1. IMPORTAMOS LA HERRAMIENTA QUE FALTABA

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- FUNCIÓN CEREBRO (GPT-4o) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT
        prompt_system = """
        Eres Gari, experto en Data Science.
        
        TU MISIÓN:
        Analizar el DataFrame 'df' (ya cargado) y generar 3 variables.
        
        REGLAS:
        1. Usa la columna 'Fecha'. Ignora 'FechaCargue'.
        2. NO inventes datos. Usa 'df'.
        
        VARIABLES A CREAR:
        A. 'resultado' (str): Nombre del mes con más ventas (en Español).
        
        B. 'tabla_resultados' (DataFrame): 
           - Agrupado por Mes, suma de 'Valor'.
           - IMPORTANTE: Ordena los meses cronológicamente (Enero, Febrero...), usa la librería 'calendar' si la necesitas.
           
        C. 'fig' (matplotlib figure):
           - Gráfico de barras.
           - Etiquetas de datos encima de las barras.
           - Formato de miles en el eje Y.
        """
        
        prompt_user = f"""
        Estructura: {info_cols}
        Muestra: {muestra}
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
        
        # 4. Ejecución (AQUÍ ESTÁ EL ARREGLO)
        # Le damos permiso para usar 'calendar'
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker, 'calendar': calendar}
        exec(codigo, globals(), local_vars)
        
        return (local_vars.get('resultado', None), 
                local_vars.get('fig', None), 
                local_vars.get('tabla_resultados', None), 
                codigo)

    except Exception as e:
        return f"Error en el código generado: {str(e)}", None, None, ""

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return pd.DataFrame()

# --- INTERFAZ GARI ---

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
        # Confirmación visual
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos reales hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Gari está analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # 1. Respuesta
                    if res_txt and "Error" not in str(res_txt):
                        st.success(f"📌 El mes ganador es: **{res_txt}**")
                    elif res_txt:
                         st.error(res_txt) # Si hay error, mostrarlo

                    # 2. Tabla
                    if res_tabla is not None:
                        st.write("### 📅 Resumen Mensual")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    # 3. Gráfico
                    if res_fig:
                        st.write("### 📊 Gráfico")
                        st.pyplot(res_fig)
                    
                    # 4. Código
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
