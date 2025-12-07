import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA (LIMPIA) ---
st.set_page_config(page_title="Gari", page_icon="🧠", layout="wide")

# --- FUNCIÓN CEREBRO (GPT-4o) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT DE GARI (INSTRUCCIONES EN ESPAÑOL Y ORDEN)
        prompt_system = """
        Eres Gari, el segundo cerebro extendido.
        
        REGLAS PARA EL CÓDIGO PYTHON:
        1. Usa la columna 'Fecha'. Ignora 'FechaCargue'.
        2. Filtra por el año pedido. Si df queda vacío, detente.
        
        3. INSTRUCCIONES DE SALIDA:
           A. Variable 'resultado': Nombre del mes con más ventas (String en Español).
           
           B. Variable 'tabla_resultados': 
              - Agrupa por mes y suma 'Valor'.
              - Crea un DataFrame con columnas ['Mes', 'Ventas'].
              - IMPORTANTE: La columna 'Mes' debe ser en ESPAÑOL (Enero, Febrero...) y estar ordenada por calendario (no alfabético).
              - Usa un diccionario: {1: 'Enero', 2: 'Febrero'...} para mapear el número de mes.
           
           C. Variable 'fig': Gráfico de barras (matplotlib).
              - Eje X: Meses en Español.
              - Eje Y: Ventas.
              - AGREGA ETIQUETAS DE DATOS: Usa ax.bar_label(bars, fmt='${:,.0f}') para poner el valor encima de las barras.
              - Rota las etiquetas del eje X.
        """
        
        prompt_user = f"""
        Datos (SQL):
        {info_cols}
        
        Muestra:
        {muestra}
        
        Pregunta: "{pregunta}"
        
        TAREA: Genera SOLO el código Python.
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
        
        # 4. Ejecución
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker}
        exec(codigo, globals(), local_vars)
        
        return (local_vars.get('resultado', None), 
                local_vars.get('fig', None), 
                local_vars.get('tabla_resultados', None), 
                codigo)

    except Exception as e:
        return f"Error: {str(e)}", None, None, ""

# --- CARGA DE DATOS SQL (SIN CACHÉ) ---
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

# --- INTERFAZ ---

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
        # Info discreta
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos disponibles hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # 1. Respuesta Texto
                    if res_txt:
                        st.success(f"📌 El mes ganador fue: **{res_txt}**")
                    else:
                        st.warning("No encontré datos para responder esa fecha.")

                    # 2. Tabla (Nueva sección)
                    if res_tabla is not None:
                        st.write("### 📅 Resumen Mensual")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    # 3. Gráfico
                    if res_fig:
                        st.write("### 📊 Gráfico Detallado")
                        st.pyplot(res_fig)
                    
                    # 4. Código
                    with st.expander("Ver código generado"):
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
