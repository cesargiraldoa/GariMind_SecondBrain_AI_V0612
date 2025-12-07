import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- FUNCIÓN CEREBRO (GPT-4o ESTÉTICO) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT DE DISEÑO
        prompt_system = """
        Eres Gari, experto en Data Visualization con Python.
        
        REGLAS DE DATOS:
        1. Usa 'df' y la columna 'Fecha'. Ignora 'FechaCargue'.
        2. NO inventes datos.
        
        REGLAS DE VISUALIZACIÓN (CRÍTICO):
        1. IDIOMA: Los meses DEBEN ser en Español (Enero, Febrero...).
           - Usa un diccionario manual: {1: 'Enero', 2: 'Febrero'...}. NO uses calendar.month_name (sale en inglés).
        
        2. TABLA:
           - Crea 'tabla_resultados' agrupando por mes.
           - Ordena cronológicamente (Enero primero).
        
        3. GRÁFICO ('fig'):
           - Título: 'Ventas Mensuales 2025'.
           - Color de barras: 'skyblue' o un color corporativo suave.
           - ETIQUETAS: Usa ax.bar_label(bars, fmt='${:,.0f}', rotation=90, padding=4).
             * IMPORTANTE: La rotación 90 evita que se superpongan.
           - MARGENES: Aumenta el límite Y (ax.set_ylim) un 20% extra para que las etiquetas verticales quepan y no se corten.
        """
        
        prompt_user = f"""
        Estructura: {info_cols}
        Muestra: {muestra}
        Pregunta: "{pregunta}"
        
        Genera SOLO el código Python para crear: 'resultado', 'tabla_resultados' y 'fig'.
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
        # Info fecha
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Diseñando gráfico..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    if res_txt:
                        st.success(f"📌 Mes ganador: **{res_txt}**")
                    else:
                        st.warning("No hay datos para esta fecha.")

                    if res_tabla is not None:
                        st.write("### 📅 Resumen (Ordenado)")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    if res_fig:
                        st.write("### 📊 Gráfico Detallado")
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
