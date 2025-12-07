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
        
        # 1. Contexto (Schema)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT DE GARI (Instrucciones precisas para Tabla y Gráfico)
        prompt_system = """
        Eres Gari, el segundo cerebro extendido.
        
        TU OBJETIVO: Generar código Python para analizar el DataFrame 'df'.
        
        REGLAS DE ORO:
        1. NO crees datos de ejemplo. Usa el DataFrame 'df' que ya existe.
        2. Trabaja con la columna 'Fecha' (datetime). Ignora 'FechaCargue'.
        
        SALIDAS REQUERIDAS (Debes crear estas 3 variables):
        
        A. Variable 'resultado' (str): 
           - El nombre del mes con mayores ventas en Español.
           
        B. Variable 'tabla_resultados' (DataFrame):
           - Debe tener dos columnas: ['Mes', 'Ventas'].
           - Agrupa las ventas por mes.
           - IMPORTANTE: Ordena la tabla por calendario (Enero, Febrero...), NO por valor de venta.
           - Tip: Usa un diccionario para mapear número de mes a nombre en Español.
           
        C. Variable 'fig' (matplotlib figure):
           - Gráfico de barras de las ventas por mes.
           - Título: 'Evolución de Ventas'.
           - ETIQUETAS: Agrega el valor exacto encima de cada barra usando ax.bar_label().
           - Formato de miles en el eje Y.
        """
        
        prompt_user = f"""
        Estructura de la tabla SQL:
        {info_cols}
        
        Muestra de datos:
        {muestra}
        
        Pregunta del usuario: "{pregunta}"
        
        TAREA: Genera SOLO el código Python necesario.
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
        
        # 4. Ejecución del código generado
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker}
        exec(codigo, globals(), local_vars)
        
        # Recuperamos las variables que creó la IA
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
        # Traemos toda la tabla
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza y Formatos
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

    with st.spinner("Conectando con la base de datos..."):
        df = cargar_datos_sql()
    
    if not df.empty:
        # Verificador discreto de fecha
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos disponibles hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Gari está analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # 1. Respuesta Texto
                    if res_txt:
                        st.success(f"📌 El mes ganador es: **{res_txt}**")
                    else:
                        st.warning("No encontré datos para responder (o el filtro de año quedó vacío).")

                    # 2. Tabla (Lo que pediste)
                    if res_tabla is not None:
                        st.write("### 📅 Resumen Mensual")
                        # Mostramos la tabla formateada con signos de pesos
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    # 3. Gráfico (Lo que pediste)
                    if res_fig:
                        st.write("### 📊 Gráfico Detallado")
                        st.pyplot(res_fig)
                    
                    # 4. Código (Transparencia)
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
