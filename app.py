import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", layout="wide")

# --- FUNCIÓN CEREBRO (LÓGICA BLINDADA) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto (Solo estructura)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT DE MÁXIMA SEGURIDAD
        prompt_system = """
        Eres Gari, experto en Python y Data Science.
        
        TU MISIÓN:
        Escribir código Python para analizar el DataFrame 'df' que YA ESTÁ EN MEMORIA.
        
        PROHIBICIONES ABSOLUTAS (SI LAS ROMPES, FALLAS):
        1. PROHIBIDO crear datos ficticios (ej: data = {...}).
        2. PROHIBIDO usar pd.DataFrame() para crear datos manuales.
        3. PROHIBIDO usar la columna 'FechaCargue'.
        
        INSTRUCCIONES DE ANÁLISIS:
        1. Usa la columna 'Fecha' (datetime).
        2. Filtra el DataFrame por el año solicitado en la pregunta.
        3. Si el DataFrame filtrado está vacío, define resultado=None y detente.
        
        INSTRUCCIONES DE SALIDA (Variables a crear):
        1. 'resultado' (str): Nombre del mes con mayor venta en ESPAÑOL.
        
        2. 'tabla_resultados' (DataFrame): 
           - Agrupa por mes y suma 'Valor'.
           - Columnas: ['Mes', 'Ventas'].
           - ORDENAMIENTO: Debes ordenar los meses cronológicamente (Enero, Febrero, Marzo...), NO por valor.
             * Tip: Crea una lista ordenada de meses en español y úsala para ordenar.
        
        3. 'fig' (matplotlib figure):
           - Gráfico de barras de 'tabla_resultados'.
           - Título: 'Ventas por Mes'.
           - ETIQUETAS: Agrega el valor numérico encima de cada barra (ax.bar_label).
           - Formato de miles en el eje Y.
        """
        
        prompt_user = f"""
        Estructura de la tabla REAL (SQL):
        {info_cols}
        
        Muestra (Solo referencia, EL DATASET COMPLETO TIENE DATOS DE 2025):
        {muestra}
        
        Pregunta: "{pregunta}"
        
        Genera SOLO el código Python. Confía en que 'df' tiene los datos del 2025 aunque la muestra sea vieja.
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
        # Traemos toda la tabla
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        # Forzar formato DD/MM/AAAA
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
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

    # Carga de datos REAL
    with st.spinner("Conectando a la Base de Datos..."):
        df = cargar_datos_sql()
    
    if not df.empty:
        # Verificación visual de fechas
        fecha_max = df['Fecha'].max()
        st.caption(f"📅 Datos reales hasta: {fecha_max.strftime('%d/%m/%Y')}")
            
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando datos reales..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df, pregunta, api_key)
                    
                    st.divider()
                    
                    # 1. Respuesta Texto
                    if res_txt:
                        st.success(f"📌 El mes ganador es: **{res_txt}**")
                    else:
                        st.warning("El código corrió pero no encontró ventas en 2025 (Filtro vacío).")

                    # 2. Tabla Ordenada
                    if res_tabla is not None:
                        st.write("### 📅 Resumen Mensual")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    # 3. Gráfico con Etiquetas
                    if res_fig:
                        st.write("### 📊 Gráfico")
                        st.pyplot(res_fig)
                    
                    # 4. Código (Para verificar que NO inventó datos)
                    with st.expander("Ver código real ejecutado"):
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
