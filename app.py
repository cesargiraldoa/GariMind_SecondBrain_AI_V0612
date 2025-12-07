import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- HERRAMIENTAS ---
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        # Columnas extra para los reportes (cuando los quieras usar)
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return pd.DataFrame()

# --- FUNCIÓN CEREBRO (PARA EL CHAT) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Contexto
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        # 2. PROMPT BLINDADO (El que ya nos funcionó)
        prompt_system = """
        Eres Gari, experto Data Scientist.
        
        TU ENTORNO:
        - Tienes un DataFrame 'df'.
        - Tienes un diccionario 'meses_es' cargado.
        
        REGLAS:
        1. Usa 'df' y columna 'Fecha'. Ignora 'FechaCargue'.
        2. NO inventes datos.
        3. Para traducir meses usa: df['Fecha'].dt.month.map(meses_es).
        
        OUTPUTS REQUERIDOS:
        A. 'resultado' (str): Mes ganador.
        B. 'tabla_resultados' (DataFrame): ['Mes', 'Ventas'], ORDENADO cronológicamente (Enero, Febrero...).
        C. 'fig' (matplotlib): Gráfico de barras con ETIQUETAS DE DATOS encima.
        """
        
        prompt_user = f"""
        Estructura: {info_cols}
        Muestra: {muestra}
        Pregunta: "{pregunta}"
        Genera SOLO código Python.
        """

        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}], temperature=0)
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker, 'meses_es': meses_es}
        exec(codigo, globals(), local_vars)
        
        return (local_vars.get('resultado', None), local_vars.get('fig', None), local_vars.get('tabla_resultados', None), codigo)

    except Exception as e:
        return f"Error: {e}", None, None, ""

# --- FUNCIÓN REPORTES (OPCIONAL) ---
def generar_reporte_periodistico(df_filtrado, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        total_ventas = df_filtrado['Valor'].sum()
        ventas_sucursal = df_filtrado.groupby('Sucursal')['Valor'].sum().sort_values(ascending=False).to_dict()
        
        prompt = f"""
        Eres un analista de negocios. Escribe un resumen ejecutivo de 5 líneas sobre estos datos:
        Ventas Totales: ${total_ventas:,.0f}
        Por Sucursal: {ventas_sucursal}
        """
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        return response.choices[0].message.content
    except: return "No pude generar el reporte."

# --- INTERFAZ PRINCIPAL ---

st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=120, caption="Gari 🐹")

# AQUÍ ESTÁ EL CAMBIO: Puse el Chat de primero para que sea lo principal
pagina = st.sidebar.radio("Menú Principal", ["🧠 Chat con Gari", "📊 Reportes Ejecutivos", "🗺️ Mapa"])

# API KEY GLOBAL
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 API Key OpenAI:", type="password")

# CARGA DE DATOS COMÚN
with st.spinner("Cargando datos..."):
    df_raw = cargar_datos_sql()

# ==============================================================================
# PÁGINA 1: CHAT CON GARI (TU PÁGINA PRINCIPAL)
# ==============================================================================
if pagina == "🧠 Chat con Gari":
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### Haz preguntas a tus datos (Tabla Ordenada + Gráfico)")
    
    if not df_raw.empty:
        # Info de fecha
        fecha_max = df_raw['Fecha'].max()
        st.caption(f"📅 Datos disponibles hasta: {fecha_max.strftime('%d/%m/%Y')}")
        
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Gari está analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df_raw, pregunta, api_key)
                    
                    st.divider()
                    
                    # 1. Respuesta Texto
                    if res_txt:
                        st.success(f"📌 Resultado: **{res_txt}**")
                    else:
                        st.warning("No encontré datos para responder.")

                    # 2. Tabla Ordenada (Lo que arreglamos)
                    if res_tabla is not None:
                        st.write("### 📅 Detalle Mensual")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)

                    # 3. Gráfico con Etiquetas (Lo que arreglamos)
                    if res_fig:
                        st.write("### 📊 Gráfico")
                        st.pyplot(res_fig)
                    
                    with st.expander("Ver código"):
                        st.code(cod, language='python')
            else:
                st.error("Falta API Key")
    else:
        st.error("No hay datos cargados.")

# ==============================================================================
# PÁGINA 2: REPORTES EJECUTIVOS (TABLERO)
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Tablero de Mando")
    
    if not df_raw.empty:
        st.sidebar.header("🎯 Filtros")
        years = sorted(df_raw['Año'].unique().tolist(), reverse=True)
        year_sel = st.sidebar.multiselect("Año", years, default=years[:1])
        
        df_filtered = df_raw.copy()
        if year_sel:
            df_filtered = df_filtered[df_filtered['Año'].isin(year_sel)]
            
        col1, col2 = st.columns(2)
        col1.metric("Ventas Totales", f"${df_filtered['Valor'].sum():,.0f}")
        col2.metric("Transacciones", f"{len(df_filtered):,}")
        
        st.bar_chart(df_filtered.groupby('Sucursal')['Valor'].sum())
        
        if st.button("✨ Análisis IA"):
            if api_key:
                st.write(generar_reporte_periodistico(df_filtered, api_key))

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa de Tablas SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
