import streamlit as st
import pandas as pd
import time
import os
import re
from google import genai
from google.genai import types

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")

# --- MENÚ LATERAL ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# ==========================================
# PÁGINA 1: CEREBRO (MODO DEBUG - SIN SILENCIADOR DE ERRORES)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    
    # 1. Iniciar Cliente (Si falla aquí, saldrá error rojo inmediato)
    client = genai.Client()
        
    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #4B5563;">Asistente de Logística & Análisis de Datos</div>', unsafe_allow_html=True)
    st.divider()

    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta_usuario = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        boton_analizar = st.button("Analizar", type="primary", use_container_width=True)

    if boton_analizar and pregunta_usuario:
        
        # --- PASO 1: Generar SQL ---
        with st.spinner('Generando SQL...'):
            
            # Prompt específico para forzar la conversión de datos
            sql_prompt = f"""
            Genera una consulta T-SQL (SQL Server) para responder: "{pregunta_usuario}"
            
            TABLA: stg.Ingresos_Detallados
            COLUMNAS: Fecha (string), Valor (NVARCHAR), Sucursal (string).
            
            REGLA DE ORO: La columna 'Valor' es TEXTO. 
            SIEMPRE usa: CAST(Valor AS FLOAT) o TRY_CAST(Valor AS FLOAT) para sumar o promediar.
            Ejemplo: SUM(TRY_CAST(Valor AS FLOAT))
            
            Salida: Solo el bloque de código SQL markdown.
            """

            # Llamada API
            response_sql = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=sql_prompt)])]
            )
            
            texto_respuesta = response_sql.text
            
            # DEBUG: Mostrar lo que respondió la IA antes de intentar ejecutarlo
            with st.expander("Ver SQL generado por IA (Debug)", expanded=False):
                st.write(texto_respuesta)

            # Extraer SQL con Regex
            match = re.search(r"```sql(.*?)```", texto_respuesta, re.DOTALL)
            
            if not match:
                st.error("⛔ La IA no generó código SQL válido inside ```sql ```.")
                st.stop()
            
            sql_query = match.group(1).strip()
            
            # DEBUG: Mostrar la consulta limpia
            st.code(sql_query, language="sql")

        # --- PASO 2: Ejecutar SQL (Sin Try/Except para ver el error real) ---
        conn = st.connection("sql", type="sql")
        df_result = conn.query(sql_query, ttl=0)
        
        st.success("✅ Datos obtenidos:")
        st.dataframe(df_result)
        
        # --- PASO 3: Análisis de Texto ---
        with st.spinner('Analizando resultados...'):
            analysis_prompt = f"""
            Pregunta: {pregunta_usuario}
            Datos:
            {df_result.to_markdown(index=False)}
            
            Dame un análisis ejecutivo breve y una recomendación.
            """
            
            response_analysis = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=analysis_prompt)])]
            )
            
            st.subheader("Respuesta:")
            st.markdown(response_analysis.text)


# ==========================================
# PÁGINA 2: REPORTES EJECUTIVOS (ESTABLE)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Reporte de Variación de Ingresos")
    conn = st.connection("sql", type="sql")
    
    # Consulta robusta manual
    query = """
        SELECT 
            Fecha as fecha, 
            CASE WHEN ISNUMERIC(Valor)=1 THEN CAST(Valor AS FLOAT) ELSE 0 END as valor,
            Sucursal as sucursal
        FROM stg.Ingresos_Detallados
        WHERE ISDATE(Fecha) = 1
        ORDER BY Fecha
    """
    df = conn.query(query, ttl=600)
    
    # Limpieza en Pandas
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['fecha'], inplace=True)
    df['mes_anio'] = df['fecha'].dt.strftime('%Y-%m')
    
    # Filtros
    st.sidebar.header("Filtros")
    filtro = st.sidebar.selectbox("Sucursal:", ["Todas"] + list(df['sucursal'].unique()))
    if filtro != "Todas":
        df = df[df['sucursal'] == filtro]

    # Cálculos
    df_grp = df.groupby('mes_anio')['valor'].sum().reset_index()
    df_grp['var'] = df_grp['valor'].pct_change().fillna(0) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", f"${df['valor'].sum():,.0f}")
    col2.metric("Promedio", f"${df_grp['valor'].mean():,.0f}")
    col3.metric("Var. Mes", f"{df_grp['var'].iloc[-1]:.1f}%")
    
    c1, c2 = st.columns(2)
    c1.bar_chart(df_grp.set_index('mes_anio')['valor'])
    c2.bar_chart(df_grp.set_index('mes_anio')['var'])
    with st.expander("Ver Datos"): st.dataframe(df_grp)

# ==========================================
# PÁGINA 3: MAPA DE DATOS (ESTABLE)
# ==========================================
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de Datos")
    conn = st.connection("sql", type="sql")
    df_tablas = conn.query("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", ttl=600)
    
    c1, c2 = st.columns([1,2])
    c1.dataframe(df_tablas)
    
    t = c2.selectbox("Tabla:", df_tablas['TABLE_SCHEMA']+"."+df_tablas['TABLE_NAME'])
    if c2.button("Ver Muestra"):
        df = conn.query(f"SELECT TOP 50 * FROM {t}", ttl=0)
        st.success("✅ Conectado")
        st.balloons()
        st.dataframe(df)
