import streamlit as st
import pandas as pd
import time
import os
import re
from google import genai
from google.genai import types

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")

# --- MENÚ LATERAL (Navegación Manual) ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# ==========================================
# PÁGINA 1: CEREBRO (INICIO) - LÓGICA DE DOS PASOS (SQL + ANÁLISIS)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    
    # --- Configuración del SDK ---
    try:
        client = genai.Client()
    except Exception as e:
        st.error(f"⛔ ERROR: No se pudo iniciar el cliente de Gemini. Asegura GEMINI_API_KEY. Detalles: {e}")
        st.stop()
        
    # --- Interacción de Usuario y UI ---
    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #4B5563;">Asistente de Logística & Análisis de Datos</div>', unsafe_allow_html=True)
    st.divider()

    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta_usuario = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        boton_analizar = st.button("Analizar", type="primary", use_container_width=True)

    # --- Lógica de Procesamiento y Llamada a la IA ---
    if boton_analizar and pregunta_usuario:
        
        # 1. Definir el Esquema de la BD (Contexto para Gemini)
        schema_info = """
        Tabla: stg.Ingresos_Detallados
        Columnas clave: 
        - Fecha (string, DD/MM/YYYY): Fecha de la transacción.
        - Valor (nvarchar): Monto del ingreso. 
        
        SINTAXIS SQL: Debes usar sintaxis T-SQL (SQL Server).
        """
        
        # 2. Instrucción para generar SOLO SQL (Paso 1)
        sql_prompt = f"""
        Eres un experto en generar consultas T-SQL robustas. Tu única tarea es generar la consulta SQL que se necesita para responder la pregunta del usuario.
        
        **Debes seguir 2 pasos strictos:**
        1. **GENERACIÓN SQL:** Genera ÚNICAMENTE la consulta SQL más precisa (T-SQL). **ENVUELVE EL CÓDIGO SQL EN BLOQUES MARKDOWN DE SQL (```sql...```) Y NADA MÁS.**
        2. **Limpieza de Datos:** Usa el campo 'Valor' directamente. (La limpieza será forzada en Python).
        
        **ESQUEMA DE BD DISPONIBLE:**
        {schema_info}
        Pregunta del usuario: {pregunta_usuario}
        """

        try:
            with st.spinner('1/2: Generando y ejecutando la consulta SQL...'):
                
                # --- LLAMADA 1: Generar solo SQL ---
                response_sql = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[sql_prompt]
                )

                full_response_text = response_sql.text
                sql_match = re.search(r"```sql(.*?)```", full_response_text, re.DOTALL)
            
                if sql_match:
                    extracted_sql = sql_match.group(1).strip()
                    st.subheader("Consulta SQL Generada y Ejecutada:")
                    st.code(extracted_sql, language="sql")
                    
                    # --- FIX PERMANENTE: Sustituir SUM/AVG(Valor) por la sintaxis robusta de limpieza ---
                    robust_sum_syntax = "SUM(CASE WHEN ISNUMERIC(Valor) = 1 THEN CAST(Valor AS FLOAT) ELSE 0 END)"
                    robust_avg_syntax = "AVG(CASE WHEN ISNUMERIC(Valor) = 1 THEN CAST(Valor AS FLOAT) ELSE 0 END)"
                    
                    cleaned_sql = re.sub(r'SUM\s*\(\s*Valor\s*\)', robust_sum_syntax, extracted_sql, flags=re.IGNORECASE)
                    cleaned_sql = re.sub(r'AVG\s*\(\s*Valor\s*\)', robust_avg_syntax, cleaned_sql, flags=re.IGNORECASE)
                    cleaned_sql = re.sub(r'CAST\s*\(\s*Valor\s*AS\s*[a-zA-Z]+\s*\)', robust_sum_syntax, cleaned_sql, flags=re.IGNORECASE)

                    # Ejecutar la consulta real (ahora limpia)
                    conn = st.connection("sql", type="sql")
                    df_result = conn.query(cleaned_sql, ttl=0)
                    
                    st.success("✅ Datos Reales Obtenidos:")
                    st.dataframe(df_result)
                    
                    # --- LLAMADA 2: Generar Análisis con datos reales ---
                    with st.spinner('2/2: Generando análisis ejecutivo con datos reales...'):
                        
                        analysis_prompt = f"""
                        Pregunta del usuario: {pregunta_usuario}
                        
                        A continuación se presenta el resultado de la consulta SQL ejecutada en la base de datos:
                        {df_result.to_markdown(index=False)}
                        
                        Utiliza estos datos para generar un Análisis Ejecutivo de alto nivel y una Recomendación Estratégica. No repitas la consulta SQL.
                        """
                        
                        response_analysis = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[analysis_prompt]
                        )
                    
                    st.subheader("Análisis de Gari Mind:")
                    st.markdown(response_analysis.text)
                    st.success("✅ Análisis Completado")

                else:
                    st.error("⛔ La IA no generó una consulta SQL válida (busque ```sql...```).")
                    st.markdown(full_response_text)
                
        except Exception as e:
            st.error(f"⛔ Error en la ejecución o procesamiento: {e}")
            st.stop()


# ==========================================
# PÁGINA 2: REPORTES EJECUTIVOS (FUNCIONAL)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Reporte de Variación de Ingresos")
    st.info("Reporte basado en la tabla 'stg.Ingresos_Detallados'.")

    # --- Conexión y Query SQL ---
    try:
        conn = st.connection("sql", type="sql")
        
        query = """
            SELECT 
                Fecha as fecha, 
                Valor as valor,
                Sucursal as sucursal
            FROM stg.Ingresos_Detallados
            ORDER BY Fecha
        """
        
        df = conn.query(query, ttl=600)
        
        # Procesamiento Pandas (Limpieza de datos - FIX de TypeError)
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce') 
        df.dropna(subset=['valor'], inplace=True) 
        
        df['mes_anio'] = df['fecha'].dt.strftime('%Y-%m')

    except Exception as e:
        st.error("⛔ Error al cargar los datos.")
        st.write(e)
        st.stop()

    # --- BARRERA DE FILTRO (DEFINICIÓN DE df_filtrado) ---
    st.sidebar.header("Filtros de Reporte")
    sucursales = ["Todas"] + list(df['sucursal'].unique())
    filtro_sucursal = st.sidebar.selectbox("Filtrar por Sucursal:", sucursales)

    df_filtrado = df.copy() 
    if filtro_sucursal != "Todas":
        df_filtrado = df[df['sucursal'] == filtro_sucursal]
    # --- FIN BARRERA DE FILTRO ---

    # --- Lógica de Variación y KPIs ---
    df_mensual = df_filtrado.groupby('mes_anio')['valor'].sum().reset_index()
    
    # FIX: Se corrige el error tipográfico df_mensura -> df_mensual
    df_mensual['variacion_pct'] = df_mensual['valor'].pct_change() * 100 
    df_mensual['variacion_pct'] = df_mensual['variacion_pct'].fillna(0)

    total_ventas = df_filtrado['valor'].sum()
    promedio_mensual = df_mensual['valor'].mean()
    ultima_variacion = df_mensual['variacion_pct'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Ingresos Totales", f"${total_ventas:,.0f}")
    col2.metric("Promedio Mensual", f"${promedio_mensual:,.0f}")
    col3.metric("Variación Último Mes", f"{ultima_variacion:.1f}%", delta=f"{ultima_variacion:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tendencia de Ingresos ($)")
        st.bar_chart(df_mensual.set_index('mes_anio')['valor'])

    with c2:
        st.subheader("Variación Porcentual (%)")
        st.bar_chart(df_mensual.set_index('mes_anio')['variacion_pct'])

    with st.expander("Ver tabla de datos detallada"):
        st.dataframe(df_mensual)

# ==========================================
# PÁGINA 3: MAPA DE DATOS (FUNCIONAL)
# ==========================================
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de la Base de Datos Dentisalud")
    st.subheader("🕵️ Explorador de Base de Datos")

    try:
        conn = st.connection("sql", type="sql")
        
        query_mapa = """
        SELECT TABLE_SCHEMA as Esquema, TABLE_NAME as Tabla 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME;
        """
        df_tablas = conn.query(query_mapa, ttl=600)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("📂 **Tablas Disponibles**")
            st.dataframe(df_tablas, use_container_width=True, height=500)

        with col2:
            st.write("🧪 **Probador de Datos**")
            lista = df_tablas["Esquema"] + "." + df_tablas["Tabla"]
            seleccion = st.selectbox("Elige una tabla:", lista)
            
            if st.button(f"Ver datos de {seleccion}"):
                try:
                    df = conn.query(f"SELECT TOP 50 * FROM {seleccion}", ttl=0)
                    st.success(f"✅ Acceso correcto: {len(df)} filas recuperadas")
                    st.balloons() 
                    st.dataframe(df)
                except Exception as e:
                    st.error("⛔ Sin permiso o tabla vacía")
                    st.write(e)

    except Exception as e:
        st.error("Error de conexión")
        st.write(e)
