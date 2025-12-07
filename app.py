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
# PÁGINA 1: CEREBRO (INICIO) - VERSIÓN FINAL ROBUSTA
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

    # --- Lógica de Procesamiento ---
    if boton_analizar and pregunta_usuario:
        
        # 1. Definición del Esquema (Prompt)
        schema_info = """
        Tabla: stg.Ingresos_Detallados
        Columnas clave: 
        - Fecha (string, DD/MM/YYYY): Fecha de la transacción.
        - Valor (nvarchar): Monto del ingreso. CONTIENE NÚMEROS PERO ES TIPO TEXTO.
        - Sucursal (string): Sede.
        
        SINTAXIS SQL: T-SQL (SQL Server).
        IMPORTANTE: NO intentes convertir el Valor en el SQL, usa 'Valor' normalmente. Python lo corregirá.
        """
        
        sql_prompt = f"""
        Genera ÚNICAMENTE la consulta SQL (T-SQL) para esta pregunta.
        Usa la columna 'Valor' para sumas o promedios.
        
        **REGLAS:**
        1. Envuelve el SQL en un bloque markdown: ```sql ... ```
        2. NO expliques nada. Solo código.
        
        Esquema: {schema_info}
        Pregunta: {pregunta_usuario}
        """

        try:
            with st.spinner('1/2: Generando consulta SQL...'):
                
                # --- LLAMADA API (Paso 1: SQL) ---
                response_sql = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=sql_prompt)])]
                )

                full_response_text = response_sql.text
                # Extraer SQL
                sql_match = re.search(r"```sql(.*?)```", full_response_text, re.DOTALL)
            
                if sql_match:
                    raw_sql = sql_match.group(1).strip()
                    
                    # --- CIRUGÍA DE CÓDIGO SQL (El Fix Definitivo) ---
                    # Esta expresión regular busca: SUM( [cualquier cosa] Valor ) y lo reemplaza por la versión segura.
                    # Captura casos como: SUM(Valor), SUM([Valor]), SUM(t.Valor), SUM(stg.Ingresos.Valor)
                    
                    robust_cast = "CAST(CASE WHEN ISNUMERIC(Valor) = 1 THEN Valor ELSE 0 END AS FLOAT)"
                    
                    # 1. Reemplazo inteligente de SUM(...)
                    # Busca SUM seguido de paréntesis, opcionalmente alias/corchetes, palabra Valor, cierre paréntesis
                    pattern_sum = r"SUM\s*\(\s*(?:[\w\[\]]+\.)?\[?Valor\]?\s*\)"
                    cleaned_sql = re.sub(pattern_sum, f"SUM({robust_cast})", raw_sql, flags=re.IGNORECASE)

                    # 2. Reemplazo inteligente de AVG(...)
                    pattern_avg = r"AVG\s*\(\s*(?:[\w\[\]]+\.)?\[?Valor\]?\s*\)"
                    cleaned_sql = re.sub(pattern_avg, f"AVG({robust_cast})", cleaned_sql, flags=re.IGNORECASE)
                    
                    # 3. Reemplazo para ORDER BY Valor (si intenta ordenar por texto)
                    # Si ordena por SUM(Valor), el paso 1 ya lo arregló. Si ordena por Valor directo:
                    pattern_order = r"ORDER BY\s+(?:[\w\[\]]+\.)?\[?Valor\]?"
                    cleaned_sql = re.sub(pattern_order, f"ORDER BY {robust_cast}", cleaned_sql, flags=re.IGNORECASE)

                    st.subheader("Consulta Ejecutada (Corregida automáticamente):")
                    st.code(cleaned_sql, language="sql")
                    
                    # Ejecutar SQL Limpio
                    conn = st.connection("sql", type="sql")
                    df_result = conn.query(cleaned_sql, ttl=0)
                    
                    st.success("✅ Datos Reales Obtenidos:")
                    st.dataframe(df_result)
                    
                    # --- LLAMADA API (Paso 2: Análisis) ---
                    with st.spinner('2/2: Analizando datos...'):
                        analysis_prompt = f"""
                        Pregunta: {pregunta_usuario}
                        Datos reales:
                        {df_result.to_markdown(index=False)}
                        
                        Genera un Análisis Ejecutivo y Recomendación breve.
                        """
                        response_analysis = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[types.Content(role="user", parts=[types.Part.from_text(text=analysis_prompt)])]
                        )
                    
                    st.subheader("Análisis de Gari Mind:")
                    st.markdown(response_analysis.text)

                else:
                    st.error("⛔ La IA no generó SQL válido.")
                    st.text(full_response_text)
                
        except Exception as e:
            st.error(f"⛔ Error: {e}")
            st.info("Intenta reformular la pregunta o verifica que la base de datos esté activa.")
            st.stop()


# ==========================================
# PÁGINA 2: REPORTES EJECUTIVOS (CORREGIDA)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Reporte de Variación de Ingresos")
    st.info("Reporte basado en 'stg.Ingresos_Detallados'")

    try:
        conn = st.connection("sql", type="sql")
        
        # SQL Manual corregido con CAST para evitar el error nvarchar
        query = """
            SELECT 
                Fecha as fecha, 
                CASE WHEN ISNUMERIC(Valor) = 1 THEN CAST(Valor AS FLOAT) ELSE 0 END as valor,
                Sucursal as sucursal
            FROM stg.Ingresos_Detallados
            ORDER BY Fecha
        """
        
        df = conn.query(query, ttl=600)
        
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        df['mes_anio'] = df['fecha'].dt.strftime('%Y-%m')

    except Exception as e:
        st.error("⛔ Error al cargar datos.")
        st.write(e)
        st.stop()

    st.sidebar.header("Filtros")
    sucursales = ["Todas"] + list(df['sucursal'].unique())
    filtro = st.sidebar.selectbox("Sucursal:", sucursales)

    df_filt = df.copy()
    if filtro != "Todas":
        df_filt = df[df['sucursal'] == filtro]

    df_grp = df_filt.groupby('mes_anio')['valor'].sum().reset_index()
    df_grp['var_pct'] = df_grp['valor'].pct_change().fillna(0) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ingresos", f"${df_filt['valor'].sum():,.0f}")
    col2.metric("Promedio Mes", f"${df_grp['valor'].mean():,.0f}")
    col3.metric("Última Var.", f"{df_grp['var_pct'].iloc[-1]:.1f}%")

    st.divider()
    c1, c2 = st.columns(2)
    c1.subheader("Tendencia ($)")
    c1.bar_chart(df_grp.set_index('mes_anio')['valor'])
    c2.subheader("Variación (%)")
    c2.bar_chart(df_grp.set_index('mes_anio')['var_pct'])


# ==========================================
# PÁGINA 3: MAPA DE DATOS
# ==========================================
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de Datos")
    try:
        conn = st.connection("sql", type="sql")
        df_tablas = conn.query("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", ttl=600)
        
        c1, c2 = st.columns([1,2])
        c1.dataframe(df_tablas)
        
        tabla = c2.selectbox("Tabla:", df_tablas['TABLE_SCHEMA'] + "." + df_tablas['TABLE_NAME'])
        if c2.button("Ver Muestra"):
            # Usamos TOP 50 porque es SQL Server
            df = conn.query(f"SELECT TOP 50 * FROM {tabla}", ttl=0)
            st.success(f"✅ {len(df)} filas")
            st.balloons()
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error: {e}")
