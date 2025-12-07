import streamlit as st
import pandas as pd
import time
import os
import re
from google import genai

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")

# --- MENÚ LATERAL ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# ==========================================
# PÁGINA 1: CEREBRO (VERSIÓN SIMPLIFICADA)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    
    # 1. Iniciar Cliente (Buscando la clave en variables de entorno)
    try:
        # Simplificación: No pasamos argumentos si la variable de entorno está bien puesta
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        st.error(f"⛔ Error iniciando cliente IA: {e}")
        st.stop()

    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.divider()

    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta_usuario = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        boton_analizar = st.button("Analizar", type="primary", use_container_width=True)

    if boton_analizar and pregunta_usuario:
        
        # --- PASO 1: Generar SQL ---
        with st.spinner('Pensando...'):
            sql_prompt = f"""
            Actúa como un experto en SQL Server (T-SQL).
            Genera una consulta SQL para responder: "{pregunta_usuario}"
            
            Esquema de Tabla: stg.Ingresos_Detallados
            Columnas: 
              - Fecha (dd/mm/yyyy)
              - Valor (ESTO ES TEXTO, DEBES CONVERTIRLO)
              - Sucursal
              
            Tu respuesta debe ser SOLAMENTE el código SQL dentro de bloques markdown ```sql ... ```.
            """
            
            try:
                # --- CORRECCIÓN CRÍTICA: Enviamos una lista de strings simples. 
                # Eliminamos types.Content y types.Part para evitar el error de argumentos.
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[sql_prompt]
                )
                
                texto_respuesta = response.text
                
                # Extraer SQL
                match = re.search(r"```sql(.*?)```", texto_respuesta, re.DOTALL)
                if not match:
                    st.error("La IA no generó SQL válido.")
                    st.write(texto_respuesta)
                    st.stop()
                
                sql_query = match.group(1).strip()
                
            except Exception as e:
                st.error(f"Error en la llamada a la IA: {e}")
                st.stop()

        # --- PASO 2: Ejecución y Auto-Corrección ---
        conn = st.connection("sql", type="sql")
        
        # PARCHE DE SEGURIDAD EN PYTHON:
        # Reemplazamos cualquier uso de "Valor" por su versión segura numéricamente
        # Esto asegura que aunque la IA olvide el CAST, Python lo arregla.
        if "TRY_CAST" not in sql_query:
            # Reemplaza 'Valor' por 'TRY_CAST(Valor AS FLOAT)' evitando romper palabras como 'Valor_Neto' si existieran
            # Usamos una sustitución simple que funciona para este caso específico
            sql_query_segura = sql_query.replace("Valor", "TRY_CAST(Valor AS FLOAT)")
        else:
            sql_query_segura = sql_query

        st.code(sql_query_segura, language="sql")
        
        try:
            df_result = conn.query(sql_query_segura, ttl=0)
            st.success("✅ Datos:")
            st.dataframe(df_result)
            
            # --- PASO 3: Análisis ---
            with st.spinner('Analizando...'):
                analysis_prompt = f"Analiza estos datos brevemente: {pregunta_usuario}\n\n{df_result.to_markdown()}"
                
                # Llamada simplificada nuevamente
                res_analysis = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=[analysis_prompt]
                )
                st.markdown(res_analysis.text)
                
        except Exception as e:
            st.error(f"Error SQL: {e}")

# ==========================================
# PÁGINA 2: REPORTES (ESTABLE)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Variación de Ingresos")
    try:
        conn = st.connection("sql", type="sql")
        # Consulta manual blindada
        query = """
            SELECT 
                Fecha, 
                CASE WHEN ISNUMERIC(Valor)=1 THEN CAST(Valor AS FLOAT) ELSE 0 END as Valor, 
                Sucursal
            FROM stg.Ingresos_Detallados 
            ORDER BY Fecha
        """
        df = conn.query(query, ttl=600)
        
        # Limpieza Pandas
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        
        # Filtros
        sucursal = st.sidebar.selectbox("Sucursal", ["Todas"] + list(df['Sucursal'].unique()))
        if sucursal != "Todas": df = df[df['Sucursal'] == sucursal]
        
        # KPI
        df_g = df.groupby('Mes')['Valor'].sum().reset_index()
        df_g['Var'] = df_g['Valor'].pct_change().fillna(0)*100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"${df['Valor'].sum():,.0f}")
        c2.metric("Promedio", f"${df_g['Valor'].mean():,.0f}")
        c3.metric("Última Var.", f"{df_g['Var'].iloc[-1]:.1f}%")
        
        st.bar_chart(df_g.set_index('Mes')['Valor'])
        
    except Exception as e:
        st.error(f"Error cargando reportes: {e}")

# ==========================================
# PÁGINA 3: MAPA (ESTABLE)
# ==========================================
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de Datos")
    try:
        conn = st.connection("sql", type="sql")
        tabs = conn.query("SELECT TABLE_SCHEMA+'.'+TABLE_NAME as Tabla FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", ttl=600)
        
        c1, c2 = st.columns([1,2])
        c1.dataframe(tabs)
        
        t = c2.selectbox("Tabla:", tabs['Tabla'])
        if c2.button("Ver Muestra"):
            df = conn.query(f"SELECT TOP 50 * FROM {t}", ttl=0)
            st.success("Conectado")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error: {e}")
