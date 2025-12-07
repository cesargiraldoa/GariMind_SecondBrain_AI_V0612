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
# PÁGINA 1: CEREBRO (CON AUTO-CORRECCIÓN DE ERRORES)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    
    # 1. Iniciar Cliente de forma segura
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        st.error("⛔ Error de API Key. Verifica tus secrets.")
        st.stop()

    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #4B5563;">Asistente de Logística & Análisis de Datos</div>', unsafe_allow_html=True)
    st.divider()

    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta_usuario = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        boton_analizar = st.button("Analizar", type="primary", use_container_width=True)

    if boton_analizar and pregunta_usuario:
        
        # --- PASO 1: Generar SQL (Prompt Simplificado) ---
        with st.spinner('Generando consulta...'):
            sql_prompt = f"""
            Genera código SQL Server (T-SQL) para: "{pregunta_usuario}"
            Tabla: stg.Ingresos_Detallados
            Columnas: Fecha (string), Valor (NVARCHAR), Sucursal (string).
            
            IMPORTANTE: Solo responde con el código SQL dentro de bloques ```sql ... ```. No des explicaciones.
            """
            
            try:
                # FIX DE API: Enviamos string directo en una lista, sin objetos complejos
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[sql_prompt]
                )
                texto_respuesta = response.text
                
                # Extraer SQL
                match = re.search(r"```sql(.*?)```", texto_respuesta, re.DOTALL)
                if not match:
                    st.error("La IA no generó SQL válido.")
                    st.stop()
                
                sql_query = match.group(1).strip()
                
            except Exception as e:
                st.error(f"Error conectando con la IA: {e}")
                st.stop()

        # --- PASO 2: Ejecución con AUTO-CURACIÓN (Self-Healing) ---
        conn = st.connection("sql", type="sql")
        
        try:
            # Intento 1: Ejecutar tal cual viene
            df_result = conn.query(sql_query, ttl=0)
            
        except Exception as e:
            # Si falla, asumimos que es por el error de nvarchar y aplicamos el parche
            # st.warning("⚠️ Detectado error de tipo de dato. Aplicando auto-corrección...")
            
            # Reemplazo agresivo: Cualquier mención a Valor se convierte en TRY_CAST
            # Evitamos reemplazar si ya tiene CAST
            if "CAST" not in sql_query:
                fixed_query = sql_query.replace("Valor", "TRY_CAST(Valor AS FLOAT)")
                # También arreglamos sumas comunes
                fixed_query = fixed_query.replace("SUM(TRY_CAST(Valor AS FLOAT))", "SUM(TRY_CAST(Valor AS FLOAT))") 
            else:
                fixed_query = sql_query

            try:
                # Intento 2: Ejecutar corregido
                df_result = conn.query(fixed_query, ttl=0)
                # st.success("✅ Auto-corrección exitosa.")
                sql_query = fixed_query # Actualizamos para mostrar la buena
            except Exception as e2:
                st.error(f"⛔ Error fatal de SQL: {e}")
                st.stop()
        
        # --- PASO 3: Mostrar Resultados ---
        st.subheader("Consulta Ejecutada:")
        st.code(sql_query, language="sql")
        
        st.success("✅ Datos Obtenidos:")
        st.dataframe(df_result)
        
        # --- PASO 4: Análisis Final ---
        with st.spinner('Analizando datos...'):
            analysis_prompt = f"Analiza estos datos brevemente para un ejecutivo logístico:\n{df_result.to_markdown()}"
            res_analysis = client.models.generate_content(model='gemini-2.5-flash', contents=[analysis_prompt])
            st.subheader("Análisis:")
            st.markdown(res_analysis.text)


# ==========================================
# PÁGINA 2: REPORTES (ESTABLE)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Variación de Ingresos")
    try:
        conn = st.connection("sql", type="sql")
        # Consulta manual blindada
        query = """
            SELECT Fecha, CASE WHEN ISNUMERIC(Valor)=1 THEN CAST(Valor AS FLOAT) ELSE 0 END as Valor, Sucursal
            FROM stg.Ingresos_Detallados WHERE ISDATE(Fecha)=1 ORDER BY Fecha
        """
        df = conn.query(query, ttl=600)
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        
        # Filtros y Gráficos
        sucursal = st.sidebar.selectbox("Sucursal", ["Todas"] + list(df['Sucursal'].unique()))
        if sucursal != "Todas": df = df[df['Sucursal'] == sucursal]
        
        df_g = df.groupby('Mes')['Valor'].sum().reset_index()
        df_g['Var'] = df_g['Valor'].pct_change().fillna(0)*100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"${df['Valor'].sum():,.0f}")
        c2.metric("Var. Mes", f"{df_g['Var'].iloc[-1]:.1f}%")
        
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
        t = st.selectbox("Tabla:", tabs['Tabla'])
        if st.button("Ver Muestra"):
            st.dataframe(conn.query(f"SELECT TOP 50 * FROM {t}", ttl=0))
    except Exception as e:
        st.error(f"Error: {e}")
