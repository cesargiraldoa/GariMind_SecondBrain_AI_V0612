import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explorador SQL", layout="wide")
st.title("🕵️ Explorador de Base de Datos")

try:
    conn = st.connection("sql", type="sql")
    st.info("Conectado a Dentisalud")
    
    # 1. Mapa de Tablas
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
                # Top 50 para no saturar
                df = conn.query(f"SELECT TOP 50 * FROM {seleccion}", ttl=0)
                st.success(f"✅ Acceso correcto: {len(df)} filas recuperadas")
                st.dataframe(df)
            except Exception as e:
                st.error("⛔ Sin permiso o tabla vacía")
                st.write(e)

except Exception as e:
    st.error("Error de conexión")
    st.write(e)
