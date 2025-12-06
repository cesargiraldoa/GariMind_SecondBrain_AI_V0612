import streamlit as st
import pandas as pd

# Configuración básica de la página
st.set_page_config(page_title="Prueba SQL Server", layout="wide")

st.title("🦷 Monitor Dentisalud - Prueba de Conexión")
st.markdown("---")

st.write("Estado de la conexión: 🟡 Esperando prueba...")

# Botón para ejecutar la prueba
if st.button("🔌 Conectar a Base de Datos"):
    try:
        # 1. ESTABLECER CONEXIÓN
        # Usamos "sql" porque en tus secrets pusiste [connections.sql]
        conn = st.connection("sql", type="sql")
        
        st.info("Intentando contactar al servidor 186.180.3.170...")

        # 2. CONSULTA (Query)
        # OJO GARI: CAMBIA 'NombreDeTuTablaReal' POR UNA TABLA REAL (Ej: Pacientes, Citas, Agenda)
        # Usamos 'TOP 5' porque es SQL Server (no usa LIMIT)
        # Esta consulta le pregunta al servidor su versión.
# No requiere permisos de tabla, así que SIEMPRE funciona si hay conexión.
query = "SELECT @@VERSION as Version;"
        
        # Ejecutar consulta
        df = conn.query(query, ttl=0)

        # 3. MOSTRAR RESULTADOS
        st.success("✅ ¡CONEXIÓN EXITOSA!")
        st.write(f"Se encontraron {len(df)} registros de prueba:")
        st.dataframe(df)

    except Exception as e:
        # Si falla, mostramos el error exacto
        st.error("❌ Ocurrió un error al conectar")
        st.warning("Detalle técnico del error:")
        st.code(e)
