import streamlit as st
import pandas as pd

# Configuración de página
st.set_page_config(page_title="Prueba Técnica", layout="wide")

st.title("🤖 Verificación de Sistema: Dentisalud")
st.markdown("---")

st.write("Presiona el botón para confirmar comunicación con el servidor.")

# Botón de prueba
if st.button("🔍 Verificar Conexión"):
    # Aquí empieza el bloque de seguridad "try"
    try:
        # 1. Conexión (Usando la configuración de Secrets)
        conn = st.connection("sql", type="sql")
        st.info("📡 Contactando al servidor 186.180.3.170...")

        # 2. Consulta de Diagnóstico 
        # Esta consulta NO requiere permisos especiales sobre tablas.
        # Solo le pregunta al servidor: "¿Quién eres?"
        query = "SELECT @@VERSION as Version_SQL;"
        
        # 3. Ejecución
        df = conn.query(query, ttl=0)
        
        # 4. Éxito
        st.success("✅ ¡CONEXIÓN TOTALMENTE EXITOSA!")
        st.write("El servidor respondió correctamente:")
        st.dataframe(df)

    # Este es el bloque "except" que faltaba antes
    except Exception as e:
        st.error("❌ Error en la ejecución")
        st.warning("Detalles técnicos:")
        st.code(e)
