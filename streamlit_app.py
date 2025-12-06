import streamlit as st

st.set_page_config(page_title="Monitor Logístico", layout="wide")

st.title("🚧 Panel de Control - Prueba de Conexión")
st.write("Verificando acceso a la base de datos...")

# Botón para ejecutar la prueba
if st.button("Iniciar Test de Conexión"):
    try:
        # 1. CONEXIÓN
        # IMPORTANTE: El nombre dentro de connection() debe coincidir con el de tus secrets.
        # Si en secrets pusiste [connections.mysql], aquí va "mysql".
        # Si no estás seguro, usa el nombre genérico o revisa tu archivo secrets.
        conn = st.connection("mysql", type="sql") 
        
        st.info("Intentando contactar al servidor...")

        # 2. CONSULTA (Query)
        # IMPORTANTE: Cambia 'nombre_de_tu_tabla' por una tabla real de tu base de datos
        # (Ej: ventas, pedidos, stock, logistica)
        query = "SELECT * FROM dbo.Usuarios LIMIT 5;"
        
        df = conn.query(query, ttl=0)

        # 3. RESULTADO
        st.success("✅ ¡Conexión EXITOSA!")
        st.write("Primeras 5 filas de datos recibidas:")
        st.dataframe(df)

    except Exception as e:
        st.error("❌ Error en la conexión")
        st.warning("Detalles del error (copia esto si necesitas ayuda):")
        st.code(e)
