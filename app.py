import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explorador SQL", layout="wide")

st.title("🗺️ Mapa de la Base de Datos Dentisalud")
st.markdown("---")

try:
    # 1. CONEXIÓN
    # Usamos la conexión que ya sabemos que funciona
    conn = st.connection("sql", type="sql")
    
    # 2. OBTENER LISTA DE TABLAS
    # Consultamos el "catálogo maestro" del servidor (INFORMATION_SCHEMA)
    st.info("🔄 Escaneando base de datos...")
    
    query_mapa = """
    SELECT 
        TABLE_SCHEMA as Esquema, 
        TABLE_NAME as Tabla 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_NAME;
    """
    
    # Ejecutamos la consulta del mapa
    df_tablas = conn.query(query_mapa, ttl=0)
    
    # 3. INTERFAZ DE EXPLORACIÓN
    col1, col2 = st.columns([1, 2])
    
    # Columna Izquierda: La Lista
    with col1:
        st.success(f"✅ Se encontraron {len(df_tablas)} tablas.")
        st.dataframe(df_tablas, height=500, use_container_width=True)

    # Columna Derecha: El Probador
    with col2:
        st.subheader("🧪 Probador de Permisos")
        st.write("Selecciona una tabla de la lista para intentar leerla:")
        
        # Crear una lista desplegable combinando Esquema y Nombre (ej: dbo.Pacientes)
        lista_opciones = df_tablas["Esquema"] + "." + df_tablas["Tabla"]
        tabla_seleccionada = st.selectbox("Selecciona tabla:", lista_opciones)
        
        # Botón para probar acceso a esa tabla específica
        if st.button(f"🔍 Espiar {tabla_seleccionada}"):
            try:
                # Intentamos traer solo 5 filas para ver si tenemos permiso
                query_prueba = f"SELECT TOP 5 * FROM {tabla_seleccionada};"
                df_preview = conn.query(query_prueba, ttl=0)
                
                # Si llega aquí, es un éxito
                st.balloons()
                st.success(f"¡BINGO! ✅ Tienes acceso a '{tabla_seleccionada}'")
                st.write("Primeras 5 filas de datos:")
                st.dataframe(df_preview)
                
            except Exception as e:
                # Si falla, mostramos el error sin romper la app
                st.error(f"⛔ Acceso Denegado a {tabla_seleccionada}")
                st.warning("El servidor dice: 'No tienes permiso SELECT o la tabla está vacía'")

except Exception as e:
    st.error("❌ Error general de conexión")
    st.code(e)
