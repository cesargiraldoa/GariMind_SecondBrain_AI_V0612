import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- HERRAMIENTAS AUXILIARES ---
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
dias_es = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# --- ESTILOS CSS PARA TABLAS ---
def color_negative_red(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4b4b; font-weight: bold'
    return 'color: black'

# --- FUNCIÓN GRÁFICA ESTÁNDAR ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#3498db', formato='dinero'):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Barras
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='none', alpha=0.9)
    
    # Etiquetas
    fmt = '${:,.0f}' if formato == 'dinero' else '{:,.0f}'
    ax.bar_label(bars, fmt=fmt, padding=3, rotation=90, fontsize=10, fontweight='bold', color='#2c3e50')
    
    # Ajustes visuales
    if not df_g.empty:
        y_max = df_g[y_col].max()
        ax.set_ylim(0, y_max * 1.4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.set_title(titulo, fontsize=13, fontweight='bold', color='#2c3e50', pad=20)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    return fig

# --- CARGA DE DATOS (SQL + ZONAS INCRUSTADAS) ---
@st.cache_data(ttl=600)
def cargar_datos_integrados():
    df_final = pd.DataFrame()
    try:
        # 1. Cargar Datos Transaccionales desde SQL
        # Asegúrate de tener configurado .streamlit/secrets.toml con la sección [connections.sql]
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza y conversión de tipos
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        # Crear columnas de tiempo
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaNum'] = df['Fecha'].dt.dayofweek
        df['Dia'] = df['DiaNum'].map(dias_es)
        df['Tx'] = 1 
        
        # 2. DEFINIR DATOS MAESTROS DE ZONAS (HARDCODED)
        # Esto elimina la dependencia del archivo externo
        datos_zonas = {
            'CLINICAS': ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'PASEO VILLA DEL RIO', 'CENTRO MAYOR', 'MULTIPLAZA', 'SALITRE', 'UNICENTRO', 'ITAGUI', 'LA PLAYA', 'POBLADO', 'CALI CIUDAD JARDIN', 'CALLE 80', 'GRAN ESTACION', 'CEDRITOS', 'PORTAL 80', 'CENTRO', 'VILLAVICENCIO', 'KENNEDY', 'ROMA', 'VILLAS', 'ALAMOS', 'CALI AV 6TA', 'MALL PLAZA BOGOTA', 'CALI CALIMA', 'PLAZA DE LAS AMERICAS', 'SUBA PLAZA IMPERIAL', 'MALL PLAZA BARRANQUILLA', 'LA FLORESTA', 'PALMIRA', 'RESTREPO', 'MALL PLAZA CALI'], 
            'ZONA': ['ZONA 4', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 5', 'ZONA 3', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 4', 'ZONA 4', 'ZONA 2', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 1'], 
            'CIUDAD': ['BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'MEDELLÍN', 'MEDELLÍN', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'VILLAVICENCIO', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BARRANQUILLA', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'CALI'], 
            'RED': ['PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'FRANQUICIA', 'FRANQUICIA', 'FRANQUICIA', 'PROPIA']
        }
        df_zonas = pd.DataFrame(datos_zonas)
        
        # 3. FUSIÓN DE DATOS (MERGE)
        try:
            # Normalizar nombres para asegurar coincidencias
            df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
            df_zonas['CLINICAS'] = df_zonas['CLINICAS'].str.upper().str.strip()
            
            # Cruce Left Join
            df_final = df.merge(df_zonas, left_on='Sucursal_Upper', right_on='CLINICAS', how='left')
            
            # Rellenar nulos para sucursales que no estén en el maestro
            df_final['ZONA'] = df_final['ZONA'].fillna('Sin Zona')
            df_final['CIUDAD'] = df_final['CIUDAD'].fillna('Otras')
            df_final['RED'] = df_final['RED'].fillna('No Def')
            
        except Exception as e:
            st.warning(f"Error en el cruce de datos: {e}")
            # Fallback seguro
            df_final = df
            df_final['ZONA'] = 'General'
            df_final['CIUDAD'] = 'General'
            df_final['RED'] = 'General'

        return df_final
        
    except Exception as e:
        st.error(f"Error General de Carga: {e}")
        return pd.DataFrame()

# --- FUNCIÓN CHAT ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        buffer = io.StringIO()
        
        # Seleccionamos columnas relevantes para el contexto
        cols_export = ['Fecha', 'Sucursal', 'Valor', 'ZONA', 'CIUDAD', 'RED'] 
        cols_existentes = [c for c in cols_export if c in df.columns]
        
        df[cols_existentes].head(5).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        prompt_system = """
        Eres Gari, un analista de datos experto en Python. Tienes un dataframe llamado 'df'.
        Reglas:
        1. La columna de fecha es 'Fecha'.
        2. Tienes columnas de negocio: 'ZONA', 'CIUDAD', 'Sucursal', 'Valor' (Venta), 'RED'.
        3. Debes generar código Python ejecutable que resuelva la pregunta.
        4. Tus outputs finales deben asignarse a las variables: 
           - 'resultado' (texto respuesta)
           - 'tabla_resultados' (dataframe pandas, si aplica)
           - 'fig' (figura matplotlib, si aplica)
        """
        prompt_user = f"Info Tipos:\n{info_cols}\n\nMuestra de Datos:\n{muestra}\n\nPregunta Usuario: {pregunta}\n\nGenera solo el código Python."
        
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}], temperature=0)
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker, 'meses_es': meses_es}
        exec(codigo, globals(), local_vars)
        return (local_vars.get('resultado', None), local_vars.get('fig', None), local_vars.get('tabla_resultados', None), codigo)
    except Exception as e: return f"Error: {e}", None, None, ""

# --- INTERFAZ DE USUARIO ---
st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=120, caption="Gari 🐹")
pagina = st.sidebar.radio("Navegación", ["🧠 Chat con Gari", "📊 Reportes Ejecutivos BI", "🗺️ Mapa"])

# Manejo de API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 API Key:", type="password")

# Carga de datos
with st.spinner("Cargando datos y aplicando zonas..."):
    df_raw = cargar_datos_integrados()

# ==============================================================================
# PÁGINA 1: CHAT
# ==============================================================================
if pagina == "🧠 Chat con Gari":
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### Haz preguntas a tus datos")
    
    if not df_raw.empty:
        st.caption(f"Datos actualizados al: {df_raw['Fecha'].max().strftime('%d/%m/%Y')}")
        pregunta = st.text_input("Consulta:", "Cual fue la Zona con mayor venta en 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df_raw, pregunta, api_key)
                    if res_txt: st.success(f"📌 {res_txt}")
                    else: st.warning("No pude generar una respuesta de texto.")
                    
                    if res_tabla is not None: st.dataframe(res_tabla)
                    if res_fig: st.pyplot(res_fig)
                    
                    with st.expander("Ver lógica Python"): st.code(cod)
            else: st.error("Falta API Key")

# ==============================================================================
# PÁGINA 2: REPORTES BI
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos BI":
    st.title("📊 Tablero de Comando Gerencial")
    
    if not df_raw.empty:
        
        # --- FILTROS ---
        with st.expander("🔍 Filtros Globales (Zona / Ciudad / Red)", expanded=True):
            c_f1, c_f2, c_f3 = st.columns(3)
            
            # Filtros dependientes
            opciones_zona = sorted(df_raw['ZONA'].astype(str).unique())
            sel_zona = c_f1.multiselect("Filtrar por Zona", opciones_zona)
            
            df_temp = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw
            opciones_ciudad = sorted(df_temp['CIUDAD'].astype(str).unique())
            sel_ciudad = c_f2.multiselect("Filtrar por Ciudad", opciones_ciudad)
            
            df_temp2 = df_temp[df_temp['CIUDAD'].isin(sel_ciudad)] if sel_ciudad else df_temp
            opciones_red = sorted(df_temp2['RED'].astype(str).unique())
            sel_red = c_f3.multiselect("Filtrar por Red", opciones_red)

        # Aplicar Filtros
        df_view = df_raw.copy()
        if sel_zona: df_view = df_view[df_view['ZONA'].isin(sel_zona)]
        if sel_ciudad: df_view = df_view[df_view['CIUDAD'].isin(sel_ciudad)]
        if sel_red: df_view = df_view[df_view['RED'].isin(sel_red)]
            
        if df_view.empty:
            st.warning("⚠️ No hay datos con los filtros seleccionados.")
            st.stop()

        # --- SELECTOR DE MÉTRICA ---
        st.markdown("---")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            metrica_grafico = st.radio("📊 Métrica Visual:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
            col_kpi = 'Valor' if metrica_grafico == "Ventas ($)" else 'Tx'
            fmt_kpi = 'dinero' if metrica_grafico == "Ventas ($)" else 'numero'
            color_kpi = '#3498db' if metrica_grafico == "Ventas ($)" else '#8e44ad'

        # --- KPI's ---
        st.header("1. Pulso del Negocio (Selección)")
        anio_actual = df_view['Año'].max()
        df_actual = df_view[df_view['Año'] == anio_actual]
        
        v_act = df_actual['Valor'].sum()
        tx_act = len(df_actual)
        tk = v_act / tx_act if tx_act > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Ventas {anio_actual}", f"${v_act:,.0f}")
        k2.metric(f"Transacciones", f"{tx_act:,}")
        k3.metric("Ticket Promedio", f"${tk:,.0f}")
        
        st.markdown("---")
        
        # --- GRÁFICOS ---
        st.header(f"2. Análisis Global {anio_actual}")
        
        # A) POR ZONA
        if not sel_zona: 
            st.subheader("A. Desempeño por Zona")
            df_zona_grp = df_actual.groupby('ZONA')[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False)
            fig_zona = graficar_barras_pro(df_zona_grp, 'ZONA', col_kpi, f'Top Zonas ({metrica_grafico})', '#e67e22', fmt_kpi)
            st.pyplot(fig_zona)

        # B) EVOLUCIÓN MENSUAL & C) SEMANAL
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.subheader("B. Evolución Mensual")
            df_mes = df_actual.groupby('MesNum').agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
            df_mes['Mes'] = df_mes['MesNum'].map(meses_es)
            fig_mes = graficar_barras_pro(df_mes, 'Mes', col_kpi, 'Mensual', color_kpi, fmt_kpi)
            st.pyplot(fig_mes)
        
        with c_g2:
            st.subheader("C. Patrón Semanal")
            df_dias = df_actual.groupby(['DiaNum', 'Dia']).agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index().sort_values('DiaNum')
            fig_dias = graficar_barras_pro(df_dias, 'Dia', col_kpi, 'Semanal', '#2ecc71', fmt_kpi)
            st.pyplot(fig_dias)

        # --- DETALLE ---
        st.markdown("---")
        st.header("🏥 Top Clínicas (Detalle)")
        top_clinicas = df_actual.groupby(['Sucursal', 'ZONA', 'CIUDAD', 'RED'])[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False).head(20)
        
        st.dataframe(
            top_clinicas.style.format({col_kpi: "${:,.0f}" if fmt_kpi == 'dinero' else "{:,.0f}"})
            .background_gradient(subset=[col_kpi], cmap="Blues")
        )

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
