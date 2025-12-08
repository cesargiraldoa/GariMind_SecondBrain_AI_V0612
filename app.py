import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- HERRAMIENTAS Y DICCIONARIOS ---
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
dias_es = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# --- ESTILOS CSS ---
def color_negative_red(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4b4b; font-weight: bold'
    return 'color: black'

# --- FUNCIÓN GRÁFICA ESTÁNDAR ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#3498db', formato='dinero'):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Barras
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='none', alpha=0.9)
    
    # Etiquetas Verticales
    fmt = '${:,.0f}' if formato == 'dinero' else '{:,.0f}'
    ax.bar_label(bars, fmt=fmt, padding=3, rotation=90, fontsize=10, fontweight='bold', color='#2c3e50')
    
    # Techo gráfico
    if not df_g.empty:
        y_max = df_g[y_col].max()
        ax.set_ylim(0, y_max * 1.4)
    
    # Limpieza visual
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
        # 1. Cargar Datos SQL
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        # Columnas BI
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaNum'] = df['Fecha'].dt.dayofweek
        df['Dia'] = df['DiaNum'].map(dias_es)
        df['Tx'] = 1 
        
        # 2. DATOS MAESTROS ZONAS (HARDCODED) - Para no depender del archivo
        datos_zonas = {
            'CLINICAS': ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'PASEO VILLA DEL RIO', 'CENTRO MAYOR', 'MULTIPLAZA', 'SALITRE', 'UNICENTRO', 'ITAGUI', 'LA PLAYA', 'POBLADO', 'CALI CIUDAD JARDIN', 'CALLE 80', 'GRAN ESTACION', 'CEDRITOS', 'PORTAL 80', 'CENTRO', 'VILLAVICENCIO', 'KENNEDY', 'ROMA', 'VILLAS', 'ALAMOS', 'CALI AV 6TA', 'MALL PLAZA BOGOTA', 'CALI CALIMA', 'PLAZA DE LAS AMERICAS', 'SUBA PLAZA IMPERIAL', 'MALL PLAZA BARRANQUILLA', 'LA FLORESTA', 'PALMIRA', 'RESTREPO', 'MALL PLAZA CALI'], 
            'ZONA': ['ZONA 4', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 5', 'ZONA 3', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 4', 'ZONA 4', 'ZONA 2', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 1'], 
            'CIUDAD': ['BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'MEDELLÍN', 'MEDELLÍN', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'VILLAVICENCIO', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BARRANQUILLA', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'CALI'], 
            'RED': ['PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'FRANQUICIA', 'FRANQUICIA', 'FRANQUICIA', 'PROPIA']
        }
        df_zonas = pd.DataFrame(datos_zonas)
        
        # 3. Merge
        try:
            df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
            df_zonas['CLINICAS'] = df_zonas['CLINICAS'].str.upper().str.strip()
            
            df_final = df.merge(df_zonas, left_on='Sucursal_Upper', right_on='CLINICAS', how='left')
            
            # Relleno de nulos
            df_final['ZONA'] = df_final['ZONA'].fillna('Sin Zona')
            df_final['CIUDAD'] = df_final['CIUDAD'].fillna('Otras')
            df_final['RED'] = df_final['RED'].fillna('No Def')
            
        except Exception as e:
            st.warning(f"Error cruce zonas: {e}")
            df_final = df
            df_final['ZONA'] = 'General'
            df_final['CIUDAD'] = 'General'
            df_final['RED'] = 'General'

        return df_final
        
    except Exception as e:
        st.error(f"Error Carga: {e}")
        return pd.DataFrame()

# --- FUNCIÓN CHAT ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        buffer = io.StringIO()
        
        # Solo enviamos info relevante para ahorrar tokens
        cols_export = ['Fecha', 'Sucursal', 'Valor', 'ZONA', 'CIUDAD', 'RED'] 
        cols_existentes = [c for c in cols_export if c in df.columns]
        
        df[cols_existentes].head(5).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        prompt_system = """
        Eres Gari. Usa 'df'. Reglas:
        1. Columna fecha 'Fecha'.
        2. Negocio: 'ZONA', 'CIUDAD', 'Sucursal', 'Valor', 'RED'.
        3. Outputs obligatorios: 'resultado' (str), 'tabla_resultados' (df), 'fig' (plt).
        4. Código Python puro.
        """
        prompt_user = f"Info: {info_cols}\nMuestra: {muestra}\nPregunta: {pregunta}\nCode only."
        
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}], temperature=0)
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker, 'meses_es': meses_es}
        exec(codigo, globals(), local_vars)
        return (local_vars.get('resultado', None), local_vars.get('fig', None), local_vars.get('tabla_resultados', None), codigo)
    except Exception as e: return f"Error: {e}", None, None, ""

# --- INTERFAZ PRINCIPAL ---
st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=120, caption="Gari 🐹")
pagina = st.sidebar.radio("Navegación", ["🧠 Chat con Gari", "📊 Reportes Ejecutivos BI", "🗺️ Mapa"])

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 API Key:", type="password")

with st.spinner("Cargando cerebro de Gari..."):
    df_raw = cargar_datos_integrados()

# ==============================================================================
# PÁGINA 1: CHAT
# ==============================================================================
if pagina == "🧠 Chat con Gari":
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### Haz preguntas a tus datos")
    
    if not df_raw.empty:
        st.caption(f"Datos hasta: {df_raw['Fecha'].max().strftime('%d/%m/%Y')}")
        pregunta = st.text_input("Consulta:", "Cual fue la Zona con mayor venta en 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Pensando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df_raw, pregunta, api_key)
                    if res_txt: st.success(f"📌 {res_txt}")
                    else: st.warning("Sin respuesta textual.")
                    if res_tabla is not None: st.dataframe(res_tabla)
                    if res_fig: st.pyplot(res_fig)
                    with st.expander("Ver código"): st.code(cod)
            else: st.error("Falta API Key")

# ==============================================================================
# PÁGINA 2: REPORTES BI
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos BI":
    st.title("📊 Tablero de Comando Gerencial")
    
    if not df_raw.empty:
        
        # --- FILTROS GLOBALES ---
        with st.expander("🔍 Filtros Globales (Zona / Ciudad / Red)", expanded=True):
            c_f1, c_f2, c_f3 = st.columns(3)
            
            # Filtros encadenados
            opc_zona = sorted(df_raw['ZONA'].astype(str).unique())
            sel_zona = c_f1.multiselect("Zona", opc_zona)
            
            df_temp = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw
            opc_ciudad = sorted(df_temp['CIUDAD'].astype(str).unique())
            sel_ciudad = c_f2.multiselect("Ciudad", opc_ciudad)
            
            df_temp2 = df_temp[df_temp['CIUDAD'].isin(sel_ciudad)] if sel_ciudad else df_temp
            opc_red = sorted(df_temp2['RED'].astype(str).unique())
            sel_red = c_f3.multiselect("Red", opc_red)

        # Aplicar Filtros al DataFrame Vista (df_view)
        df_view = df_raw.copy()
        if sel_zona: df_view = df_view[df_view['ZONA'].isin(sel_zona)]
        if sel_ciudad: df_view = df_view[df_view['CIUDAD'].isin(sel_ciudad)]
        if sel_red: df_view = df_view[df_view['RED'].isin(sel_red)]
            
        if df_view.empty:
            st.warning("⚠️ Sin datos para estos filtros.")
            st.stop()

        # --- SELECTOR DE MÉTRICA ---
        st.markdown("---")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            metrica_grafico = st.radio("📊 Métrica:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
            col_kpi = 'Valor' if metrica_grafico == "Ventas ($)" else 'Tx'
            fmt_kpi = 'dinero' if metrica_grafico == "Ventas ($)" else 'numero'
            color_kpi = '#3498db' if metrica_grafico == "Ventas ($)" else '#8e44ad'

        # --- 1. PULSO DEL NEGOCIO ---
        st.header("1. Pulso del Negocio (YTD)")
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
        
        # --- 2. ANÁLISIS GLOBAL ---
        st.header(f"2. Análisis Global {anio_actual}")
        
        # A) Gráfico Zonas (Solo si no filtro por una sola zona)
        if not sel_zona or len(sel_zona) > 1:
            st.subheader("A. Desempeño por Zona")
            df_z = df_actual.groupby('ZONA')[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False)
            fig_z = graficar_barras_pro(df_z, 'ZONA', col_kpi, 'Ranking Zonas', '#e67e22', fmt_kpi)
            st.pyplot(fig_z)

        # B) Mensual y Semanal
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

        # --- 3. ANÁLISIS POR CLÍNICA (RECUPERADO) ---
        st.markdown("---")
        st.header("🏥 3. Análisis por Clínica (Detalle)")
        st.info("Despliega cada clínica para ver su detalle individual. Responde a los filtros globales.")

        # Obtenemos las sucursales FILTRADAS (no todas las raw)
        sucursales_filtradas = sorted(df_actual['Sucursal'].unique())
        
        if not sucursales_filtradas:
            st.warning("No hay clínicas para mostrar con los filtros actuales.")

        for suc in sucursales_filtradas:
            # Info extra para el título del expander
            info_suc = df_actual[df_actual['Sucursal'] == suc].iloc[0]
            label_zona = info_suc.get('ZONA', 'N/A')
            
            with st.expander(f"📍 {suc} ({label_zona})", expanded=False):
                df_suc = df_actual[df_actual['Sucursal'] == suc]
                
                c1, c2 = st.columns(2)
                
                # Gráficos Dinámicos de la Sucursal
                with c1:
                    df_s_mes = df_suc.groupby('MesNum').agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
                    df_s_mes['Mes'] = df_s_mes['MesNum'].map(meses_es)
                    fig_sm = graficar_barras_pro(df_s_mes, 'Mes', col_kpi, 'Mensual', color_kpi, fmt_kpi)
                    st.pyplot(fig_sm)
                    
                with c2:
                    df_s_dia = df_suc.groupby(['DiaNum', 'Dia']).agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index().sort_values('DiaNum')
                    fig_sd = graficar_barras_pro(df_s_dia, 'Dia', col_kpi, 'Semanal', '#2ecc71', fmt_kpi)
                    st.pyplot(fig_sd)
                
                # Tabla Detallada con Variaciones
                df_s_mes['Var $'] = df_s_mes['Valor'].pct_change() * 100
                df_s_mes['Var Tx'] = df_s_mes['Tx'].pct_change() * 100
                
                st.table(
                    df_s_mes[['Mes', 'Valor', 'Var $', 'Tx', 'Var Tx']].style
                    .format({
                        "Valor": "${:,.0f}", "Var $": "{:+.1f}%", 
                        "Tx": "{:,.0f}", "Var Tx": "{:+.1f}%"
                    })
                    .applymap(color_negative_red, subset=['Var $', 'Var Tx'])
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
