import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib.parse
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- GESTIÓN DE SESIÓN Y LOGIN ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    """Retorna True si el usuario/clave son correctos."""
    def login_form():
        st.title("🔒 Acceso Seguro - Gari AI")
        st.write("Por favor, inicie sesión para ver los datos de Dentisalud.")
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            usuario = st.text_input("Usuario")
            clave = st.text_input("Contraseña", type="password")
            
            if st.button("Ingresar 🔐"):
                # --- USUARIOS CONFIGURADOS ---
                usuarios_validos = {
                    "gerente": "alivio2025", 
                    "admin": "admin123",
                    "gari": "hamster"
                }
                
                if usuario in usuarios_validos and usuarios_validos[usuario] == clave:
                    st.session_state.authenticated = True
                    st.success("Acceso concedido. Cargando...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")

    if not st.session_state.authenticated:
        login_form()
        return False
    return True

# --- SI NO ESTÁ AUTENTICADO, SE DETIENE AQUÍ ---
if not check_password():
    st.stop()

# ==============================================================================
# 🚀 COMIENZO DE LA APLICACIÓN (SOLO VISIBLE SI LOGIN OK)
# ==============================================================================

# --- LOGOUT EN SIDEBAR ---
st.sidebar.markdown(f"👤 **Usuario:** Conectado")
if st.sidebar.button("Cerrar Sesión 🔒"):
    st.session_state.authenticated = False
    st.rerun()
st.sidebar.markdown("---")

# --- HERRAMIENTAS Y DICCIONARIOS ---
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

dias_es = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# --- ESTILOS CSS ---
def color_negative_red(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4b4b; font-weight: bold'
    return 'color: black'

# --- FUNCIÓN GRÁFICA ESTÁNDAR ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#3498db', formato='dinero'):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='none', alpha=0.9)
    fmt = '${:,.0f}' if formato == 'dinero' else '{:,.0f}'
    ax.bar_label(bars, fmt=fmt, padding=3, rotation=90, fontsize=10, fontweight='bold', color='#2c3e50')
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

# --- FUNCIÓN REPORTE PMV COMPLETO (WHATSAPP) ---
# 🔥 OPTIMIZACIÓN CRÍTICA: @st.cache_data para evitar recalcular y colgar la app
@st.cache_data(show_spinner=False) 
def generar_reporte_pmv_whatsapp(df):
    try:
        if df.empty: return "https://wa.me/"
        anio_actual = df['Año'].max()
        df_act = df[df['Año'] == anio_actual].copy() # Usamos copy para no afectar original
        
        if df_act.empty: return "https://wa.me/"

        # Pre-calcular agrupamientos (Mucho más rápido que filtrar en bucle)
        # Nivel 1: Total
        v_total = df_act['Valor'].sum()
        tx_total = len(df_act)
        
        # Nivel 2: Agrupado por Zonas
        df_zonas = df_act.groupby('ZONA')['Valor'].sum().sort_values(ascending=False)
        
        # Nivel 3: Agrupado por Zona y Sucursal (Pre-calculado)
        df_detalle = df_act.groupby(['ZONA', 'Sucursal'])['Valor'].sum().reset_index()
        
        # --- CONSTRUCCIÓN DEL MENSAJE ---
        mensaje = f"*🐹 REPORTE PMV - DENTISALUD {anio_actual}*\n"
        mensaje += f"📅 Corte: {df_act['Fecha'].max().strftime('%d/%m/%Y')}\n\n"
        
        mensaje += f"🏢 *TOTAL COMPAÑÍA*\n"
        mensaje += f"💰 Venta: ${v_total:,.0f}\n"
        mensaje += f"🧾 Tx: {tx_total:,.0f}\n"
        mensaje += "➖➖➖➖➖➖➖➖\n" # Quitamos salto extra para ahorrar espacio URL

        # Iteración optimizada
        for zona, valor_zona in df_zonas.items():
            mensaje += f"\n📍 *{zona}*: ${valor_zona:,.0f}\n"
            
            # Filtramos sobre el dataframe PEQUEÑO pre-agrupado (rápido)
            sucursales_zona = df_detalle[df_detalle['ZONA'] == zona].sort_values('Valor', ascending=False)
            
            for _, row in sucursales_zona.iterrows():
                # Limite de seguridad: WhatsApp a veces falla con URLs muy largas.
                # Opcional: Podríamos poner un break aquí si son demasiadas clínicas.
                mensaje += f"   • {row['Sucursal']}: ${row['Valor']:,.0f}\n"

        mensaje += "\n_Generado por Gari AI_ 🐹"
        
        # Codificar URL (safe para caracteres especiales)
        mensaje_codificado = urllib.parse.quote(mensaje)
        return f"https://wa.me/?text={mensaje_codificado}"
        
    except Exception as e:
        print(f"Error generando reporte WA: {e}")
        return "https://wa.me/"

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_integrados():
    df_final = pd.DataFrame()
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaNum'] = df['Fecha'].dt.dayofweek
        df['Dia'] = df['DiaNum'].map(dias_es)
        df['Tx'] = 1 
        
        datos_zonas = {
            'CLINICAS': ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'PASEO VILLA DEL RIO', 'CENTRO MAYOR', 'MULTIPLAZA', 'SALITRE', 'UNICENTRO', 'ITAGUI', 'LA PLAYA', 'POBLADO', 'CALI CIUDAD JARDIN', 'CALLE 80', 'GRAN ESTACION', 'CEDRITOS', 'PORTAL 80', 'CENTRO', 'VILLAVICENCIO', 'KENNEDY', 'ROMA', 'VILLAS', 'ALAMOS', 'CALI AV 6TA', 'MALL PLAZA BOGOTA', 'CALI CALIMA', 'PLAZA DE LAS AMERICAS', 'SUBA PLAZA IMPERIAL', 'MALL PLAZA BARRANQUILLA', 'LA FLORESTA', 'PALMIRA', 'RESTREPO', 'MALL PLAZA CALI'], 
            'ZONA': ['ZONA 4', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 5', 'ZONA 3', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 4', 'ZONA 4', 'ZONA 2', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 1'], 
            'CIUDAD': ['BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'MEDELLÍN', 'MEDELLÍN', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'VILLAVICENCIO', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BARRANQUILLA', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'CALI'], 
            'RED': ['PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'FRANQUICIA', 'FRANQUICIA', 'FRANQUICIA', 'PROPIA']
        }
        df_zonas = pd.DataFrame(datos_zonas)
        
        try:
            df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
            df_zonas['CLINICAS'] = df_zonas['CLINICAS'].str.upper().str.strip()
            df_final = df.merge(df_zonas, left_on='Sucursal_Upper', right_on='CLINICAS', how='left')
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
        cols_export = ['Fecha', 'Sucursal', 'Valor', 'ZONA', 'CIUDAD', 'RED'] 
        cols_existentes = [c for c in cols_export if c in df.columns]
        df[cols_existentes].head(5).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        prompt_system = """
        Eres Gari. Output rules: 'resultado', 'tabla_resultados', 'fig'. Code only.
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
        # --- FILTROS ---
        with st.expander("🔍 Filtros Globales (Zona / Ciudad / Red)", expanded=True):
            c_f1, c_f2, c_f3 = st.columns(3)
            opc_zona = sorted(df_raw['ZONA'].astype(str).unique())
            sel_zona = c_f1.multiselect("Zona", opc_zona)
            df_temp = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw
            opc_ciudad = sorted(df_temp['CIUDAD'].astype(str).unique())
            sel_ciudad = c_f2.multiselect("Ciudad", opc_ciudad)
            df_temp2 = df_temp[df_temp['CIUDAD'].isin(sel_ciudad)] if sel_ciudad else df_temp
            opc_red = sorted(df_temp2['RED'].astype(str).unique())
            sel_red = c_f3.multiselect("Red", opc_red)

        df_view = df_raw.copy()
        if sel_zona: df_view = df_view[df_view['ZONA'].isin(sel_zona)]
        if sel_ciudad: df_view = df_view[df_view['CIUDAD'].isin(sel_ciudad)]
        if sel_red: df_view = df_view[df_view['RED'].isin(sel_red)]
            
        if df_view.empty:
            st.warning("⚠️ Sin datos para estos filtros.")
            st.stop()

        # --- KPI's ---
        st.markdown("---")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            metrica_grafico = st.radio("📊 Métrica:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
            col_kpi = 'Valor' if metrica_grafico == "Ventas ($)" else 'Tx'
            fmt_kpi = 'dinero' if metrica_grafico == "Ventas ($)" else 'numero'
            color_kpi = '#3498db' if metrica_grafico == "Ventas ($)" else '#8e44ad'

        st.header("1. Pulso del Negocio (Comparativo YTD)")
        anio_actual = df_view['Año'].max()
        anio_anterior = anio_actual - 1
        df_actual = df_view[df_view['Año'] == anio_actual]
        fecha_max_actual = df_actual['Fecha'].max()
        fecha_limite_anterior = fecha_max_actual.replace(year=anio_anterior)
        df_anterior = df_view[(df_view['Año'] == anio_anterior) & (df_view['Fecha'] <= fecha_limite_anterior)]
        
        v_act = df_actual['Valor'].sum()
        v_ant = df_anterior['Valor'].sum()
        delta_v = ((v_act - v_ant) / v_ant) * 100 if v_ant > 0 else 0
        tx_act = len(df_actual)
        tx_ant = len(df_anterior)
        delta_tx = ((tx_act - tx_ant) / tx_ant) * 100 if tx_ant > 0 else 0
        tk = v_act / tx_act if tx_act > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Ventas {anio_actual} (YTD)", f"${v_act:,.0f}", f"{delta_v:+.1f}%")
        k2.metric(f"Transacciones (YTD)", f"{tx_act:,}", f"{delta_tx:+.1f}%")
        k3.metric("Ticket Promedio", f"${tk:,.0f}")
        
        # --- TABLA COMPARATIVA ---
        st.subheader("📈 Tabla Comparativa Histórica")
        df_anual = df_view.groupby('Año').agg(Ventas=('Valor', 'sum'), Transacciones=('Tx', 'sum')).sort_index(ascending=False)
        df_anual['Crec. Ventas %'] = df_anual['Ventas'].pct_change(-1) * 100
        df_anual['Crec. Tx %'] = df_anual['Transacciones'].pct_change(-1) * 100
        st.table(df_anual[['Ventas', 'Crec. Ventas %', 'Transacciones', 'Crec. Tx %']].style.format({"Ventas": "${:,.0f}", "Transacciones": "{:,.0f}", "Crec. Ventas %": "{:+.1f}%", "Crec. Tx %": "{:+.1f}%"}).applymap(color_negative_red, subset=['Crec. Ventas %', 'Crec. Tx %']))
        
        # --- GRÁFICOS ---
        st.markdown("---")
        st.header(f"2. Análisis Global {anio_actual}")
        if not sel_zona or len(sel_zona) > 1:
            st.subheader("A. Desempeño por Zona")
            df_z = df_actual.groupby('ZONA')[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False)
            fig_z = graficar_barras_pro(df_z, 'ZONA', col_kpi, 'Ranking Zonas', '#e67e22', fmt_kpi)
            st.pyplot(fig_z)

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

        # --- DETALLE CLINICAS ---
        st.markdown("---")
        st.header("🏥 3. Análisis por Clínica (Detalle)")
        sucursales_filtradas = sorted(df_actual['Sucursal'].unique())
        for suc in sucursales_filtradas:
            info_suc = df_actual[df_actual['Sucursal'] == suc].iloc[0]
            label_zona = info_suc.get('ZONA', 'N/A')
            with st.expander(f"📍 {suc} ({label_zona})", expanded=False):
                df_suc = df_actual[df_actual['Sucursal'] == suc]
                c1, c2 = st.columns(2)
                with c1:
                    df_s_mes = df_suc.groupby('MesNum').agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
                    df_s_mes['Mes'] = df_s_mes['MesNum'].map(meses_es)
                    st.pyplot(graficar_barras_pro(df_s_mes, 'Mes', col_kpi, 'Mensual', color_kpi, fmt_kpi))
                with c2:
                    df_s_dia = df_suc.groupby(['DiaNum', 'Dia']).agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index().sort_values('DiaNum')
                    st.pyplot(graficar_barras_pro(df_s_dia, 'Dia', col_kpi, 'Semanal', '#2ecc71', fmt_kpi))
                df_s_mes['Var $'] = df_s_mes['Valor'].pct_change() * 100
                df_s_mes['Var Tx'] = df_s_mes['Tx'].pct_change() * 100
                st.table(df_s_mes[['Mes', 'Valor', 'Var $', 'Tx', 'Var Tx']].style.format({"Valor": "${:,.0f}", "Var $": "{:+.1f}%", "Tx": "{:,.0f}", "Var Tx": "{:+.1f}%"}).applymap(color_negative_red, subset=['Var $', 'Var Tx']))

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")

# ==============================================================================
# BOTÓN WHATSAPP (PMV)
# ==============================================================================
if not df_raw.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("📲 Reporte Gerencial")
    
    # Generamos el reporte usando la función cacheada
    link_wa = generar_reporte_pmv_whatsapp(df_raw)
    
    st.sidebar.markdown(f"""
    <a href="{link_wa}" target="_blank">
        <button style="
            background-color:#25D366; 
            color:white; 
            border:none; 
            padding:10px 20px; 
            border-radius:5px; 
            font-weight:bold; 
            cursor:pointer;
            width:100%;">
            Generar Reporte PMV 🚀
        </button>
    </a>
    <div style="text-align:center; margin-top:5px; font-size:0.8em; color:gray;">
        Reporte Completo: Compañía > Zonas > Clínicas
    </div>
    """, unsafe_allow_html=True)
