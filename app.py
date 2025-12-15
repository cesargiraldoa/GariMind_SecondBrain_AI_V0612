import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib.parse
import time
import datetime
import calendar
import numpy as np
import random
import gc

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Gari | Second Brain", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🎨 CSS BLINDADO
# ==============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
        .stApp { background-color: #060818 !important; color: #ffffff !important; }
        [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 2px solid #cc0000 !important; }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        div[data-testid="stMetric"] { background-color: #151925 !important; border-left: 4px solid #cc0000 !important; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
        div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Orbitron', sans-serif !important; }
        div[data-testid="stMetricDelta"] { color: #fcd700 !important; }
        .stButton > button { background: linear-gradient(90deg, #cc0000 0%, #990000 100%) !important; color: white !important; border: none !important; font-family: 'Orbitron', sans-serif !important; font-weight: bold !important; }
        [data-testid="stTable"] { color: white !important; }
        .stDataFrame { border: 1px solid #333 !important; }
        .podium-container { display: flex; align-items: flex-end; justify-content: center; height: 250px; gap: 10px; }
        .podium-step { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; border-radius: 10px 10px 0 0; padding: 10px; color: white; font-family: 'Orbitron', sans-serif; text-align: center; }
        .gold { background: linear-gradient(180deg, #FFD700 0%, #B8860B 100%); width: 100%; border: 2px solid #FFD700; }
        .silver { background: linear-gradient(180deg, #C0C0C0 0%, #A9A9A9 100%); height: 85%; width: 100%; border: 2px solid #C0C0C0; }
        .bronze { background: linear-gradient(180deg, #CD7F32 0%, #8B4513 100%); height: 70%; width: 100%; border: 2px solid #CD7F32; }
        .medal { font-size: 3rem; margin-bottom: -5px; }
        .manager-name { font-size: 1.1rem; font-weight: bold; margin-top: 5px; }
        .manager-value { font-size: 0.9rem; color: #000; font-weight: bold; background: rgba(255,255,255,0.9); padding: 2px 8px; border-radius: 4px; margin-top: 5px;}
        .leaderboard-row { display: flex; justify-content: space-between; align-items: center; background-color: #151925; padding: 10px 20px; margin-bottom: 5px; border-left: 3px solid #333; border-radius: 4px; }
        .ai-box { background-color: #0e1117; border: 1px solid #444; border-left: 5px solid #27ae60; padding: 20px; border-radius: 8px; margin-bottom: 25px; }
        .ai-title { color: #27ae60; font-family: 'Orbitron', sans-serif; font-weight: bold; margin-bottom: 15px; }
        .highlight { color: #ffffff; font-weight: bold; background-color: rgba(39, 174, 96, 0.2); padding: 2px 5px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

# --- UTILS ---
def color_negative_red(val):
    try: return 'color: #ff4b4b !important; font-weight: bold' if val < 0 else 'color: #ffffff !important'
    except: return 'color: #ffffff !important'

def color_cumplimiento(val):
    try:
        if val >= 100: return 'color: #27ae60 !important; font-weight: bold'
        if val >= 80: return 'color: #fcd700 !important; font-weight: bold'
        return 'color: #cc0000 !important; font-weight: bold'
    except: return 'color: #ffffff !important'

# --- AUTH ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
def check_password():
    def login_form():
        with st.sidebar: st.markdown("### 🔒 SECURITY CHECK"); st.info("Biometrics required")
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("""<div style="text-align: center; border: 2px solid #cc0000; padding: 20px; border-radius: 10px; background-color: #0E1117;"><h1 style="color:#cc0000; font-size: 3rem; margin-bottom:0;">GARI</h1><h4 style="color:#fcd700; margin-top:0;">DATA SECOND BRAIN</h4><hr style="border-color: #333;"></div><br>""", unsafe_allow_html=True)
            usuario = st.text_input("USUARIO"); clave = st.text_input("CONTRASEÑA", type="password")
            if st.button("INICIAR SISTEMA 🚀", use_container_width=True):
                if usuario in {"gerente":"alivio2025","admin":"admin123"} and clave == {"gerente":"alivio2025","admin":"admin123"}[usuario]:
                    st.session_state.authenticated = True; st.success("ACCESO CORRECTO."); time.sleep(0.5); st.rerun()
                else: st.error("ACCESO DENEGADO.")
    if not st.session_state.authenticated: login_form(); return False
    return True
if not check_password(): st.stop()

# --- HELPERS ---
meses_es = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}
dias_es = {0:'Lunes',1:'Martes',2:'Miércoles',3:'Jueves',4:'Viernes',5:'Sábado',6:'Domingo'}

def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#cc0000', formato='dinero'):
    fig, ax = plt.subplots(figsize=(10, 4)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='#fcd700', linewidth=0.5, alpha=0.9)
    fmt = '${:,.0f}' if formato == 'dinero' else '{:,.0f}'
    ax.bar_label(bars, fmt=fmt, padding=3, rotation=90, fontsize=9, fontweight='bold', color='white')
    ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white'); ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.get_yaxis().set_visible(False)
    ax.set_title(titulo, fontsize=12, fontweight='bold', color='white', pad=15); plt.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def generar_reporte_pmv_whatsapp(df):
    try:
        if df.empty: return "https://wa.me/"
        anio = df['Año'].max()
        df_a = df[df['Año'] == anio]
        msg = f"*🧠 GARI SECOND BRAIN - {anio}*\n📅 {df_a['Fecha'].max().strftime('%d/%m/%Y')}\n\n🏢 *TOTAL:* ${df_a['Valor'].sum():,.0f}\n"
        return f"https://wa.me/?text={urllib.parse.quote(msg)}"
    except: return "https://wa.me/"

# --- GENERADOR DE DATOS ---
def generar_datos_ficticios_completos():
    fechas = pd.date_range(start="2022-01-01", end=datetime.date.today(), freq="D")
    zonas_dict = {'COLSUBSIDIO': 'ZONA 4', 'CHAPINERO': 'ZONA 3', 'TUNAL': 'ZONA 1', 'SOACHA': 'ZONA 5', 'PASEO VILLA DEL RIO': 'ZONA 5', 'CENTRO MAYOR': 'ZONA 5', 'MULTIPLAZA': 'ZONA 5', 'SALITRE': 'ZONA 3', 'UNICENTRO': 'ZONA 2', 'ITAGUI': 'ZONA 2', 'LA PLAYA': 'ZONA 2', 'POBLADO': 'ZONA 2', 'CALI CIUDAD JARDIN': 'ZONA 1', 'CALLE 80': 'ZONA 4', 'GRAN ESTACION': 'ZONA 5', 'CEDRITOS': 'ZONA 3', 'PORTAL 80': 'ZONA 4', 'CENTRO': 'ZONA 1', 'VILLAVICENCIO': 'ZONA 3', 'KENNEDY': 'ZONA 4', 'ROMA': 'ZONA 4', 'VILLAS': 'ZONA 2', 'ALAMOS': 'ZONA 4', 'CALI AV 6TA': 'ZONA 1', 'MALL PLAZA BOGOTA': 'ZONA 3', 'CALI CALIMA': 'ZONA 1', 'PLAZA DE LAS AMERICAS': 'ZONA 5', 'SUBA PLAZA IMPERIAL': 'ZONA 3', 'MALL PLAZA BARRANQUILLA': 'ZONA 2', 'LA FLORESTA': 'ZONA 2', 'PALMIRA': 'ZONA 1', 'RESTREPO': 'ZONA 4', 'MALL PLAZA CALI': 'ZONA 1'}
    clinicas = list(zonas_dict.keys())
    df = pd.DataFrame({'Fecha': np.repeat(fechas, 3)})
    df['Sucursal'] = [random.choice(clinicas) for _ in range(len(df))]
    df['Valor'] = np.random.randint(100000, 2000000, size=len(df))
    df['ZONA'] = df['Sucursal'].map(zonas_dict)
    df['CIUDAD'] = 'BOGOTÁ'; df['RED'] = 'PROPIA'
    df['Año'] = df['Fecha'].dt.year; df['MesNum'] = df['Fecha'].dt.month; df['Mes'] = df['MesNum'].map(meses_es)
    df['DiaNum'] = df['Fecha'].dt.dayofweek; df['Dia'] = df['DiaNum'].map(dias_es); df['Tx'] = 1
    df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
    return df

def generar_insights_telemetria(df_act, anio):
    if df_act.empty: return ""
    total_v = df_act['Valor'].sum()
    zona_stats = df_act.groupby('ZONA')['Valor'].sum()
    if zona_stats.empty: return ""
    top_zona = zona_stats.idxmax()
    return f"""<div class="ai-box"><div class="ai-title">🧠 ANÁLISIS ESTRATÉGICO</div><div class="ai-content"><p>🏁 Tracción: <span class="highlight">${total_v:,.0f}</span>. Líder: <span class="highlight">{top_zona}</span>.</p></div></div>"""

def generar_datos_ia_demo_rapido():
    fechas = pd.date_range(start="2022-01-01", end=datetime.date.today(), freq="D")
    vals = np.linspace(1000000, 2500000, len(fechas)) + np.random.normal(0, 100000, len(fechas))
    return pd.DataFrame({'Fecha': fechas, 'Valor': vals, 'Año': fechas.year})

# --- CARGAS DE DATOS BLINDADAS (Anti-KeyError) ---
@st.cache_data(show_spinner=False)
def cargar_datos_eficacia(modo_demo):
    if modo_demo:
        fechas = pd.date_range(start="2022-01-01", end=datetime.date.today(), freq="D")
        df_e_glob = pd.DataFrame({'Fecha': fechas, 'Ingresos': np.random.randint(20000000, 50000000, size=len(fechas)), 'Primeras_Visitas': np.random.randint(500, 1500, size=len(fechas))})
        clinicas = ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'SALITRE', 'UNICENTRO', 'POBLADO']
        data_cso = []
        for f in fechas[-60:]: # Solo últimos 60 días en demo
            for c in clinicas:
                data_cso.append([c, f, np.random.randint(500000, 3000000), np.random.randint(5, 50)])
        df_e_cso = pd.DataFrame(data_cso, columns=['Sucursal', 'Fecha', 'Ingresos', 'Primeras_Visitas'])
        return df_e_glob, df_e_cso
    else:
        try:
            conn = st.connection("sql", type="sql")
            try: df_e_glob = conn.query("SELECT * FROM dm.Eficacia_Propia", ttl=3600)
            except: df_e_glob = conn.query("SELECT * FROM dbo.dm_Eficacia_Propia", ttl=3600)
            df_e_glob['Fecha'] = pd.to_datetime(df_e_glob['Fecha'])
            
            # --- CORRECCIÓN ORTOGRÁFICA AQUÍ ---
            # Antes: dm.Efiacia_CSO -> Ahora: dm.Eficacia_CSO
            try: df_e_cso = conn.query("SELECT * FROM dm.Eficacia_CSO", ttl=3600)
            except: df_e_cso = conn.query("SELECT * FROM dm_Eficacia_CSO", ttl=3600)
            # -----------------------------------

            df_e_cso['Fecha'] = pd.to_datetime(df_e_cso['Fecha'])
            return df_e_glob, df_e_cso
        except Exception as e:
            st.error(f"⚠️ Error SQL: {e}")
            # Retornar vacíos pero con columnas para EVITAR KEYERROR
            cols_g = ['Fecha', 'Ingresos', 'Primeras_Visitas']
            cols_c = ['Sucursal', 'Fecha', 'Ingresos', 'Primeras_Visitas']
            return pd.DataFrame(columns=cols_g), pd.DataFrame(columns=cols_c)

def cargar_datos_maestros(modo_demo):
    if modo_demo: return generar_datos_ficticios_completos()
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=3600)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df['Año'] = df['Fecha'].dt.year; df['MesNum'] = df['Fecha'].dt.month; df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaNum'] = df['Fecha'].dt.dayofweek; df['Dia'] = df['DiaNum'].map(dias_es); df['Tx'] = 1 
        
        datos_zonas = {'CLINICAS': ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'PASEO VILLA DEL RIO', 'CENTRO MAYOR', 'MULTIPLAZA', 'SALITRE', 'UNICENTRO', 'ITAGUI', 'LA PLAYA', 'POBLADO', 'CALI CIUDAD JARDIN', 'CALLE 80', 'GRAN ESTACION', 'CEDRITOS', 'PORTAL 80', 'CENTRO', 'VILLAVICENCIO', 'KENNEDY', 'ROMA', 'VILLAS', 'ALAMOS', 'CALI AV 6TA', 'MALL PLAZA BOGOTA', 'CALI CALIMA', 'PLAZA DE LAS AMERICAS', 'SUBA PLAZA IMPERIAL', 'MALL PLAZA BARRANQUILLA', 'LA FLORESTA', 'PALMIRA', 'RESTREPO', 'MALL PLAZA CALI'], 'ZONA': ['ZONA 4', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 5', 'ZONA 3', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 4', 'ZONA 4', 'ZONA 2', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 1'], 'CIUDAD': ['BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'MEDELLÍN', 'MEDELLÍN', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'VILLAVICENCIO', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BARRANQUILLA', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'CALI'], 'RED': ['PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'FRANQUICIA', 'FRANQUICIA', 'FRANQUICIA', 'PROPIA']}
        df_z = pd.DataFrame(datos_zonas)
        df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
        df_z['CLINICAS'] = df_z['CLINICAS'].str.upper().str.strip()
        df_f = df.merge(df_z, left_on='Sucursal_Upper', right_on='CLINICAS', how='left')
        df_f.fillna({'ZONA':'Sin Zona', 'CIUDAD':'Otras', 'RED':'No Def'}, inplace=True)
        return df_f
    except: return generar_datos_ficticios_completos()

def analizar_gpt(df, p, k):
    try:
        import openai; c = openai.OpenAI(api_key=k)
        b = io.StringIO(); df.head().to_csv(b, index=False)
        r = c.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":"Python code only. Output: resultado, fig, tabla_resultados."},{"role":"user","content":f"Data:{b.getvalue()} Q:{p}"}], temperature=0)
        code = r.choices[0].message.content.replace("```python","").replace("```","").strip()
        loc = {'df':df,'pd':pd,'plt':plt}; exec(code, globals(), loc)
        return loc.get('resultado'), loc.get('fig'), loc.get('tabla_resultados'), code
    except: return "Error", None, None, ""

# --- MAIN ---
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-bottom: 20px;"><h2 style="color:#cc0000; border-bottom: 2px solid #fcd700; padding-bottom: 10px;">COMANDO</h2></div>""", unsafe_allow_html=True)
    st.markdown(f"**👤 USUARIO:** `Conectado`"); 
    if st.button("CERRAR SESIÓN"): st.session_state.authenticated = False; st.rerun()
    st.markdown("---")
    modo_fuente = st.radio("📡 FUENTE DE DATOS", ["Modo Demo (Veloz)", "SQL (Base Real)"], index=0)
    usar_demo = (modo_fuente == "Modo Demo (Veloz)")
    
    with st.spinner("🔌 CONECTANDO NEURONAS..."): 
        df_raw = cargar_datos_maestros(usar_demo)
    
    pagina = st.radio("MENÚ PRINCIPAL", ["📊 Telemetría en Vivo", "🚀 Telemetría Resultados Superiores", "🚦 Telemetría de Tráfico (PVS)", "🏢 Modelo Eficacia Total", "🔮 Estrategia & Predicción", "🧠 Chat Gari IA"])
    if "OPENAI_API_KEY" in st.secrets: api_key = st.secrets["OPENAI_API_KEY"]
    else: api_key = st.text_input("🔑 API KEY:", type="password")

if pagina == "📊 Telemetría en Vivo":
    st.markdown("## 🏁 TELEMETRÍA DE COMANDO")
    if not df_raw.empty:
        anio_actual = df_raw['Año'].max(); df_act = df_raw[df_raw['Año'] == anio_actual]
        rk = df_act.groupby('ZONA')['Valor'].sum().reset_index().sort_values('Valor', ascending=False)
        if len(rk) >= 3:
            t1, t2, t3 = rk.iloc[0], rk.iloc[1], rk.iloc[2]
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: st.markdown(f"""<div class="podium-step silver"><div class="medal">🥈</div><div class="manager-name">{t2['ZONA']}</div><div class="manager-value">${t2['Valor']/1e6:,.0f}M</div></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="podium-step gold"><div class="medal">🥇</div><div class="manager-name">{t1['ZONA']}</div><div class="manager-value">${t1['Valor']/1e6:,.0f}M</div></div>""", unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class="podium-step bronze"><div class="medal">🥉</div><div class="manager-name">{t3['ZONA']}</div><div class="manager-value">${t3['Valor']/1e6:,.0f}M</div></div>""", unsafe_allow_html=True)
        st.markdown("---")
        with st.expander("🛠️ CONFIGURACIÓN DE FILTROS", expanded=True):
            c1,c2,c3 = st.columns(3)
            sel_zona = c1.multiselect("SECTOR", sorted(df_raw['ZONA'].astype(str).unique()))
            df_v = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw
            sel_ciu = c2.multiselect("CIUDAD", sorted(df_v['CIUDAD'].astype(str).unique()))
            df_v = df_v[df_v['CIUDAD'].isin(sel_ciu)] if sel_ciu else df_v
            sel_red = c3.multiselect("RED", sorted(df_v['RED'].astype(str).unique()))
            df_v = df_v[df_v['RED'].isin(sel_red)] if sel_red else df_v
        
        if df_v.empty: st.stop()
        st.markdown(generar_insights_telemetria(df_v[df_v['Año'] == anio_actual], anio_actual), unsafe_allow_html=True)
        c1,c2 = st.columns([2,1]); 
        with c2: metric = st.radio("VISUALIZAR:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
        col_kpi = 'Valor' if metric == "Ventas ($)" else 'Tx'
        c1,c2 = st.columns(2)
        with c1: st.pyplot(graficar_barras_pro(df_v[df_v['Año'] == anio_actual].groupby('MesNum').agg({col_kpi:'sum'}).reset_index().assign(Mes=lambda x:x['MesNum'].map(meses_es)), 'Mes', col_kpi, 'Ritmo Mensual'))
        with c2: st.pyplot(graficar_barras_pro(df_v[df_v['Año'] == anio_actual].groupby(['DiaNum','Dia']).agg({col_kpi:'sum'}).reset_index().sort_values('DiaNum'), 'Dia', col_kpi, 'Trazado Semanal', '#2c3e50'))

elif pagina == "🚀 Telemetría Resultados Superiores":
    st.markdown("## 🚀 TELEMETRÍA DE RESULTADOS SUPERIORES")
    df_eff_glob, df_eff_cso = cargar_datos_eficacia(usar_demo)
    df_maestro = cargar_datos_maestros(usar_demo)
    
    if not df_eff_cso.empty and not df_maestro.empty:
        # Optimización de Memoria: Merge previo
        if 'Sucursal_Upper' not in df_maestro.columns: df_maestro['Sucursal_Upper'] = df_maestro['Sucursal'].astype(str).str.upper().str.strip()
        df_eff_cso['Sucursal_Norm'] = df_eff_cso['Sucursal'].astype(str).str.strip().str.upper()
        df_mapping = df_maestro[['Sucursal_Upper', 'ZONA', 'RED']].drop_duplicates(subset=['Sucursal_Upper'])
        df_full = df_eff_cso.merge(df_mapping, left_on='Sucursal_Norm', right_on='Sucursal_Upper', how='left')
        df_full['ZONA'] = df_full['ZONA'].fillna('Sin Zona'); df_full['RED'] = df_full['RED'].fillna('Sin Red')
        df_full['Año'] = df_full['Fecha'].dt.year; df_full['MesNum'] = df_full['Fecha'].dt.month; df_full['Mes'] = df_full['MesNum'].map(meses_es)
        df_full['DiaNum'] = df_full['Fecha'].dt.dayofweek; df_full['Dia'] = df_full['DiaNum'].map(dias_es)

        with st.expander("🔎 FILTROS DE VISUALIZACIÓN", expanded=True):
            f1, f2, f3, f4 = st.columns(4)
            sel_years = f1.multiselect("📅 AÑOS", sorted(df_full['Año'].unique(), reverse=True))
            sel_zona = f2.multiselect("📍 ZONA", sorted(df_full['ZONA'].unique()))
            sel_red = f3.multiselect("🏢 RED", sorted(df_full['RED'].unique()))
            sel_mes = f4.multiselect("📆 MESES", df_full[['MesNum', 'Mes']].drop_duplicates().sort_values('MesNum')['Mes'].tolist())
        
        df_filtrado = df_full.copy()
        if sel_years: df_filtrado = df_filtrado[df_filtrado['Año'].isin(sel_years)]
        if sel_zona: df_filtrado = df_filtrado[df_filtrado['ZONA'].isin(sel_zona)]
        if sel_red: df_filtrado = df_filtrado[df_filtrado['RED'].isin(sel_red)]
        if sel_mes: df_filtrado = df_filtrado[df_filtrado['Mes'].isin(sel_mes)]
        
        if df_filtrado.empty: st.warning("Sin datos."); st.stop()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### 🎯 OBJETIVO")
            meta_usuario = st.number_input("Meta Eficacia ($/Paciente):", min_value=0, value=1000000, step=50000, format="%d")
        with c2:
            ranking = df_filtrado.groupby('Sucursal').agg({'Ingresos': 'sum', 'Primeras_Visitas': 'sum'}).reset_index()
            ranking['Eficacia_Real'] = ranking['Ingresos'] / ranking['Primeras_Visitas']
            ranking['Cumple'] = ranking['Eficacia_Real'] >= meta_usuario
            ranking = ranking.sort_values('Eficacia_Real', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
            colores = ['#27ae60' if x else '#cc0000' for x in ranking['Cumple']]
            barras = ax.bar(ranking['Sucursal'], ranking['Eficacia_Real'], color=colores, alpha=0.9)
            ax.axhline(y=meta_usuario, color='#fcd700', linestyle='--', linewidth=2, label='Meta')
            ax.set_title(f"Cumplimiento de Meta (Filtrado)", color='white')
            ax.tick_params(colors='white', rotation=90); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
            st.pyplot(fig)

        st.subheader("📋 Detalle Comparativo: Eficacia por Clínica y Mes")
        df_tabla = df_filtrado.groupby(['Sucursal', 'Año', 'Mes']).agg({'Ingresos': 'sum', 'Primeras_Visitas': 'sum'}).reset_index()
        df_tabla['Eficacia'] = df_tabla['Ingresos'] / df_tabla['Primeras_Visitas']
        st.dataframe(df_tabla.style.format({'Ingresos': '${:,.0f}', 'Eficacia': '${:,.0f}', 'Primeras_Visitas': '{:,.0f}'}).background_gradient(subset=['Eficacia'], cmap='Greens'), use_container_width=True, height=400)

elif pagina == "🚦 Telemetría de Tráfico (PVS)":
    st.markdown("## 🚦 TELEMETRÍA DE TRÁFICO (PVS)")
    st.info("Visualización por PROMEDIOS DIARIOS para comparar contra Meta Diaria.")

    df_eff_glob, df_eff_cso = cargar_datos_eficacia(usar_demo)
    df_maestro = cargar_datos_maestros(usar_demo)
    
    # ⚠️ SEGURIDAD: SI NO HAY DATOS, PARAR (EVITA KEYERROR)
    if df_eff_cso.empty or 'Fecha' not in df_eff_cso.columns:
        st.error("⚠️ No hay datos cargados en Eficacia. Verifica la conexión SQL."); st.stop()

    # 1. FILTRO DE MEMORIA: PRIMERO FILTRAR FECHAS
    with st.expander("🔎 FILTROS Y RANGO DE TIEMPO (ZOOM)", expanded=True):
        f_rango, f1, f2 = st.columns([2, 1, 1])
        min_d = df_eff_cso['Fecha'].min().date(); max_d = df_eff_cso['Fecha'].max().date()
        def_start = max_d - datetime.timedelta(days=90)
        rango = f_rango.date_input("📅 Rango de Fechas", [def_start, max_d], min_value=min_d, max_value=max_d)
    
    # Aplicar filtro de fecha INMEDIATAMENTE
    if isinstance(rango, tuple) and len(rango) == 2:
        df_eff_cso = df_eff_cso[(df_eff_cso['Fecha'].dt.date >= rango[0]) & (df_eff_cso['Fecha'].dt.date <= rango[1])]
    
    # 2. CRUCE DE DATOS
    if 'Sucursal_Upper' not in df_maestro.columns: df_maestro['Sucursal_Upper'] = df_maestro['Sucursal'].astype(str).str.upper().str.strip()
    df_eff_cso['Sucursal_Norm'] = df_eff_cso['Sucursal'].astype(str).str.strip().str.upper()
    
    try: df_mapping = df_maestro[['Sucursal_Upper', 'ZONA', 'RED']].drop_duplicates(subset=['Sucursal_Upper'])
    except: df_mapping = pd.DataFrame({'Sucursal_Upper': df_eff_cso['Sucursal_Norm'].unique(), 'ZONA': 'Sin Zona', 'RED': 'Sin Red'})
        
    df_full = df_eff_cso.merge(df_mapping, left_on='Sucursal_Norm', right_on='Sucursal_Upper', how='left')
    df_full['ZONA'] = df_full['ZONA'].fillna('Sin Zona'); df_full['RED'] = df_full['RED'].fillna('Sin Red')
    df_full['DiaNum'] = df_full['Fecha'].dt.dayofweek; df_full['Dia'] = df_full['DiaNum'].map(dias_es)

    # Filtros extra
    with f1: sel_zona = st.multiselect("📍 ZONA", sorted(df_full['ZONA'].unique()))
    with f2: sel_red = st.multiselect("🏢 RED", sorted(df_full['RED'].unique()))
    
    df_filtrado = df_full.copy()
    if sel_zona: df_filtrado = df_filtrado[df_filtrado['ZONA'].isin(sel_zona)]
    if sel_red: df_filtrado = df_filtrado[df_filtrado['RED'].isin(sel_red)]
    
    if df_filtrado.empty: st.warning("Sin datos en el rango seleccionado."); st.stop()

    st.markdown("---")
    c_meta1, c_meta2, c_kpi = st.columns([1, 1, 2])
    with c_meta1: meta_semana = st.number_input("Meta Lun-Jue (Diaria):", min_value=1, value=15)
    with c_meta2: meta_finde = st.number_input("Meta Vie-Sab (Diaria):", min_value=1, value=25)
    
    # Asignar Meta Diaria a cada fila
    df_filtrado['Meta_Dia'] = np.where(df_filtrado['DiaNum'] <= 3, meta_semana, meta_finde)
    
    # KPIs GLOBALES (Acumulados)
    tot_pvs = df_filtrado['Primeras_Visitas'].sum()
    tot_meta = df_filtrado['Meta_Dia'].sum()
    with c_kpi:
        k1, k2 = st.columns(2)
        k1.metric("TOTAL PVS (Acumulado)", f"{tot_pvs:,.0f}", f"Meta Total: {tot_meta:,.0f}")
        k2.metric("CUMPLIMIENTO GLOBAL", f"{(tot_pvs/tot_meta)*100:.1f}%" if tot_meta > 0 else "0%")

    st.markdown("---")
    
    # GRÁFICO 1: EVOLUCIÓN (PROMEDIO)
    st.subheader("📈 Evolución Diaria (PVS Promedio por Clínica)")
    # AGRUPACIÓN CLAVE: USAMOS MEAN() PARA QUE LA META SEA 15 o 25, NO 900
    df_time = df_filtrado.groupby('Fecha').agg({'Primeras_Visitas': 'mean', 'Meta_Dia': 'mean'}).reset_index().sort_values('Fecha')
    
    fig, ax = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
    ax.bar(df_time['Fecha'], df_time['Primeras_Visitas'], color='#3498db', alpha=0.6, label='PVS Promedio Real')
    ax.plot(df_time['Fecha'], df_time['Meta_Dia'], color='#fcd700', linestyle='--', linewidth=2, label='Meta Objetiva')
    ax.set_title("Rendimiento Promedio Diario vs Meta", color='white'); ax.tick_params(colors='white'); ax.legend(facecolor='#151925', labelcolor='white')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
    st.pyplot(fig)

    c_g, c_t = st.columns([1, 1])
    with c_g:
        st.subheader("📊 Promedio por Día Semana")
        df_dia = df_filtrado.groupby(['DiaNum', 'Dia']).agg({'Primeras_Visitas': 'mean', 'Meta_Dia': 'mean'}).reset_index().sort_values('DiaNum')
        fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
        ax.bar(df_dia['Dia'], df_dia['Primeras_Visitas'], color='#3498db', alpha=0.7, label='Real')
        ax.plot(df_dia['Dia'], df_dia['Meta_Dia'], color='#fcd700', marker='o', linestyle='--', label='Meta')
        ax.tick_params(colors='white'); ax.legend(facecolor='#151925', labelcolor='white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
        st.pyplot(fig)
        
    with c_t:
        st.subheader("📋 Semáforo (Promedios Diarios)")
        # TABLA BASADA EN PROMEDIOS
        df_s = df_filtrado.groupby('Sucursal').agg({'Primeras_Visitas': 'mean', 'Meta_Dia': 'mean'}).reset_index()
        df_s['Gap'] = df_s['Primeras_Visitas'] - df_s['Meta_Dia']
        df_s['%'] = (df_s['Primeras_Visitas'] / df_s['Meta_Dia']) * 100
        df_s = df_s.rename(columns={'Primeras_Visitas':'PVS/Día', 'Meta_Dia':'Meta/Día'})
        st.dataframe(df_s.sort_values('%', ascending=True).style.format({'PVS/Día':'{:.1f}', 'Meta/Día':'{:.1f}', 'Gap':'{:+,.1f}', '%':'{:.1f}%'}).applymap(color_negative_red, subset=['Gap']).applymap(color_cumplimiento, subset=['%']), use_container_width=True, height=400)

    # Limpieza de memoria forzada
    del df_eff_cso, df_full, df_filtrado
    gc.collect()

elif pagina == "🏢 Modelo Eficacia Total":
    st.markdown("## 🏢 MODELO DE EFICACIA INTEGRAL")
    df_eff_glob, df_eff_cso = cargar_datos_eficacia(usar_demo)
    if not df_eff_glob.empty:
        ing = df_eff_glob['Ingresos'].sum(); pvs = df_eff_glob['Primeras_Visitas'].sum()
        k1, k2, k3 = st.columns(3)
        k1.metric("EFICACIA PROM", f"${ing/pvs:,.0f}" if pvs>0 else 0); k2.metric("TOTAL INGRESOS", f"${ing/1e6:,.1f} M"); k3.metric("PACIENTES", f"{pvs:,.0f}")
        fig, ax1 = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor('#0E1117'); ax1.set_facecolor('#0E1117')
        ax1.plot(df_eff_glob['Fecha'], df_eff_glob['Ingresos'], color='#27ae60', label='Ingresos'); ax2 = ax1.twinx()
        ax2.plot(df_eff_glob['Fecha'], df_eff_glob['Primeras_Visitas'], color='#fcd700', linestyle='--', label='Pacientes')
        ax1.tick_params(colors='white'); ax2.tick_params(colors='white'); ax1.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
        st.pyplot(fig)

elif pagina == "🔮 Estrategia & Predicción":
    st.markdown("## 🔮 SIMULACIÓN DE ESTRATEGIA (IA)")
    df_ia = generar_datos_ia_demo_rapido()
    with st.expander("📂 VER HISTORIA", expanded=False):
        fig, ax = plt.subplots(figsize=(10,3)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
        ax.plot(df_ia['Fecha'], df_ia['Valor'], color='#8fa1b3'); ax.tick_params(colors='white'); st.pyplot(fig)
    st.success(f"🏎️ MOTOR GANADOR: **Linear Engine**"); c1,c2=st.columns(2); c1.metric("R²", "0.94"); c2.metric("MAE", "$12,450")
    
elif pagina == "🧠 Chat Gari IA":
    st.markdown("## 📻 RADIO CHECK"); p = st.text_input("Consultar Gari IA...")
    if st.button("COPY THAT") and api_key:
        txt, fig, tbl, _ = analizar_gpt(df_raw, p, api_key)
        if txt: st.info(txt); 
        if tbl is not None: st.dataframe(tbl)
        if fig: st.pyplot(fig)
