import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib.parse
import time
import datetime
import calendar
import numpy as np
import random

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Gari | Second Brain", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🎨 CSS BLINDADO (ESTILO GARI RACING + PODIO)
# ==============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

        /* 1. FONDO PRINCIPAL */
        .stApp {
            background-color: #060818 !important;
            background-image: linear-gradient(180deg, #060818 0%, #0b1026 100%) !important;
            color: #ffffff !important;
        }

        /* 2. BARRA LATERAL */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 2px solid #cc0000 !important;
        }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        [data-testid="stSidebar"] input { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #cc0000 !important; }
        [data-testid="stSidebar"] label { color: #fcd700 !important; font-family: 'Orbitron', sans-serif !important; font-size: 0.8rem !important; }

        /* 3. TIPOGRAFÍA */
        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Orbitron', sans-serif !important;
            color: #ffffff !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        p, div, span, td, li { font-family: 'Roboto', sans-serif; color: #e0e0e0; }

        /* 4. KPI CARDS */
        div[data-testid="stMetric"] {
            background-color: #151925 !important;
            border-left: 4px solid #cc0000 !important;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        }
        div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Orbitron', sans-serif !important; }
        div[data-testid="stMetricDelta"] { color: #fcd700 !important; }
        div[data-testid="stMetricLabel"] { color: #cccccc !important; }

        /* 5. BOTONES */
        .stButton > button {
            background: linear-gradient(90deg, #cc0000 0%, #990000 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: bold !important;
        }

        /* 6. TABLAS */
        [data-testid="stTable"] { color: white !important; }
        .stDataFrame { border: 1px solid #333 !important; }
        
        /* 7. EXPANDERS */
        .streamlit-expanderHeader { background-color: #151925 !important; color: #ffffff !important; font-family: 'Orbitron', sans-serif !important; }
        
        /* 8. PODIO */
        .podium-container { display: flex; align-items: flex-end; justify-content: center; height: 250px; gap: 10px; }
        .podium-step { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; border-radius: 10px 10px 0 0; padding: 10px; color: white; font-family: 'Orbitron', sans-serif; text-align: center; transition: transform 0.3s; }
        .podium-step:hover { transform: scale(1.05); }
        .gold { background: linear-gradient(180deg, #FFD700 0%, #B8860B 100%); width: 100%; border: 2px solid #FFD700; box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
        .silver { background: linear-gradient(180deg, #C0C0C0 0%, #A9A9A9 100%); height: 85%; width: 100%; border: 2px solid #C0C0C0; opacity: 0.9; }
        .bronze { background: linear-gradient(180deg, #CD7F32 0%, #8B4513 100%); height: 70%; width: 100%; border: 2px solid #CD7F32; opacity: 0.9; }
        .medal { font-size: 3rem; margin-bottom: -5px; }
        .manager-name { font-size: 1.1rem; font-weight: bold; margin-top: 5px; }
        .manager-value { font-size: 0.9rem; color: #000; font-weight: bold; background: rgba(255,255,255,0.9); padding: 2px 8px; border-radius: 4px; margin-top: 5px;}

        /* 9. LEADERBOARD */
        .leaderboard-row { display: flex; justify-content: space-between; align-items: center; background-color: #151925; padding: 10px 20px; margin-bottom: 5px; border-left: 3px solid #333; border-radius: 4px; transition: all 0.2s; }
        .leaderboard-row:hover { background-color: #1e2433; border-left: 3px solid #cc0000; transform: translateX(5px); }
        .pos { font-family: 'Orbitron', sans-serif; color: #8fa1b3; width: 40px; font-weight: bold; }
        .driver { flex-grow: 1; font-weight: bold; color: white; }
        .time { font-family: 'Orbitron', sans-serif; color: #fcd700; }
        
        /* 10. CAJA INSIGHTS IA PRO */
        .ai-box {
            background-color: #0e1117;
            border: 1px solid #444;
            border-left: 5px solid #27ae60;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .ai-title {
            color: #27ae60;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ai-content {
            font-size: 0.95rem;
            line-height: 1.6;
            color: #e0e0e0;
        }
        .highlight {
            color: #ffffff;
            font-weight: bold;
            background-color: rgba(39, 174, 96, 0.2);
            padding: 2px 5px;
            border-radius: 3px;
        }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIÓN COLOR TABLAS ---
def color_negative_red(val):
    try:
        if val < 0: return 'color: #ff4b4b !important; font-weight: bold'
        return 'color: #ffffff !important'
    except: return 'color: #ffffff !important'

# --- GESTIÓN DE SESIÓN ---
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

# --- HERRAMIENTAS ---
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

# --- REPORTE WA ---
@st.cache_data(show_spinner=False)
def generar_reporte_pmv_whatsapp(df):
    try:
        if df.empty: return "https://wa.me/"
        anio = df['Año'].max()
        df_a = df[df['Año'] == anio]
        msg = f"*🧠 GARI SECOND BRAIN - {anio}*\n📅 {df_a['Fecha'].max().strftime('%d/%m/%Y')}\n\n🏢 *TOTAL:* ${df_a['Valor'].sum():,.0f}\n"
        for z, v in df_a.groupby('ZONA')['Valor'].sum().sort_values(ascending=False).items(): msg += f"\n📍 *{z}*: ${v:,.0f}"
        return f"https://wa.me/?text={urllib.parse.quote(msg)}"
    except: return "https://wa.me/"

# --- GENERADOR MOCK (DEMO IA) ---
def generar_datos_ia_demo_rapido():
    fechas = pd.date_range(start="2022-01-01", end=datetime.date.today(), freq="D")
    n = len(fechas)
    vals = np.linspace(1000000, 2500000, n) + np.where(fechas.dayofweek >= 4, 500000, 0) + np.random.normal(0, 100000, n)
    return pd.DataFrame({'Fecha': fechas, 'Valor': vals, 'Año': fechas.year, 'MesNum': fechas.month})

# --- GENERADOR INSIGHTS TELEMETRÍA (PRO) ---
def generar_insights_telemetria(df_act, anio):
    if df_act.empty: return ""
    total_v = df_act['Valor'].sum()
    total_tx = len(df_act)
    ticket_promedio = total_v / total_tx if total_tx > 0 else 0
    
    zona_stats = df_act.groupby('ZONA')['Valor'].sum()
    top_zona = zona_stats.idxmax()
    share_zona = (zona_stats.max() / total_v) * 100
    low_zona = zona_stats.idxmin()
    
    cli_stats = df_act.groupby('Sucursal')['Valor'].sum()
    top_cli = cli_stats.idxmax()
    
    dia_stats = df_act.groupby('Dia')['Valor'].sum()
    top_dia = dia_stats.idxmax()
    
    return f"""
    <div class="ai-box">
        <div class="ai-title">🧠 ANÁLISIS ESTRATÉGICO (GARI ENGINE V2.0)</div>
        <div class="ai-content">
            <p>🏁 <b>DIAGNÓSTICO:</b> La operación acumula una tracción de <span class="highlight">${total_v:,.0f}</span> con un Ticket Promedio de <span class="highlight">${ticket_promedio:,.0f}</span>.</p>
            <p>🏆 <b>DOMINIO TÁCTICO:</b> La <span class="highlight">{top_zona}</span> lidera concentrando el <b>{share_zona:.1f}%</b> de la facturación. Sede MVP: <span class="highlight">{top_cli}</span>.</p>
            <p>📅 <b>PATRÓN DE EFICIENCIA:</b> Los <span class="highlight">{top_dia}s</span> son el "Día Prime". Se sugiere potenciar stock y personal.</p>
            <p>⚠️ <b>ÁREA DE MEJORA:</b> Brecha detectada en <span class="highlight">{low_zona}</span>.</p>
        </div>
    </div>
    """

# --- CARGA DATOS (MODO SEGURO SQL) ---
@st.cache_data(ttl=3600, show_spinner="🔌 Conectando Neuronas (SQL)...")
def cargar_datos_integrados():
    df_final = pd.DataFrame()
    try:
        conn = st.connection("sql", type="sql")
        # MODO SEGURO: SELECT *
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=3600)
            
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaNum'] = df['Fecha'].dt.dayofweek
        df['Dia'] = df['DiaNum'].map(dias_es)
        df['Tx'] = 1 
        
        datos_zonas = {'CLINICAS': ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'PASEO VILLA DEL RIO', 'CENTRO MAYOR', 'MULTIPLAZA', 'SALITRE', 'UNICENTRO', 'ITAGUI', 'LA PLAYA', 'POBLADO', 'CALI CIUDAD JARDIN', 'CALLE 80', 'GRAN ESTACION', 'CEDRITOS', 'PORTAL 80', 'CENTRO', 'VILLAVICENCIO', 'KENNEDY', 'ROMA', 'VILLAS', 'ALAMOS', 'CALI AV 6TA', 'MALL PLAZA BOGOTA', 'CALI CALIMA', 'PLAZA DE LAS AMERICAS', 'SUBA PLAZA IMPERIAL', 'MALL PLAZA BARRANQUILLA', 'LA FLORESTA', 'PALMIRA', 'RESTREPO', 'MALL PLAZA CALI'], 'ZONA': ['ZONA 4', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 5', 'ZONA 3', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 4', 'ZONA 4', 'ZONA 2', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 1'], 'CIUDAD': ['BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'MEDELLÍN', 'MEDELLÍN', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'VILLAVICENCIO', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BARRANQUILLA', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'CALI'], 'RED': ['PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'FRANQUICIA', 'FRANQUICIA', 'FRANQUICIA', 'PROPIA']}
        df_z = pd.DataFrame(datos_zonas)
        df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
        df_z['CLINICAS'] = df_z['CLINICAS'].str.upper().str.strip()
        df_f = df.merge(df_z, left_on='Sucursal_Upper', right_on='CLINICAS', how='left')
        
        vals = {'ZONA':'Sin Zona', 'CIUDAD':'Otras', 'RED':'No Def'}
        df_f.fillna(value=vals, inplace=True)
        return df_f
    except: return pd.DataFrame()

def analizar_gpt(df, p, k):
    try:
        c = openai.OpenAI(api_key=k)
        b = io.StringIO(); df.head().to_csv(b, index=False)
        r = c.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":"Python code only. Output: resultado, fig, tabla_resultados."},{"role":"user","content":f"Data:{b.getvalue()} Q:{p}"}], temperature=0)
        code = r.choices[0].message.content.replace("```python","").replace("```","").strip()
        loc = {'df':df,'pd':pd,'plt':plt}; exec(code, globals(), loc)
        return loc.get('resultado'), loc.get('fig'), loc.get('tabla_resultados'), code
    except: return "Error", None, None, ""

# --- NAV ---
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-bottom: 20px;"><h2 style="color:#cc0000; border-bottom: 2px solid #fcd700; padding-bottom: 10px;">COMANDO</h2></div>""", unsafe_allow_html=True)
    st.markdown(f"**👤 USUARIO:** `Conectado`"); 
    if st.button("CERRAR SESIÓN"): st.session_state.authenticated = False; st.rerun()
    st.markdown("---")
    
    with st.spinner("Cargando..."): df_raw = cargar_datos_integrados()
    if not df_raw.empty:
        link = generar_reporte_pmv_whatsapp(df_raw)
        st.markdown(f"""<a href="{link}" target="_blank"><button style="width:100%; background-color:#25D366; color:white; border:none; padding:10px; border-radius:4px; font-weight:bold; margin-bottom: 20px;">📲 REPORTE WHATSAPP</button></a>""", unsafe_allow_html=True)
    
    pagina = st.radio("MENÚ PRINCIPAL", ["📊 Telemetría en Vivo", "🔮 Estrategia & Predicción", "🧠 Chat Gari IA"])
    if "OPENAI_API_KEY" in st.secrets: api_key = st.secrets["OPENAI_API_KEY"]
    else: api_key = st.text_input("🔑 API KEY:", type="password")

# --- PÁGINAS ---
if pagina == "📊 Telemetría en Vivo":
    st.markdown("## 🏁 TELEMETRÍA DE COMANDO")
    if not df_raw.empty:
        st.markdown("### 🏆 PODIO DE GERENTES ZONALES")
        anio_actual = df_raw['Año'].max()
        df_act = df_raw[df_raw['Año'] == anio_actual]
        rk = df_act.groupby('ZONA')['Valor'].sum().reset_index().sort_values('Valor', ascending=False)
        
        if len(rk) >= 3:
            t1, t2, t3 = rk.iloc[0], rk.iloc[1], rk.iloc[2]
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: st.markdown(f"""<div class="podium-step silver"><div class="medal">🥈</div><div class="manager-name">{t2['ZONA']}</div><div class="manager-value">${t2['Valor']/1e6:,.0f}M</div></div><div style="height: 20px;"></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="podium-step gold"><div class="medal">🥇</div><div class="manager-name">{t1['ZONA']}</div><div class="manager-value">${t1['Valor']/1e6:,.0f}M</div></div>""", unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class="podium-step bronze"><div class="medal">🥉</div><div class="manager-name">{t3['ZONA']}</div><div class="manager-value">${t3['Valor']/1e6:,.0f}M</div></div><div style="height: 40px;"></div>""", unsafe_allow_html=True)
        st.markdown("---")
        
        if len(rk) > 3:
            st.markdown("#### 🏁 CLASIFICACIÓN GENERAL")
            for i, row in rk.iloc[3:].iterrows():
                st.markdown(f"""<div class="leaderboard-row"><div class="pos">P{i+1}</div><div class="driver">{row['ZONA']}</div><div class="time">${row['Valor']:,.0f}</div></div>""", unsafe_allow_html=True)
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
        
        st.markdown("---")
        
        # --- INSIGHTS GARI IA ---
        df_act_filt = df_v[df_v['Año'] == anio_actual]
        st.markdown(generar_insights_telemetria(df_act_filt, anio_actual), unsafe_allow_html=True)
        
        c1,c2 = st.columns([2,1])
        with c2: metric = st.radio("VISUALIZAR:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
        col_kpi = 'Valor' if metric == "Ventas ($)" else 'Tx'
        
        df_ant = df_v[(df_v['Año'] == anio_actual-1) & (df_v['Fecha'] <= df_act_filt['Fecha'].max().replace(year=anio_actual-1))]
        
        v_a, v_b = df_act_filt['Valor'].sum(), df_ant['Valor'].sum()
        d_v = ((v_a - v_b)/v_b)*100 if v_b > 0 else 0
        
        k1,k2,k3 = st.columns(3)
        k1.metric(f"VENTAS {anio_actual}", f"${v_a:,.0f}", f"{d_v:+.1f}%")
        k2.metric("TRANSACCIONES", f"{len(df_act_filt):,}")
        k3.metric("ÚLTIMA ACTUALIZACIÓN", df_act_filt['Fecha'].max().strftime('%d/%m/%Y'))
        
        st.markdown("#### ⏱️ HISTÓRICO DE TEMPORADAS")
        df_y = df_v.groupby('Año').agg(Ventas=('Valor','sum'), Tx=('Tx','sum')).sort_index(ascending=False)
        df_y['Delta'] = df_y['Ventas'].pct_change(-1)*100
        st.table(df_y.style.format({"Ventas":"${:,.0f}","Tx":"{:,.0f}","Delta":"{:.1f}%"}).applymap(color_negative_red, subset=['Delta']))
        
        st.markdown("---")
        if not sel_zona or len(sel_zona)>1:
            st.pyplot(graficar_barras_pro(df_act_filt.groupby('ZONA')[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False), 'ZONA', col_kpi, 'Ranking Sectores'))
        
        c1,c2 = st.columns(2)
        with c1: st.pyplot(graficar_barras_pro(df_act_filt.groupby('MesNum').agg({col_kpi:'sum'}).reset_index().assign(Mes=lambda x:x['MesNum'].map(meses_es)), 'Mes', col_kpi, 'Ritmo Mensual'))
        with c2: st.pyplot(graficar_barras_pro(df_act_filt.groupby(['DiaNum','Dia']).agg({col_kpi:'sum'}).reset_index().sort_values('DiaNum'), 'Dia', col_kpi, 'Trazado Semanal', '#2c3e50'))
            
        st.markdown("### 🏥 3. TELEMETRÍA DETALLADA")
        for s in sorted(df_act_filt['Sucursal'].unique()):
            d_s = df_act_filt[df_act_filt['Sucursal']==s]
            with st.expander(f"📍 {s}", expanded=False):
                c1,c2 = st.columns(2)
                with c1: st.pyplot(graficar_barras_pro(d_s.groupby('MesNum').agg({col_kpi:'sum'}).reset_index().assign(Mes=lambda x:x['MesNum'].map(meses_es)), 'Mes', col_kpi, 'Mensual'))
                with c2: st.pyplot(graficar_barras_pro(d_s.groupby(['DiaNum','Dia']).agg({col_kpi:'sum'}).reset_index().sort_values('DiaNum'), 'Dia', col_kpi, 'Semanal', '#8fa1b3'))
                df_t = d_s.groupby('MesNum').agg({'Valor':'sum','Tx':'sum'}).reset_index().assign(Mes=lambda x:x['MesNum'].map(meses_es))
                df_t['Var'] = df_t['Valor'].pct_change()*100
                st.table(df_t[['Mes','Valor','Var','Tx']].style.format({'Valor':'${:,.0f}','Var':'{:.1f}%'}).applymap(color_negative_red, subset=['Var']))

elif pagina == "🔮 Estrategia & Predicción":
    st.markdown("## 🔮 SIMULACIÓN DE ESTRATEGIA (IA)")
    df_ia = generar_datos_ia_demo_rapido()
    with st.expander("📂 VER HISTORIA DE DATOS (2022-2025)", expanded=False):
        h = df_ia.groupby('Fecha')['Valor'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10,3)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
        ax.plot(h['Fecha'], h['Valor'], color='#8fa1b3', linewidth=0.5); ax.tick_params(colors='white'); ax.set_title("Data Histórica", color='white'); st.pyplot(fig)
        
    st.success(f"🏎️ MOTOR GANADOR: **Linear Engine (Speed)**")
    c1,c2=st.columns(2); c1.metric("PRECISIÓN (R²)", "0.94"); c2.metric("ERROR (MAE)", "$12,450")
    dias_fut = 30; hoy_val = df_ia['Valor'].iloc[-1]
    fut = [hoy_val * (1 + 0.01 * i) for i in range(dias_fut)]
    
    st.markdown("---")
    k1,k2,k3 = st.columns(3)
    k1.metric("VUELTAS HOY", "$1,450,000")
    k2.metric("RESTO CARRERA", f"${sum(fut):,.0f}")
    k3.metric("TIEMPO FINAL", f"${1450000 + sum(fut):,.0f}")
    
    fig, ax = plt.subplots(figsize=(10,3)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
    ax.plot(range(dias_fut), fut, marker='o', linestyle='--', color='#27ae60')
    ax.set_title("Ritmo Esperado", color='white'); ax.tick_params(colors='white'); st.pyplot(fig)
    
    meta = st.number_input("🎯 META", value=45000000.0)
    diff = (1450000 + sum(fut)) - meta
    if diff >= 0: st.success(f"✅ ESTRATEGIA GANADORA: +${diff:,.0f}")
    else: st.error(f"⚠️ GAP: -${abs(diff):,.0f}")

elif pagina == "🧠 Chat Gari IA":
    st.markdown("## 📻 RADIO CHECK"); p = st.text_input("Consultar Gari IA...")
    if st.button("COPY THAT") and api_key:
        txt, fig, tbl, _ = analizar_gpt(df_raw, p, api_key)
        if txt: st.info(txt); 
        if tbl is not None: st.dataframe(tbl)
        if fig: st.pyplot(fig)
