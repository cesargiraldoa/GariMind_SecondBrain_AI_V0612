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

# --- LIBRERÍAS ML ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Gari | RB Racing", 
    page_icon="🏎️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🎨 CSS BLINDADO
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

        /* 2. BARRA LATERAL (SIDEBAR) */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 2px solid #cc0000 !important;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] input {
            background-color: #111111 !important;
            color: #ffffff !important;
            border: 1px solid #cc0000 !important;
        }
        [data-testid="stSidebar"] label {
            color: #fcd700 !important;
            font-family: 'Orbitron', sans-serif !important;
            font-size: 0.8rem !important;
        }

        /* 3. TIPOGRAFÍA */
        h1, h2, h3, h4 {
            font-family: 'Orbitron', sans-serif !important;
            color: #ffffff !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        p, div, span, td {
            font-family: 'Roboto', sans-serif;
        }

        /* 4. KPI CARDS */
        div[data-testid="stMetric"] {
            background-color: #151925 !important;
            border-left: 4px solid #cc0000 !important;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #fcd700 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #cccccc !important;
        }

        /* 5. BOTONES */
        .stButton > button {
            background: linear-gradient(90deg, #cc0000 0%, #990000 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: bold !important;
            text-transform: uppercase !important;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(204, 0, 0, 0.8);
        }

        /* 6. TABLAS */
        [data-testid="stTable"] {
            color: white !important;
        }
        .stDataFrame {
            border: 1px solid #333 !important;
        }
        
        /* 7. EXPANDERS */
        .streamlit-expanderHeader {
            background-color: #151925 !important;
            color: #ffffff !important;
            font-family: 'Orbitron', sans-serif !important;
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
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    def login_form():
        with st.sidebar:
            st.markdown("### 🔒 SECURITY CHECK")
            st.info("Biometrics required")
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("""
            <div style="text-align: center; border: 2px solid #cc0000; padding: 20px; border-radius: 10px; background-color: #0E1117;">
                <h1 style="color:#cc0000; font-size: 3rem; margin-bottom:0;">GARI</h1>
                <h4 style="color:#fcd700; margin-top:0;">RED BULL RACING DATA</h4>
                <hr style="border-color: #333;">
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            usuario = st.text_input("PILOTO (Usuario)")
            clave = st.text_input("CÓDIGO (Password)", type="password")
            
            if st.button("START ENGINE 🏁", use_container_width=True):
                usuarios_validos = {
                    "gerente": "alivio2025", 
                    "admin": "admin123",
                    "gari": "hamster"
                }
                if usuario in usuarios_validos and usuarios_validos[usuario] == clave:
                    st.session_state.authenticated = True
                    st.success("ACCESS GRANTED.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("ACCESS DENIED.")

    if not st.session_state.authenticated:
        login_form()
        return False
    return True

if not check_password():
    st.stop()

# ==============================================================================
# 🚀 APLICACIÓN PRINCIPAL
# ==============================================================================

# --- HERRAMIENTAS ---
meses_es = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
dias_es = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# --- GRÁFICOS ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#cc0000', formato='dinero'):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0E1117') 
    ax.set_facecolor('#0E1117')
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='#fcd700', linewidth=0.5, alpha=0.9)
    fmt = '${:,.0f}' if formato == 'dinero' else '{:,.0f}'
    ax.bar_label(bars, fmt=fmt, padding=3, rotation=90, fontsize=9, fontweight='bold', color='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white', rotation=0, labelsize=9)
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(titulo, fontsize=12, fontweight='bold', color='white', pad=15)
    plt.tight_layout()
    return fig

# --- IA HÍBRIDA ---
@st.cache_resource
def entrenar_modelo_predictivo(df):
    try:
        df_ml = df.groupby('Fecha')['Valor'].sum().reset_index().sort_values('Fecha')
        df_ml['DiaNum'] = df_ml['Fecha'].dt.dayofweek
        df_ml['DiaMes'] = df_ml['Fecha'].dt.day
        df_ml['FechaOrdinal'] = df_ml['Fecha'].apply(lambda x: x.toordinal())
        df_ml['Lag_1'] = df_ml['Valor'].shift(1).fillna(0)
        df_ml = df_ml.iloc[1:]
        if len(df_ml) < 10: return None, None, "Insuficiente Data"
        X = df_ml[['DiaNum', 'DiaMes', 'Lag_1', 'FechaOrdinal']]
        y = df_ml['Valor']
        split = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        m_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
        r2_rf = r2_score(y_test, m_rf.predict(X_test))
        m_lr = LinearRegression().fit(X_train, y_train)
        r2_lr = r2_score(y_test, m_lr.predict(X_test))
        
        if r2_rf > 0 and r2_rf > r2_lr: return m_rf, {"R2": r2_rf, "MAE": mean_absolute_error(y_test, m_rf.predict(X_test))}, "Random Forest (Aero)"
        else: return m_lr, {"R2": r2_lr, "MAE": mean_absolute_error(y_test, m_lr.predict(X_test))}, "Linear Engine (Speed)"
    except: return None, None, "Error"

def predecir_cierre_mes(modelo, df_h, fecha_ultima):
    try:
        anio, mes = fecha_ultima.year, fecha_ultima.month
        fin_mes = pd.Timestamp(anio, mes, calendar.monthrange(anio, mes)[1])
        inicio_futuro = fecha_ultima + pd.Timedelta(days=1)
        if inicio_futuro > fin_mes: return pd.DataFrame(), 0
        
        rango = pd.date_range(inicio_futuro, fin_mes)
        df_p = []
        lag = df_h.groupby('Fecha')['Valor'].sum().iloc[-1]
        suma = 0
        for f in rango:
            ft = pd.DataFrame([{'DiaNum': f.dayofweek, 'DiaMes': f.day, 'Lag_1': lag, 'FechaOrdinal': f.toordinal()}])
            pred = max(0, modelo.predict(ft)[0])
            suma += pred; lag = pred
            df_p.append({'Fecha': f, 'Predicción': pred})
        return pd.DataFrame(df_p), suma
    except: return pd.DataFrame(), 0

# --- REPORTE WA ---
@st.cache_data(show_spinner=False)
def generar_reporte_pmv_whatsapp(df):
    try:
        if df.empty: return "https://wa.me/"
        anio = df['Año'].max()
        df_a = df[df['Año'] == anio]
        msg = f"*🏎️ GARI TELEMETRY {anio}*\n📅 {df_a['Fecha'].max().strftime('%d/%m/%Y')}\n\n🏢 *TOTAL:* ${df_a['Valor'].sum():,.0f}\n"
        for z, v in df_a.groupby('ZONA')['Valor'].sum().sort_values(ascending=False).items(): msg += f"\n📍 *{z}*: ${v:,.0f}"
        return f"https://wa.me/?text={urllib.parse.quote(msg)}"
    except: return "https://wa.me/"

# --- CARGA DATOS ---
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
        
        datos_zonas = {'CLINICAS': ['COLSUBSIDIO', 'CHAPINERO', 'TUNAL', 'SOACHA', 'PASEO VILLA DEL RIO', 'CENTRO MAYOR', 'MULTIPLAZA', 'SALITRE', 'UNICENTRO', 'ITAGUI', 'LA PLAYA', 'POBLADO', 'CALI CIUDAD JARDIN', 'CALLE 80', 'GRAN ESTACION', 'CEDRITOS', 'PORTAL 80', 'CENTRO', 'VILLAVICENCIO', 'KENNEDY', 'ROMA', 'VILLAS', 'ALAMOS', 'CALI AV 6TA', 'MALL PLAZA BOGOTA', 'CALI CALIMA', 'PLAZA DE LAS AMERICAS', 'SUBA PLAZA IMPERIAL', 'MALL PLAZA BARRANQUILLA', 'LA FLORESTA', 'PALMIRA', 'RESTREPO', 'MALL PLAZA CALI'], 'ZONA': ['ZONA 4', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 5', 'ZONA 3', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 4', 'ZONA 4', 'ZONA 2', 'ZONA 4', 'ZONA 1', 'ZONA 3', 'ZONA 1', 'ZONA 5', 'ZONA 3', 'ZONA 2', 'ZONA 2', 'ZONA 1', 'ZONA 4', 'ZONA 1'], 'CIUDAD': ['BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'MEDELLÍN', 'MEDELLÍN', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'VILLAVICENCIO', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'CALI', 'BOGOTÁ', 'BOGOTÁ', 'BARRANQUILLA', 'MEDELLÍN', 'CALI', 'BOGOTÁ', 'CALI'], 'RED': ['PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'PROPIA', 'FRANQUICIA', 'FRANQUICIA', 'FRANQUICIA', 'PROPIA']}
        df_z = pd.DataFrame(datos_zonas)
        df['Sucursal_Upper'] = df['Sucursal'].str.upper().str.strip()
        df_z['CLINICAS'] = df_z['CLINICAS'].str.upper().str.strip()
        df_f = df.merge(df_z, left_on='Sucursal_Upper', right_on='CLINICAS', how='left')
        df_f['ZONA'] = df_f['ZONA'].fillna('Sin Zona')
        df_f['CIUDAD'] = df_f['CIUDAD'].fillna('Otras')
        df_f['RED'] = df_f['RED'].fillna('No Def')
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
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color:#cc0000; border-bottom: 2px solid #fcd700; padding-bottom: 10px;">PIT WALL</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**👤 DRIVER:** `Conectado`")
    if st.button("BOX BOX (LOGOUT)"):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    
    # 🚨 AQUÍ ESTÁ LA MAGIA: CARGAR DATOS Y BOTÓN WA ANTES DE RENDERIZAR PÁGINAS 🚨
    with st.spinner("Calentando..."): df_raw = cargar_datos_integrados()
    
    if not df_raw.empty:
        link = generar_reporte_pmv_whatsapp(df_raw)
        st.markdown(f"""
        <a href="{link}" target="_blank">
            <button style="width:100%; background-color:#25D366; color:white; border:none; padding:10px; border-radius:4px; font-weight:bold; margin-bottom: 20px;">
            📲 RADIO A BOXES (WhatsApp)
            </button>
        </a>
        """, unsafe_allow_html=True)
        
    pagina = st.radio("MENÚ DE CARRERA", ["📊 Telemetría en Vivo", "🔮 Estrategia & Predicción", "🗺️ Track Map", "🧠 Ingeniero IA"])
    
    if "OPENAI_API_KEY" in st.secrets: api_key = st.secrets["OPENAI_API_KEY"]
    else: api_key = st.text_input("🔑 API KEY:", type="password")

# --- PÁGINAS ---
if pagina == "📊 Telemetría en Vivo":
    st.markdown("## 🏁 TELEMETRÍA DE COMANDO")
    if not df_raw.empty:
        with st.expander("🛠️ CONFIGURACIÓN DE PISTA", expanded=True):
            c1,c2,c3 = st.columns(3)
            sel_zona = c1.multiselect("SECTOR", sorted(df_raw['ZONA'].astype(str).unique()))
            df_v = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw
            sel_ciu = c2.multiselect("CIUDAD", sorted(df_v['CIUDAD'].astype(str).unique()))
            df_v = df_v[df_v['CIUDAD'].isin(sel_ciu)] if sel_ciu else df_v
            sel_red = c3.multiselect("RED", sorted(df_v['RED'].astype(str).unique()))
            df_v = df_v[df_v['RED'].isin(sel_red)] if sel_red else df_v
        
        if df_v.empty: st.stop()
        
        st.markdown("---")
        c1,c2 = st.columns([2,1])
        with c2: metric = st.radio("VISUALIZAR:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
        col_kpi = 'Valor' if metric == "Ventas ($)" else 'Tx'
        
        anio = df_v['Año'].max()
        df_act = df_v[df_v['Año'] == anio]
        df_ant = df_v[(df_v['Año'] == anio-1) & (df_v['Fecha'] <= df_act['Fecha'].max().replace(year=anio-1))]
        
        v_a, v_b = df_act['Valor'].sum(), df_ant['Valor'].sum()
        d_v = ((v_a - v_b)/v_b)*100 if v_b > 0 else 0
        
        k1,k2,k3 = st.columns(3)
        k1.metric(f"VENTAS {anio}", f"${v_a:,.0f}", f"{d_v:+.1f}%")
        k2.metric("TRANSACCIONES", f"{len(df_act):,}")
        k3.metric("ÚLTIMA VUELTA", df_act['Fecha'].max().strftime('%d/%m/%Y'))
        
        st.markdown("#### ⏱️ TIEMPOS POR TEMPORADA")
        df_y = df_v.groupby('Año').agg(Ventas=('Valor','sum'), Tx=('Tx','sum')).sort_index(ascending=False)
        df_y['Delta'] = df_y['Ventas'].pct_change(-1)*100
        st.table(df_y.style.format({"Ventas":"${:,.0f}","Tx":"{:,.0f}","Delta":"{:.1f}%"}).applymap(color_negative_red, subset=['Delta']))
        
        st.markdown("---")
        if not sel_zona or len(sel_zona)>1:
            st.pyplot(graficar_barras_pro(df_act.groupby('ZONA')[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False), 'ZONA', col_kpi, 'Ranking Sectores'))
        
        c1,c2 = st.columns(2)
        with c1:
            df_m = df_act.groupby('MesNum').agg({col_kpi:'sum'}).reset_index(); df_m['Mes'] = df_m['MesNum'].map(meses_es)
            st.pyplot(graficar_barras_pro(df_m, 'Mes', col_kpi, 'Ritmo Mensual'))
        with c2:
            df_d = df_act.groupby(['DiaNum','Dia']).agg({col_kpi:'sum'}).reset_index().sort_values('DiaNum')
            st.pyplot(graficar_barras_pro(df_d, 'Dia', col_kpi, 'Trazado Semanal', '#2c3e50'))
            
        st.markdown("### 🏥 3. TELEMETRÍA DETALLADA")
        for s in sorted(df_act['Sucursal'].unique()):
            d_s = df_act[df_act['Sucursal']==s]
            with st.expander(f"📍 {s}", expanded=False):
                c1,c2 = st.columns(2)
                with c1: st.pyplot(graficar_barras_pro(d_s.groupby('MesNum').agg({col_kpi:'sum'}).reset_index().assign(Mes=lambda x:x['MesNum'].map(meses_es)), 'Mes', col_kpi, 'Mensual'))
                with c2: st.pyplot(graficar_barras_pro(d_s.groupby(['DiaNum','Dia']).agg({col_kpi:'sum'}).reset_index().sort_values('DiaNum'), 'Dia', col_kpi, 'Semanal', '#8fa1b3'))
                
                df_tbl = d_s.groupby('MesNum').agg({'Valor':'sum','Tx':'sum'}).reset_index().assign(Mes=lambda x:x['MesNum'].map(meses_es))
                df_tbl['Var'] = df_tbl['Valor'].pct_change()*100
                st.table(df_tbl[['Mes','Valor','Var','Tx']].style.format({'Valor':'${:,.0f}','Var':'{:.1f}%'}).applymap(color_negative_red, subset=['Var']))

elif pagina == "🔮 Estrategia & Predicción":
    st.markdown("## 🔮 SIMULACIÓN DE ESTRATEGIA (IA)")
    if not df_raw.empty:
        anio, mes = df_raw['Año'].max(), df_raw['Fecha'].max().month
        df_act = df_raw[df_raw['Año']==anio]
        
        with st.expander("📂 VER HISTORIAL DE VUELTAS (2022-2025)", expanded=False):
            hist = df_raw.groupby('Fecha')['Valor'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10,3)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
            ax.plot(hist['Fecha'], hist['Valor'], color='#8fa1b3', linewidth=0.5); ax.tick_params(colors='white'); ax.set_title("Data Histórica", color='white')
            st.pyplot(fig)
            
        with st.spinner("Compitiendo motores..."):
            mod, met, nom = entrenar_modelo_predictivo(df_raw)
        
        if mod:
            st.success(f"🏎️ MOTOR GANADOR: **{nom}**")
            c1,c2=st.columns(2)
            c1.metric("PRECISIÓN (R²)", f"{met['R2']:.2f}"); c2.metric("ERROR (MAE)", f"${met['MAE']:,.0f}")
            
            df_p, sum_fut = predecir_cierre_mes(mod, df_raw, df_act['Fecha'].max())
            hoy = df_act[df_act['MesNum']==mes]['Valor'].sum()
            final = hoy + sum_fut
            
            st.markdown("---")
            k1,k2,k3 = st.columns(3)
            k1.metric("VUELTAS HOY", f"${hoy:,.0f}")
            k2.metric("RESTO CARRERA", f"${sum_fut:,.0f}")
            k3.metric("TIEMPO FINAL", f"${final:,.0f}")
            
            if not df_p.empty:
                fig, ax = plt.subplots(figsize=(10,3)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
                ax.plot(df_p['Fecha'].dt.day, df_p['Predicción'], marker='o', linestyle='--', color='#27ae60')
                ax.set_title("Ritmo Esperado", color='white'); ax.tick_params(colors='white'); st.pyplot(fig)
            
            meta = st.number_input("🎯 META", value=float(final*1.05))
            diff = final - meta
            if diff >= 0: st.success(f"✅ ESTRATEGIA GANADORA: +${diff:,.0f}")
            else: st.error(f"⚠️ GAP: -${abs(diff):,.0f}")

elif pagina == "🗺️ Track Map":
    st.markdown("## 🗺️ LOCALIZACIÓN DEL CIRCUITO"); st.map(pd.DataFrame({'lat': [4.6097], 'lon': [-74.0817]}))

elif pagina == "🧠 Ingeniero IA":
    st.markdown("## 📻 RADIO CHECK"); p = st.text_input("Consultar ingeniero...")
    if st.button("COPY THAT") and api_key:
        txt, fig, tbl, _ = analizar_gpt(df_raw, p, api_key)
        if txt: st.info(txt)
        if tbl is not None: st.dataframe(tbl)
        if fig: st.pyplot(fig)
