import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
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

# --- CONFIGURACIÓN DE PÁGINA (WIDE & DARK PRE-CONFIG) ---
st.set_page_config(
    page_title="Gari | Oracle Red Bull Data", 
    page_icon="🏎️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🎨 ESTILOS CSS PERSONALIZADOS (RED BULL RACING THEME)
# ==============================================================================
st.markdown("""
    <style>
        /* Importar fuente futurista */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

        /* FONDO PRINCIPAL - Azul Oscuro Mate RB */
        .stApp {
            background-color: #060818;
            background-image: linear-gradient(180deg, #060818 0%, #0b1026 100%);
            color: #ffffff;
        }

        /* TIPOGRAFÍA */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            color: #ffffff !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        p, div, label {
            font-family: 'Roboto', sans-serif;
            color: #e0e0e0;
        }

        /* KPI CARDS (Estilo Telemetría) */
        div[data-testid="stMetric"] {
            background-color: #151925;
            border-left: 4px solid #cc0000; /* Rojo RB */
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        div[data-testid="stMetricLabel"] {
            color: #8fa1b3 !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.8rem !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #fcd700 !important; /* Amarillo RB */
        }

        /* SIDEBAR (Estilo Fibra de Carbono) */
        section[data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #333;
        }

        /* BOTONES (Estilo Botones Volante F1) */
        .stButton > button {
            background: linear-gradient(90deg, #cc0000 0%, #990000 100%);
            color: white;
            border: none;
            border-radius: 4px;
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            text-transform: uppercase;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 10px rgba(204, 0, 0, 0.6);
        }

        /* TABLAS */
        .stDataFrame {
            border: 1px solid #333;
        }
        
        /* INPUTS */
        .stTextInput > div > div > input {
            background-color: #151925;
            color: white;
            border: 1px solid #444;
        }
        
        /* MENSAJES */
        .stAlert {
            background-color: #151925;
            border: 1px solid #444;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE SESIÓN Y LOGIN (Estilo Paddock) ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    def login_form():
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="color:#cc0000; font-size: 3rem;">GARI</h1>
                <h3 style="color:#fcd700;">RACING DATA HQ</h3>
                <hr style="border-color: #cc0000;">
            </div>
            """, unsafe_allow_html=True)
            
            usuario = st.text_input("PILOTO (Usuario)")
            clave = st.text_input("CÓDIGO DE ACCESO (Password)", type="password")
            
            if st.button("START ENGINE 🏁"):
                usuarios_validos = {
                    "gerente": "alivio2025", 
                    "admin": "admin123",
                    "gari": "hamster"
                }
                if usuario in usuarios_validos and usuarios_validos[usuario] == clave:
                    st.session_state.authenticated = True
                    st.success("ACCESS GRANTED. WELCOME TO THE PADDOCK.")
                    time.sleep(1)
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

# --- SIDEBAR: PIT WALL ---
st.sidebar.markdown("### 🏎️ PIT WALL")
st.sidebar.markdown(f"👨‍✈️ **Piloto:** Conectado")
if st.sidebar.button("BOX BOX (Salir)"):
    st.session_state.authenticated = False
    st.rerun()
st.sidebar.markdown("---")

# --- HERRAMIENTAS ---
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
dias_es = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# --- GRÁFICOS ESTILO DARK MODE (TELEMETRÍA) ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#cc0000', formato='dinero'):
    # Fondo oscuro para la figura
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0E1117') # Mismo color que el fondo de la app o similar
    ax.set_facecolor('#0E1117')
    
    # Barras con borde brillante
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='#fcd700', linewidth=0.5, alpha=0.9)
    
    # Etiquetas
    fmt = '${:,.0f}' if formato == 'dinero' else '{:,.0f}'
    ax.bar_label(bars, fmt=fmt, padding=3, rotation=90, fontsize=10, fontweight='bold', color='white')
    
    # Ejes y textos en blanco
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Limpieza
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.set_title(titulo, fontsize=13, fontweight='bold', color='white', pad=20)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    return fig

# --- FUNCIÓN IA HÍBRIDA ---
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

        split_point = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        modelo_rf.fit(X_train, y_train)
        r2_rf = r2_score(y_test, modelo_rf.predict(X_test))

        modelo_lr = LinearRegression()
        modelo_lr.fit(X_train, y_train)
        r2_lr = r2_score(y_test, modelo_lr.predict(X_test))

        if r2_rf > 0 and r2_rf > r2_lr:
            return modelo_rf, {"R2": r2_rf, "MAE": mean_absolute_error(y_test, modelo_rf.predict(X_test))}, "Random Forest (Aero Package)"
        else:
            return modelo_lr, {"R2": r2_lr, "MAE": mean_absolute_error(y_test, modelo_lr.predict(X_test))}, "Linear Engine (Top Speed)"

    except Exception as e:
        return None, None, "Engine Fail"

def predecir_cierre_mes(modelo, df_historico, fecha_ultima_real):
    try:
        anio = fecha_ultima_real.year
        mes = fecha_ultima_real.month
        _, last_day = calendar.monthrange(anio, mes)
        fecha_fin_mes = pd.Timestamp(anio, mes, last_day)
        
        fecha_inicio_futuro = fecha_ultima_real + pd.Timedelta(days=1)
        if fecha_inicio_futuro > fecha_fin_mes: return pd.DataFrame(), 0
        
        rango_futuro = pd.date_range(start=fecha_inicio_futuro, end=fecha_fin_mes)
        df_predicciones = []
        lag_actual = df_historico.groupby('Fecha')['Valor'].sum().iloc[-1]
        predicciones_sum = 0
        
        for fecha in rango_futuro:
            features = pd.DataFrame([{
                'DiaNum': fecha.dayofweek, 'DiaMes': fecha.day,
                'Lag_1': lag_actual, 'FechaOrdinal': fecha.toordinal()
            }])
            pred = max(0, modelo.predict(features)[0])
            predicciones_sum += pred
            lag_actual = pred 
            df_predicciones.append({'Fecha': fecha, 'Predicción': pred})
        
        return pd.DataFrame(df_predicciones), predicciones_sum
    except: return pd.DataFrame(), 0

# --- REPORTE WA ---
@st.cache_data(show_spinner=False) 
def generar_reporte_pmv_whatsapp(df):
    try:
        if df.empty: return "https://wa.me/"
        anio_actual = df['Año'].max()
        df_act = df[df['Año'] == anio_actual].copy()
        if df_act.empty: return "https://wa.me/"

        v_total = df_act['Valor'].sum()
        mensaje = f"*🏎️ GARI TELEMETRY {anio_actual}*\n\n🏢 *TOTAL:* ${v_total:,.0f}\n"
        
        df_zonas = df_act.groupby('ZONA')['Valor'].sum().sort_values(ascending=False)
        for zona, valor in df_zonas.items():
            mensaje += f"\n📍 *{zona}*: ${valor:,.0f}"

        mensaje_codificado = urllib.parse.quote(mensaje)
        return f"https://wa.me/?text={mensaje_codificado}"
    except: return "https://wa.me/"

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
        
        # ZONAS HARDCODED
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
        except:
            df_final = df
            df_final['ZONA'] = 'General'
            df_final['CIUDAD'] = 'General'
            df_final['RED'] = 'General'
        return df_final
    except: return pd.DataFrame()

# --- CHAT GPT ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        buffer = io.StringIO()
        df.head(5).to_csv(buffer, index=False)
        prompt_system = "Eres un ingeniero de datos de F1. Responde con código Python. Usa 'df'."
        prompt_user = f"Data: {buffer.getvalue()}\nPregunta: {pregunta}\nCode only."
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}], temperature=0)
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        local_vars = {'df': df, 'pd': pd, 'plt': plt}
        exec(codigo, globals(), local_vars)
        return local_vars.get('resultado', None), local_vars.get('fig', None), local_vars.get('tabla_resultados', None), codigo
    except: return "Error en boxes", None, None, ""

# --- NAVEGACIÓN ---
pagina = st.sidebar.radio("MENÚ DE CARRERA", ["📊 Telemetría en Vivo", "🔮 Estrategia & Predicción", "🗺️ Track Map", "🧠 Ingeniero IA"])

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 API KEY ENGINE:", type="password")

with st.spinner("Calentando neumáticos..."):
    df_raw = cargar_datos_integrados()

# ==============================================================================
# 📊 PÁGINA 1: TELEMETRÍA (COMANDO)
# ==============================================================================
if pagina == "📊 Telemetría en Vivo":
    st.markdown("## 🏁 TELEMETRÍA DE VENTAS")
    
    if not df_raw.empty:
        with st.expander("🛠️ CONFIGURACIÓN DE PISTA (FILTROS)", expanded=False):
            c1, c2, c3 = st.columns(3)
            opc_zona = sorted(df_raw['ZONA'].astype(str).unique())
            sel_zona = c1.multiselect("SECTOR (Zona)", opc_zona)
            df_view = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw

        # KPIs
        anio_actual = df_view['Año'].max()
        df_act = df_view[df_view['Año'] == anio_actual]
        v_act = df_act['Valor'].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"VENTAS {anio_actual}", f"${v_act:,.0f}", "Sector 1")
        c2.metric("TRANSACCIONES", f"{len(df_act):,}", "Sector 2")
        c3.metric("TICKET PROMEDIO", f"${v_act/len(df_act):,.0f}" if len(df_act)>0 else "0", "Sector 3")

        st.markdown("---")
        
        # Gráficos
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### 📈 RITMO DE CARRERA (MENSUAL)")
            df_mes = df_act.groupby('Mes')['Valor'].sum().reset_index()
            st.pyplot(graficar_barras_pro(df_mes, 'Mes', 'Valor', 'Evolución', '#cc0000'))
        
        with g2:
            st.markdown("### 🏆 POSICIONES (ZONAS)")
            df_z = df_act.groupby('ZONA')['Valor'].sum().reset_index().sort_values('Valor', ascending=False)
            st.pyplot(graficar_barras_pro(df_z, 'ZONA', 'Valor', 'Ranking', '#fcd700'))

# ==============================================================================
# 🔮 PÁGINA 2: ESTRATEGIA (PREDICCIONES)
# ==============================================================================
elif pagina == "🔮 Estrategia & Predicción":
    st.markdown("## 🔮 SIMULACIÓN DE CARRERA (IA)")
    
    if not df_raw.empty:
        anio_actual = df_raw['Año'].max()
        df_act = df_raw[df_raw['Año'] == anio_actual]
        mes_actual = df_act['Fecha'].max().month
        nombre_mes = meses_es[mes_actual]
        
        with st.spinner("Corriendo simulaciones en el túnel de viento..."):
            modelo, metricas, nombre_modelo = entrenar_modelo_predictivo(df_raw)
        
        if modelo:
            st.success(f"🏎️ MOTOR SELECCIONADO: **{nombre_modelo}**")
            
            # Auditoría
            m1, m2 = st.columns(2)
            m1.metric("PRECISIÓN MOTOR (R²)", f"{metricas['R2']:.2f}")
            m2.metric("MARGEN ERROR (MAE)", f"${metricas['MAE']:,.0f}")
            
            # Proyección
            fecha_max = df_act['Fecha'].max()
            df_pred, suma_futura = predecir_cierre_mes(modelo, df_raw, fecha_max)
            venta_hoy = df_act[df_act['MesNum'] == mes_actual]['Valor'].sum()
            cierre = venta_hoy + suma_futura
            
            st.markdown("---")
            k1, k2, k3 = st.columns(3)
            k1.metric("VUELTAS COMPLETADAS (HOY)", f"${venta_hoy:,.0f}")
            k2.metric("RESTO DE CARRERA (PRED)", f"${suma_futura:,.0f}")
            k3.metric("TIEMPO FINAL ESTIMADO", f"${cierre:,.0f}", delta="Puesto 1")
            
            # Meta
            meta = st.number_input(f"🎯 TIEMPO OBJETIVO (META {nombre_mes})", value=float(cierre*1.05))
            diff = cierre - meta
            if diff >= 0: st.success(f"✅ ESTRATEGIA GANADORA: +${diff:,.0f}")
            else: st.error(f"⚠️ PELIGRO: GAP DE -${abs(diff):,.0f}")

# ==============================================================================
# 🗺️ PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Track Map":
    st.markdown("## 🗺️ LOCALIZACIÓN DEL CIRCUITO")
    st.map(pd.DataFrame({'lat': [4.6097], 'lon': [-74.0817]}))

# ==============================================================================
# 🧠 PÁGINA 4: CHAT
# ==============================================================================
elif pagina == "🧠 Ingeniero IA":
    st.markdown("## 📻 RADIO CHECK (CHAT IA)")
    pregunta = st.text_input("Ingeniero, dame datos sobre...", "Mejor día de venta")
    if st.button("COPY THAT"):
        if api_key:
            res_txt, res_fig, res_tbl, _ = analizar_con_gpt(df_raw, pregunta, api_key)
            if res_txt: st.info(f"📻 {res_txt}")
            if res_tbl is not None: st.dataframe(res_tbl)
            if res_fig: st.pyplot(res_fig)

# --- WHATSAPP BUTTON ---
if not df_raw.empty:
    st.sidebar.markdown("---")
    link = generar_reporte_pmv_whatsapp(df_raw)
    st.sidebar.markdown(f"""
    <a href="{link}" target="_blank">
        <button style="width:100%; background-color:#25D366; color:white; border:none; padding:10px; border-radius:4px; font-weight:bold;">
        📲 RADIO A BOXES (WhatsApp)
        </button>
    </a>
    """, unsafe_allow_html=True)
