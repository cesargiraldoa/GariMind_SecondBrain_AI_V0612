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

# --- LIBRERÍAS DE MACHINE LEARNING ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Comando y Control", page_icon="📊", layout="wide")

# --- GESTIÓN DE SESIÓN Y LOGIN ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    def login_form():
        st.title("🔒 Acceso Corporativo")
        st.write("Reporte de Comando y Control - Inicie Sesión")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            usuario = st.text_input("Usuario")
            clave = st.text_input("Contraseña", type="password")
            if st.button("Ingresar 🔐"):
                usuarios_validos = {
                    "gerente": "alivio2025", 
                    "admin": "admin123",
                    "gari": "hamster"
                }
                if usuario in usuarios_validos and usuarios_validos[usuario] == clave:
                    st.session_state.authenticated = True
                    st.success("Validando credenciales...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Credenciales inválidas.")

    if not st.session_state.authenticated:
        login_form()
        return False
    return True

if not check_password():
    st.stop()

# ==============================================================================
# 🚀 APLICACIÓN PRINCIPAL
# ==============================================================================

# --- SIDEBAR ---
st.sidebar.markdown(f"👤 **Usuario:** Activo")
if st.sidebar.button("Cerrar Sesión 🔒"):
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

def color_negative_red(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4b4b; font-weight: bold'
    return 'color: black'

def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#2c3e50', formato='dinero'):
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

# --- FUNCIÓN INTELIGENCIA ARTIFICIAL (SCIKIT-LEARN) ---
@st.cache_resource
def entrenar_modelo_predictivo(df):
    """
    Entrena un modelo Random Forest con los datos históricos.
    Retorna: modelo, métricas, datos de test.
    """
    try:
        # 1. Preparar Datos para ML
        # Agrupamos por fecha para tener venta diaria total
        df_ml = df.groupby('Fecha')['Valor'].sum().reset_index()
        
        # Ingeniería de Características (Features)
        df_ml['DiaNum'] = df_ml['Fecha'].dt.dayofweek
        df_ml['DiaMes'] = df_ml['Fecha'].dt.day
        df_ml['Mes'] = df_ml['Fecha'].dt.month
        df_ml['EsFinDeSemana'] = df_ml['DiaNum'].apply(lambda x: 1 if x >= 5 else 0)
        # Lag (Venta de ayer) - ayuda a capturar tendencias inmediatas
        df_ml['Lag_1'] = df_ml['Valor'].shift(1).fillna(0)
        
        # Eliminamos primera fila por el Lag vacío
        df_ml = df_ml.iloc[1:]

        if len(df_ml) < 10:
            return None, None # Muy pocos datos para entrenar

        X = df_ml[['DiaNum', 'DiaMes', 'Mes', 'EsFinDeSemana', 'Lag_1']]
        y = df_ml['Valor']

        # 2. Split Train/Test (80% entrenar, 20% validar)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # 3. Entrenar Random Forest
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # 4. Evaluar
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {"MAE": mae, "R2": r2}
        
        return modelo, metrics
    except Exception as e:
        print(f"Error ML: {e}")
        return None, None

def predecir_cierre_mes(modelo, df_historico, fecha_ultima_real):
    """Usa el modelo entrenado para predecir los días restantes del mes."""
    anio = fecha_ultima_real.year
    mes = fecha_ultima_real.month
    _, last_day = calendar.monthrange(anio, mes)
    fecha_fin_mes = pd.Timestamp(anio, mes, last_day)
    
    # Generar rango de fechas futuras (desde mañana hasta fin de mes)
    fecha_inicio_futuro = fecha_ultima_real + pd.Timedelta(days=1)
    
    if fecha_inicio_futuro > fecha_fin_mes:
        return pd.DataFrame(), 0 # Ya acabó el mes
    
    rango_futuro = pd.date_range(start=fecha_inicio_futuro, end=fecha_fin_mes)
    
    futuro_data = []
    ultima_venta_conocida = df_historico.groupby('Fecha')['Valor'].sum().iloc[-1]
    
    predicciones_sum = 0
    df_predicciones = []

    # Predicción iterativa (Rolling Forecast)
    # Necesitamos predecir el día 1 para usarlo como 'Lag' del día 2, etc.
    lag_actual = ultima_venta_conocida
    
    for fecha in rango_futuro:
        features = {
            'DiaNum': fecha.dayofweek,
            'DiaMes': fecha.day,
            'Mes': fecha.month,
            'EsFinDeSemana': 1 if fecha.dayofweek >= 5 else 0,
            'Lag_1': lag_actual
        }
        X_futuro = pd.DataFrame([features])
        pred = modelo.predict(X_futuro)[0]
        
        # Guardamos para el reporte y para el siguiente lag
        predicciones_sum += pred
        lag_actual = pred 
        
        df_predicciones.append({'Fecha': fecha, 'Predicción': pred})
    
    return pd.DataFrame(df_predicciones), predicciones_sum

# --- REPORTE WA ---
@st.cache_data(show_spinner=False) 
def generar_reporte_pmv_whatsapp(df):
    try:
        if df.empty: return "https://wa.me/"
        anio_actual = df['Año'].max()
        df_act = df[df['Año'] == anio_actual].copy()
        
        if df_act.empty: return "https://wa.me/"

        v_total = df_act['Valor'].sum()
        tx_total = len(df_act)
        df_zonas = df_act.groupby('ZONA')['Valor'].sum().sort_values(ascending=False)
        df_detalle = df_act.groupby(['ZONA', 'Sucursal'])['Valor'].sum().reset_index()
        
        mensaje = f"*📊 REPORTE COMANDO Y CONTROL {anio_actual}*\n"
        mensaje += f"📅 Corte: {df_act['Fecha'].max().strftime('%d/%m/%Y')}\n\n"
        mensaje += f"🏢 *TOTAL COMPAÑÍA*\n"
        mensaje += f"💰 Venta: ${v_total:,.0f}\n"
        mensaje += f"🧾 Tx: {tx_total:,.0f}\n"
        mensaje += "➖➖➖➖➖➖➖➖\n"

        for zona, valor_zona in df_zonas.items():
            mensaje += f"\n📍 *{zona}*: ${valor_zona:,.0f}\n"
            sucursales_zona = df_detalle[df_detalle['ZONA'] == zona].sort_values('Valor', ascending=False)
            for _, row in sucursales_zona.iterrows():
                mensaje += f"   • {row['Sucursal']}: ${row['Valor']:,.0f}\n"

        mensaje += "\n_Generado por Sistema de Comando_ 🤖"
        mensaje_codificado = urllib.parse.quote(mensaje)
        return f"https://wa.me/?text={mensaje_codificado}"
    except Exception as e:
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

# --- CHAT ---
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
        Eres un analista de datos experto. Output rules: 'resultado', 'tabla_resultados', 'fig'. Code only.
        """
        prompt_user = f"Info: {info_cols}\nMuestra: {muestra}\nPregunta: {pregunta}\nCode only."
        
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}], temperature=0)
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker, 'meses_es': meses_es}
        exec(codigo, globals(), local_vars)
        return (local_vars.get('resultado', None), local_vars.get('fig', None), local_vars.get('tabla_resultados', None), codigo)
    except Exception as e: return f"Error: {e}", None, None, ""

# --- NAVEGACIÓN ---
pagina = st.sidebar.radio("Navegación", ["📊 Reporte Comando y Control", "🔮 Predicciones AI", "🗺️ Mapa", "🧠 Chat IA"])

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 API Key:", type="password")

with st.spinner("Conectando con base de datos..."):
    df_raw = cargar_datos_integrados()

# ==============================================================================
# PÁGINA: REPORTE COMANDO Y CONTROL
# ==============================================================================
if pagina == "📊 Reporte Comando y Control":
    st.title("Reporte de Comando y Control")
    
    if not df_raw.empty:
        with st.expander("🔍 Filtros de Visualización", expanded=False):
            c1, c2, c3 = st.columns(3)
            opc_zona = sorted(df_raw['ZONA'].astype(str).unique())
            sel_zona = c1.multiselect("Zona", opc_zona)
            df_temp = df_raw[df_raw['ZONA'].isin(sel_zona)] if sel_zona else df_raw
            opc_ciudad = sorted(df_temp['CIUDAD'].astype(str).unique())
            sel_ciudad = c2.multiselect("Ciudad", opc_ciudad)
            df_temp2 = df_temp[df_temp['CIUDAD'].isin(sel_ciudad)] if sel_ciudad else df_temp
            opc_red = sorted(df_temp2['RED'].astype(str).unique())
            sel_red = c3.multiselect("Red", opc_red)

        df_view = df_raw.copy()
        if sel_zona: df_view = df_view[df_view['ZONA'].isin(sel_zona)]
        if sel_ciudad: df_view = df_view[df_view['CIUDAD'].isin(sel_ciudad)]
        if sel_red: df_view = df_view[df_view['RED'].isin(sel_red)]
            
        if df_view.empty:
            st.warning("Sin datos para los filtros seleccionados.")
            st.stop()

        # KPIs
        st.markdown("### 1. Pulso del Negocio (YTD)")
        col_t1, col_t2 = st.columns([3, 1])
        with col_t2:
            metrica = st.selectbox("Métrica:", ["Ventas ($)", "Transacciones (#)"])
            col_kpi = 'Valor' if metrica == "Ventas ($)" else 'Tx'
            fmt_kpi = 'dinero' if metrica == "Ventas ($)" else 'numero'

        anio_actual = df_view['Año'].max()
        anio_anterior = anio_actual - 1
        df_act = df_view[df_view['Año'] == anio_actual]
        fecha_max = df_act['Fecha'].max()
        
        fecha_limite_ant = fecha_max.replace(year=anio_anterior)
        df_ant = df_view[(df_view['Año'] == anio_anterior) & (df_view['Fecha'] <= fecha_limite_ant)]
        
        v_act = df_act['Valor'].sum()
        v_ant = df_ant['Valor'].sum()
        delta_v = ((v_act - v_ant) / v_ant) * 100 if v_ant > 0 else 0
        
        tx_act = len(df_act)
        tx_ant = len(df_ant)
        delta_tx = ((tx_act - tx_ant) / tx_ant) * 100 if tx_ant > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Ventas {anio_actual}", f"${v_act:,.0f}", f"{delta_v:+.1f}% vs Año Ant")
        k2.metric(f"Transacciones", f"{tx_act:,}", f"{delta_tx:+.1f}% vs Año Ant")
        k3.metric("Última Actualización", fecha_max.strftime('%d/%m/%Y'))

        st.markdown("---")
        
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.subheader("Evolución Mensual")
            df_mes = df_act.groupby('MesNum').agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
            df_mes['Mes'] = df_mes['MesNum'].map(meses_es)
            st.pyplot(graficar_barras_pro(df_mes, 'Mes', col_kpi, f'Tendencia {metrica}', '#2c3e50', fmt_kpi))
            
        with c_g2:
            st.subheader("Ranking por Zona")
            df_z = df_act.groupby('ZONA')[col_kpi].sum().reset_index().sort_values(col_kpi, ascending=False)
            st.pyplot(graficar_barras_pro(df_z, 'ZONA', col_kpi, f'Top Zonas {metrica}', '#e67e22', fmt_kpi))

        st.markdown("---")
        st.subheader("Detalle Operativo por Clínica")
        with st.expander("Ver Tabla Completa", expanded=False):
            st.dataframe(df_act.groupby(['ZONA', 'Sucursal'])[['Valor', 'Tx']].sum().sort_values('Valor', ascending=False))

# ==============================================================================
# PÁGINA: PREDICCIONES AI (RANDOM FOREST)
# ==============================================================================
elif pagina == "🔮 Predicciones AI":
    st.title("🔮 Modelo Predictivo IA (Random Forest)")
    
    if not df_raw.empty:
        # 1. Preparación de Datos
        anio_actual = df_raw['Año'].max()
        df_act = df_raw[df_raw['Año'] == anio_actual]
        mes_actual = df_act['Fecha'].max().month
        nombre_mes = meses_es[mes_actual]
        fecha_max = df_act['Fecha'].max()
        
        st.markdown(f"### 🤖 Entrenamiento del Modelo Predictivo")
        st.write("El sistema está analizando patrones históricos (Día de la semana, estacionalidad mensual y tendencias recientes) usando un algoritmo de **Bosques Aleatorios (Random Forest)**.")
        
        with st.spinner("Entrenando red neuronal simplificada..."):
            modelo, metricas = entrenar_modelo_predictivo(df_raw)
        
        if modelo:
            # Mostrar Métricas de Calidad
            st.success("Modelo entrenado exitosamente.")
            
            with st.expander("📊 Ver Métricas de Confianza del Modelo (Auditoría Técnica)", expanded=True):
                m1, m2 = st.columns(2)
                r2_val = metricas['R2']
                mae_val = metricas['MAE']
                
                m1.metric("R² (Precisión de Varianza)", f"{r2_val:.2f}", help="Indica qué tan bien el modelo replica los patrones históricos. 1.0 es perfecto, 0.0 es aleatorio.")
                m2.metric("MAE (Margen de Error Diario)", f"${mae_val:,.0f}", help="Promedio de error en pesos que el modelo puede tener por día.")
                
                if r2_val > 0.7:
                    st.caption("✅ **Modelo Confiable:** El R² indica una alta capacidad predictiva.")
                elif r2_val > 0.4:
                    st.caption("⚠️ **Modelo Regular:** Puede servir como guía, pero con cautela.")
                else:
                    st.caption("❌ **Modelo No Confiable:** Faltan datos históricos para patrones claros.")

            # --- PROYECCIÓN ---
            st.markdown("---")
            st.header(f"1. Proyección de Cierre ({nombre_mes})")
            
            df_pred, suma_futura = predecir_cierre_mes(modelo, df_raw, fecha_max)
            
            venta_acumulada_hoy = df_act[df_act['MesNum'] == mes_actual]['Valor'].sum()
            
            if not df_pred.empty:
                cierre_estimado = venta_acumulada_hoy + suma_futura
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Venta Real (Hoy)", f"${venta_acumulada_hoy:,.0f}")
                k2.metric("Predicción Días Restantes", f"${suma_futura:,.0f}", f"{len(df_pred)} días")
                k3.metric("Cierre Estimado IA", f"${cierre_estimado:,.0f}", delta="Modelo ML")
                
                # Gráfico de la Predicción
                st.subheader("📅 Calendario Predictivo")
                df_pred['Día'] = df_pred['Fecha'].dt.day
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_pred['Día'], df_pred['Predicción'], marker='o', linestyle='--', color='#27ae60', label='Predicción IA')
                ax.set_title("Comportamiento Esperado para el Resto del Mes")
                ax.set_xlabel("Día del Mes")
                ax.set_ylabel("Venta Proyectada")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.success(f"🏁 Mes terminado. Cierre Total: ${venta_acumulada_hoy:,.0f}")
                cierre_estimado = venta_acumulada_hoy

            # --- CONTROL DE METAS ---
            st.markdown("---")
            st.header("2. Control vs Meta")
            
            c_meta1, c_meta2 = st.columns([1, 2])
            with c_meta1:
                meta_input = st.number_input(f"Define tu Meta para {nombre_mes} ($)", value=float(cierre_estimado * 1.05), step=1000000.0)
            
            with c_meta2:
                diff = cierre_estimado - meta_input
                pct_cumplimiento = (cierre_estimado / meta_input) * 100
                
                st.metric("Cumplimiento Proyectado (IA)", f"{pct_cumplimiento:.1f}%", f"${diff:,.0f} vs Meta")
                
                if diff < 0:
                    st.warning(f"⚠️ La IA predice que faltarán **${abs(diff):,.0f}** para la meta.")
                else:
                    st.success("🚀 La IA predice que superarás la meta.")

        else:
            st.warning("No hay suficientes datos históricos para entrenar la IA (mínimo 10 días).")

# ==============================================================================
# PÁGINA: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa de Cobertura")
    st.map(pd.DataFrame({'lat': [4.6097], 'lon': [-74.0817]}))
    st.info("Visualización geográfica de puntos de venta.")

# ==============================================================================
# PÁGINA: CHAT IA
# ==============================================================================
elif pagina == "🧠 Chat IA":
    st.title("Asistente de Inteligencia Artificial")
    st.write("Haz preguntas libres sobre tus datos.")
    
    if not df_raw.empty:
        pregunta = st.text_input("Pregunta:", "¿Cuál fue el mejor día de ventas?")
        if st.button("Consultar"):
            if api_key:
                with st.spinner("Analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df_raw, pregunta, api_key)
                    if res_txt: st.success(res_txt)
                    if res_tabla is not None: st.dataframe(res_tabla)
                    if res_fig: st.pyplot(res_fig)
            else:
                st.error("Requiere API Key")

# ==============================================================================
# BOTÓN WHATSAPP
# ==============================================================================
if not df_raw.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("📲 Comunicación")
    link_wa = generar_reporte_pmv_whatsapp(df_raw)
    st.sidebar.markdown(f"""
    <a href="{link_wa}" target="_blank">
        <button style="background-color:#25D366; color:white; border:none; padding:10px; border-radius:5px; width:100%; font-weight:bold;">
        📤 Enviar Reporte Gerencial
        </button>
    </a>
    """, unsafe_allow_html=True)
