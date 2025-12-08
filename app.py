import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- HERRAMIENTAS ---
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

# --- FUNCIÓN GRÁFICA FLEXIBLE (DINERO O TRANSACCIONES) ---
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
    
    # Limpieza
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.set_title(titulo, fontsize=13, fontweight='bold', color='#2c3e50', pad=20)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    return fig

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
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
        
        # Columna Dummy para contar transacciones
        df['Tx'] = 1 
        
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return pd.DataFrame()

# --- FUNCIÓN CHAT ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        prompt_system = """
        Eres Gari. Usa 'df'. Reglas:
        1. Usa 'Fecha'.
        2. Outputs: 'resultado', 'tabla_resultados' (ordenada cronológicamente), 'fig'.
        """
        prompt_user = f"Info: {info_cols}\nMuestra: {muestra}\nPregunta: {pregunta}\nCódigo Python only."
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

with st.spinner("Cargando datos..."):
    df_raw = cargar_datos_sql()

# ==============================================================================
# PÁGINA 1: CHAT
# ==============================================================================
if pagina == "🧠 Chat con Gari":
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### Haz preguntas a tus datos")
    
    if not df_raw.empty:
        st.caption(f"Datos hasta: {df_raw['Fecha'].max().strftime('%d/%m/%Y')}")
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df_raw, pregunta, api_key)
                    if res_txt: st.success(f"📌 {res_txt}")
                    else: st.warning("Sin datos.")
                    if res_tabla is not None: st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}))
                    if res_fig: st.pyplot(res_fig)
                    with st.expander("Ver código"): st.code(cod)
            else: st.error("Falta API Key")

# ==============================================================================
# PÁGINA 2: REPORTES EJECUTIVOS BI (CON TRANSACCIONES)
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos BI":
    st.title("📊 Tablero de Comando Gerencial")
    
    if not df_raw.empty:
        
        # --- SELECTOR DE MÉTRICA PARA GRÁFICOS ---
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            metrica_grafico = st.radio("📊 Ver Gráficos por:", ["Ventas ($)", "Transacciones (#)"], horizontal=True)
            col_kpi = 'Valor' if metrica_grafico == "Ventas ($)" else 'Tx'
            fmt_kpi = 'dinero' if metrica_grafico == "Ventas ($)" else 'numero'
            color_kpi = '#3498db' if metrica_grafico == "Ventas ($)" else '#8e44ad' # Azul o Morado

        # --- 1. PULSO DEL NEGOCIO (YTD) ---
        st.header("1. Pulso del Negocio (YTD)")
        
        anio_actual = df_raw['Año'].max()
        anio_anterior = anio_actual - 1
        fecha_corte = df_raw[df_raw['Año'] == anio_actual]['Fecha'].max()
        fecha_limite_anterior = fecha_corte.replace(year=anio_anterior)
        
        # Cálculos Dinero
        v_act = df_raw[df_raw['Año'] == anio_actual]['Valor'].sum()
        v_ant = df_raw[(df_raw['Año'] == anio_anterior) & (df_raw['Fecha'] <= fecha_limite_anterior)]['Valor'].sum()
        var_v = ((v_act - v_ant) / v_ant) * 100 if v_ant > 0 else 0
        
        # Cálculos Transacciones
        tx_act = len(df_raw[df_raw['Año'] == anio_actual])
        tx_ant = len(df_raw[(df_raw['Año'] == anio_anterior) & (df_raw['Fecha'] <= fecha_limite_anterior)])
        var_tx = ((tx_act - tx_ant) / tx_ant) * 100 if tx_ant > 0 else 0
        
        # Ticket Promedio
        tk = v_act / tx_act if tx_act > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Ventas {anio_actual} (YTD)", f"${v_act:,.0f}", f"{var_v:+.1f}%")
        k2.metric(f"Transacciones (YTD)", f"{tx_act:,}", f"{var_tx:+.1f}%") # Ahora tiene semáforo
        k3.metric("Ticket Promedio", f"${tk:,.0f}")
        
        # Día más fuerte (según selección)
        df_dias_kpi = df_raw[df_raw['Año'] == anio_actual].groupby('Dia')[col_kpi].sum()
        mejor_dia = df_dias_kpi.idxmax()
        val_dia = df_dias_kpi.max()
        prefijo = "$" if fmt_kpi == 'dinero' else ""
        k4.metric(f"Día Top ({metrica_grafico})", mejor_dia, f"{prefijo}{val_dia:,.0f}")
        
        st.markdown("---")
        
        # --- 2. ANÁLISIS GLOBAL ---
        st.header(f"2. Análisis Global {anio_actual}")
        
        # A) GRÁFICO MESES
        st.subheader("A. Evolución Mensual")
        df_mes = df_raw[df_raw['Año'] == anio_actual].groupby('MesNum').agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
        df_mes['Mes'] = df_mes['MesNum'].map(meses_es)
        
        fig_mes = graficar_barras_pro(df_mes, 'Mes', col_kpi, f'Evolución Mensual ({metrica_grafico})', color_kpi, fmt_kpi)
        st.pyplot(fig_mes)
        
        # B) GRÁFICO DÍAS
        st.subheader("B. Patrón Semanal")
        df_dias = df_raw[df_raw['Año'] == anio_actual].groupby(['DiaNum', 'Dia']).agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
        df_dias['Dia'] = pd.Categorical(df_dias['Dia'], categories=orden_dias, ordered=True)
        df_dias = df_dias.sort_values('Dia')
        
        fig_dias = graficar_barras_pro(df_dias, 'Dia', col_kpi, f'Comportamiento Semanal ({metrica_grafico})', '#2ecc71', fmt_kpi)
        st.pyplot(fig_dias)

        # C) TABLA MAESTRA (DOBLE MÉTRICA)
        st.subheader("C. Detalle Completo")
        
        # Variaciones
        df_mes['Var $'] = df_mes['Valor'].pct_change() * 100
        df_mes['Var Tx'] = df_mes['Tx'].pct_change() * 100
        
        st.table(
            df_mes[['Mes', 'Valor', 'Var $', 'Tx', 'Var Tx']].style
            .format({
                "Valor": "${:,.0f}", "Var $": "{:+.1f}%", 
                "Tx": "{:,.0f}", "Var Tx": "{:+.1f}%"
            })
            .applymap(color_negative_red, subset=['Var $', 'Var Tx'])
        )

        st.markdown("---")
        st.header("🏥 Análisis por Clínica")
        st.info("Despliega para ver detalle. Los gráficos responden al selector superior.")

        # --- 3. POR CLÍNICA ---
        sucursales = sorted(df_raw['Sucursal'].unique())
        
        for suc in sucursales:
            with st.expander(f"📍 {suc}", expanded=False):
                df_suc = df_raw[(df_raw['Sucursal'] == suc) & (df_raw['Año'] == anio_actual)]
                
                if not df_suc.empty:
                    c1, c2 = st.columns(2)
                    
                    # Gráficos Dinámicos
                    with c1:
                        df_s_mes = df_suc.groupby('MesNum').agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
                        df_s_mes['Mes'] = df_s_mes['MesNum'].map(meses_es)
                        fig_sm = graficar_barras_pro(df_s_mes, 'Mes', col_kpi, 'Mensual', color_kpi, fmt_kpi)
                        st.pyplot(fig_sm)
                        
                    with c2:
                        df_s_dia = df_suc.groupby(['DiaNum', 'Dia']).agg({'Valor': 'sum', 'Tx': 'sum'}).reset_index()
                        df_s_dia['Dia'] = pd.Categorical(df_s_dia['Dia'], categories=orden_dias, ordered=True)
                        df_s_dia = df_s_dia.sort_values('Dia')
                        fig_sd = graficar_barras_pro(df_s_dia, 'Dia', col_kpi, 'Semanal', '#2ecc71', fmt_kpi)
                        st.pyplot(fig_sd)
                    
                    # Tabla Doble
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
                else:
                    st.warning("Sin movimientos este año.")

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
