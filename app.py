import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gari", page_icon="🐹", layout="wide")

# --- HERRAMIENTAS DE TIEMPO ---
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

dias_es = {
    0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
    4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
}

# --- ESTILOS CSS ---
def color_negative_red(val):
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4b4b; font-weight: bold'
    return 'color: black'

# --- FUNCIÓN GRÁFICA MEJORADA (VERTICAL + LIMPIA) ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, color_barras='#3498db'):
    fig, ax = plt.subplots(figsize=(10, 5)) # Un poco más alto
    
    # Barras
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras, edgecolor='none', alpha=0.85)
    
    # ETIQUETAS VERTICALES (SOLUCIÓN AL TRASLAPE)
    # padding=5 las separa de la barra, rotation=90 las pone paradas
    ax.bar_label(bars, fmt='${:,.0f}', padding=5, rotation=90, fontsize=9, fontweight='bold', color='#2c3e50')
    
    # Aumentar el techo del gráfico (margen superior) para que quepan los números
    y_max = df_g[y_col].max()
    ax.set_ylim(0, y_max * 1.35) # 35% más de espacio arriba
    
    # Limpieza visual
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.set_title(titulo, fontsize=13, fontweight='bold', color='#2c3e50', pad=20)
    plt.xticks(rotation=0, fontsize=10) # Texto horizontal en eje X
    plt.tight_layout()
    return fig

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        # Columnas BI
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaNum'] = df['Fecha'].dt.dayofweek
        df['Dia'] = df['DiaNum'].map(dias_es)
        
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
# PÁGINA 2: REPORTES EJECUTIVOS BI (VERSION MAESTRA)
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos BI":
    st.title("📊 Tablero de Comando Gerencial")
    
    if not df_raw.empty:
        
        # --- SECCIÓN A: KPIS ESTRATÉGICOS (¿VAMOS GANANDO?) ---
        st.header("1. Pulso del Negocio (YTD)")
        
        # Lógica de Comparación Año a la Fecha (YTD)
        anio_actual = df_raw['Año'].max()
        anio_anterior = anio_actual - 1
        
        # Fecha máxima de datos en el año actual (para cortar el año anterior igual)
        fecha_corte = df_raw[df_raw['Año'] == anio_actual]['Fecha'].max()
        # Creamos una fecha límite para el año anterior (mismo día y mes)
        fecha_limite_anterior = fecha_corte.replace(year=anio_anterior)
        
        # Filtrar datos
        ventas_actual_ytd = df_raw[df_raw['Año'] == anio_actual]['Valor'].sum()
        ventas_anterior_ytd = df_raw[(df_raw['Año'] == anio_anterior) & (df_raw['Fecha'] <= fecha_limite_anterior)]['Valor'].sum()
        
        # Variación
        var_ytd = 0
        if ventas_anterior_ytd > 0:
            var_ytd = ((ventas_actual_ytd - ventas_anterior_ytd) / ventas_anterior_ytd) * 100
            
        ticket_prom = df_raw[df_raw['Año'] == anio_actual]['Valor'].mean()

        # TARJETAS KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Ventas {anio_actual} (YTD)", f"${ventas_actual_ytd:,.0f}", f"{var_ytd:+.1f}% vs año pasado")
        k2.metric("Ticket Promedio", f"${ticket_prom:,.0f}")
        k3.metric("Transacciones", f"{len(df_raw[df_raw['Año'] == anio_actual]):,}")
        k4.metric("Día Más Fuerte", df_raw[df_raw['Año'] == anio_actual]['Dia'].mode()[0])
        
        st.markdown("---")
        
        # --- SECCIÓN B: ANÁLISIS POR DÍA DE LA SEMANA (NUEVO) ---
        st.header("2. Patrones de Venta")
        
        c_dia, c_mes = st.columns(2)
        
        with c_dia:
            st.subheader("📅 Ventas por Día de Semana")
            # Agrupar por día numérico para ordenar Lunes-Domingo
            df_dias = df_raw[df_raw['Año'] == anio_actual].groupby(['DiaNum', 'Dia'])['Valor'].sum().reset_index()
            # Graficar
            fig_dias = graficar_barras_pro(df_dias, 'Dia', 'Valor', 'Acumulado por Día', '#9b59b6')
            st.pyplot(fig_dias)
            
        with c_mes:
            st.subheader(f"🗓️ Evolución Mensual {anio_actual}")
            df_mes = df_raw[df_raw['Año'] == anio_actual].groupby('MesNum')['Valor'].sum().reset_index()
            df_mes['Mes'] = df_mes['MesNum'].map(meses_es)
            
            # Tabla Estática (st.table muestra todo sin scroll)
            st.dataframe(
                df_mes[['Mes', 'Valor']].style.format({"Valor": "${:,.0f}"}),
                use_container_width=True,
                hide_index=True 
            )

        st.markdown("---")
        st.header("🏥 Radiografía por Clínica")
        st.info("Despliega para ver el detalle individual.")

        # --- SECCIÓN C: BUCLE POR CLÍNICA ---
        sucursales = sorted(df_raw['Sucursal'].unique())
        
        for suc in sucursales:
            with st.expander(f"📍 {suc}", expanded=False):
                df_suc = df_raw[(df_raw['Sucursal'] == suc) & (df_raw['Año'] == anio_actual)]
                
                if not df_suc.empty:
                    col_izq, col_der = st.columns(2)
                    
                    with col_izq:
                        st.markdown("**Ventas por Día de Semana**")
                        df_s_dia = df_suc.groupby(['DiaNum', 'Dia'])['Valor'].sum().reset_index()
                        if not df_s_dia.empty:
                            fig_sd = graficar_barras_pro(df_s_dia, 'Dia', 'Valor', '', '#e67e22')
                            st.pyplot(fig_sd)
                        
                    with col_der:
                        st.markdown("**Detalle Mensual**")
                        df_s_mes = df_suc.groupby('MesNum')['Valor'].sum().reset_index()
                        df_s_mes['Mes'] = df_s_mes['MesNum'].map(meses_es)
                        df_s_mes['Var %'] = df_s_mes['Valor'].pct_change() * 100
                        
                        # Tabla completa sin scroll (usamos dataframe pero configurado limpio)
                        st.dataframe(
                            df_s_mes[['Mes', 'Valor', 'Var %']].style
                            .format({"Valor": "${:,.0f}", "Var %": "{:+.1f}%"})
                            .applymap(color_negative_red, subset=['Var %']),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.warning("Sin datos este año.")

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
