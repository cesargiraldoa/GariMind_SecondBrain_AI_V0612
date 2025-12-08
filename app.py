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

# --- CARGA DE DATOS SQL (SIN CACHÉ) ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza y Formatos
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        # Columnas derivadas para Análisis BI
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        df['DiaSemana'] = df['Fecha'].dt.day_name() # Para ver qué días se vende más
        
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return pd.DataFrame()

# --- FUNCIÓN CEREBRO (CHAT) ---
def analizar_con_gpt(df, pregunta, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        prompt_system = """
        Eres Gari, experto Data Scientist.
        REGLAS:
        1. Usa 'df' y 'Fecha'. Ignora 'FechaCargue'.
        2. NO inventes datos.
        3. OUTPUTS: 'resultado' (str), 'tabla_resultados' (DataFrame ordenado cronológicamente), 'fig' (matplotlib con etiquetas).
        """
        prompt_user = f"Estructura: {info_cols}\nMuestra: {muestra}\nPregunta: {pregunta}\nGenera código Python."

        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}], temperature=0)
        codigo = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        
        local_vars = {'df': df, 'pd': pd, 'plt': plt, 'ticker': ticker, 'meses_es': meses_es}
        exec(codigo, globals(), local_vars)
        return (local_vars.get('resultado', None), local_vars.get('fig', None), local_vars.get('tabla_resultados', None), codigo)
    except Exception as e: return f"Error: {e}", None, None, ""

# --- FUNCIÓN PERIODISTA EJECUTIVO (BI) ---
def generar_reporte_ejecutivo(df_filtered, kpis, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt_system = """
        Eres un Consultor de Estrategia de Negocios Senior.
        Escribe un análisis ejecutivo (tipo periódico financiero) basado en los datos proporcionados.
        
        ESTRUCTURA DEL REPORTE:
        1. 📢 Titular de Impacto.
        2. 💰 Análisis Financiero: Interpreta el Ticket Promedio y la Venta Total.
        3. 🏢 Comportamiento de Sucursales: Quién lidera y quién preocupa.
        4. 💳 Tendencias de Pago/Convenio: Insights sobre cómo pagan los pacientes.
        5. 🚀 Recomendación Estratégica: Una acción concreta basada en los datos.
        """
        
        prompt_user = f"""
        DATOS DEL PERIODO SELECCIONADO:
        - Ventas Totales: ${kpis['venta_total']:,.0f}
        - Ticket Promedio por Paciente: ${kpis['ticket_promedio']:,.0f}
        - Top Sucursal: {kpis['top_sucursal']} (${kpis['venta_top_sucursal']:,.0f})
        - Mix de Medios de Pago (Top 3): {kpis['top_medios_pago']}
        - Top Convenios: {kpis['top_convenios']}
        
        Analiza esto y dame el reporte.
        """
        
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}])
        return response.choices[0].message.content
    except Exception as e: return f"Error generando reporte: {e}"

# --- INTERFAZ ---

st.sidebar.image("https://img.freepik.com/premium-photo/cute-hamster-face-portrait_1029469-218417.jpg", width=120, caption="Gari 🐹")
pagina = st.sidebar.radio("Navegación", ["🧠 Chat con Gari", "📊 Reportes Ejecutivos BI", "🗺️ Mapa"])

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 API Key OpenAI:", type="password")

with st.spinner("Cargando datos..."):
    df_raw = cargar_datos_sql()

# ==============================================================================
# 🧠 PÁGINA 1: CHAT (Igual que antes)
# ==============================================================================
if pagina == "🧠 Chat con Gari":
    st.title("Hola soy Gari tu segundo cerebro extendido")
    st.write("### Haz preguntas puntuales a tus datos")
    
    if not df_raw.empty:
        st.caption(f"Datos hasta: {df_raw['Fecha'].max().strftime('%d/%m/%Y')}")
        pregunta = st.text_input("Consulta:", "Cual fue el mes de mayor venta en el año 2025?")
        
        if st.button("Analizar"):
            if api_key:
                with st.spinner("Analizando..."):
                    res_txt, res_fig, res_tabla, cod = analizar_con_gpt(df_raw, pregunta, api_key)
                    st.divider()
                    if res_txt: st.success(f"📌 {res_txt}")
                    else: st.warning("Sin datos.")
                    if res_tabla is not None: 
                        st.write("### 📅 Detalle")
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}), use_container_width=True)
                    if res_fig: st.pyplot(res_fig)
                    with st.expander("Ver código"): st.code(cod)
            else: st.error("Falta API Key")

# ==============================================================================
# 📊 PÁGINA 2: REPORTES EJECUTIVOS BI (LA MEJORA)
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos BI":
    st.title("📊 Tablero Estratégico de Negocio")
    
    if not df_raw.empty:
        # --- FILTROS ---
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Filtros de Tablero")
        
        # Filtro Año
        years = sorted(df_raw['Año'].unique().tolist(), reverse=True)
        year_sel = st.sidebar.multiselect("Año", years, default=years[:1])
        
        # Filtro Mes
        all_months = list(meses_es.values())
        month_sel = st.sidebar.multiselect("Mes", all_months, default=[]) # Si está vacío, son todos
        
        # Aplicar Filtros
        df_f = df_raw.copy()
        if year_sel: df_f = df_f[df_f['Año'].isin(year_sel)]
        if month_sel: df_f = df_f[df_f['Mes'].isin(month_sel)]
        
        # --- CÁLCULO DE KPIS (Python Puro - Rápido y Seguro) ---
        venta_total = df_f['Valor'].sum()
        transacciones = len(df_f)
        ticket_promedio = venta_total / transacciones if transacciones > 0 else 0
        
        top_sucursal_nombre = "N/A"
        top_sucursal_valor = 0
        if not df_f.empty:
            sucursal_stats = df_f.groupby('Sucursal')['Valor'].sum().sort_values(ascending=False)
            top_sucursal_nombre = sucursal_stats.index[0]
            top_sucursal_valor = sucursal_stats.values[0]

        # --- VISUALIZACIÓN DE TARJETAS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Ventas Totales", f"${venta_total:,.0f}")
        col2.metric("🧾 Transacciones", f"{transacciones:,}")
        col3.metric("🎫 Ticket Promedio", f"${ticket_promedio:,.0f}")
        col4.metric("🏆 Top Sucursal", top_sucursal_nombre)
        
        st.markdown("---")
        
        # --- GRÁFICOS DE NEGOCIO ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📈 Tendencia de Ventas (Evolución)")
            # Agrupamos por fecha para ver línea de tiempo
            if not df_f.empty:
                trend = df_f.groupby('Fecha')['Valor'].sum().reset_index()
                st.line_chart(trend, x='Fecha', y='Valor', color='#3498db')
            else:
                st.info("Sin datos para graficar.")
                
        with c2:
            st.subheader("💳 Mix de Medios de Pago")
            if not df_f.empty:
                medios = df_f.groupby('Forma_de_Pago')['Valor'].sum().sort_values(ascending=False)
                st.bar_chart(medios, horizontal=True, color='#2ecc71')

        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("🏥 Top Convenios/Aseguradoras")
            if not df_f.empty:
                convenios = df_f.groupby('Convenio_Paciente')['Valor'].sum().sort_values(ascending=False).head(5)
                st.bar_chart(convenios, color='#9b59b6')
        
        with c4:
             st.subheader("🏢 Ranking de Sucursales")
             if not df_f.empty:
                 sucursales = df_f.groupby('Sucursal')['Valor'].sum().sort_values(ascending=False)
                 st.dataframe(sucursales.to_frame().style.format("${:,.0f}"), use_container_width=True)

        # --- SECCIÓN: ANÁLISIS IA ---
        st.markdown("---")
        st.header("🧠 Análisis Estratégico (Periodista IA)")
        st.write("Gari analiza todos los KPIs anteriores y redacta un informe ejecutivo.")
        
        if st.button("✨ Redactar Informe Ejecutivo"):
            if api_key and not df_f.empty:
                with st.spinner("Leyendo los KPIs y redactando..."):
                    # Preparamos el resumen para la IA
                    kpis_para_ia = {
                        'venta_total': venta_total,
                        'ticket_promedio': ticket_promedio,
                        'top_sucursal': top_sucursal_nombre,
                        'venta_top_sucursal': top_sucursal_valor,
                        'top_medios_pago': df_f.groupby('Forma_de_Pago')['Valor'].sum().nlargest(3).to_dict(),
                        'top_convenios': df_f.groupby('Convenio_Paciente')['Valor'].sum().nlargest(3).to_dict()
                    }
                    
                    informe = generar_reporte_ejecutivo(df_f, kpis_para_ia, api_key)
                    st.markdown(informe)
            else:
                st.error("Falta API Key o no hay datos filtrados.")

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
