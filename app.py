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

# --- ESTILOS DE TABLA (ROJO SI BAJA) ---
def color_negative_red(val):
    """
    Toma un escalar y devuelve un string con el estilo css.
    Rojo si es negativo, Negro (por defecto) si es positivo.
    """
    color = 'red' if isinstance(val, (int, float)) and val < 0 else 'black'
    return f'color: {color}'

# --- CARGA DE DATOS SQL ---
@st.cache_data(ttl=0)
def cargar_datos_sql():
    try:
        conn = st.connection("sql", type="sql")
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # Limpieza
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        # Columnas Auxiliares
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
        
        return df
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return pd.DataFrame()

# --- FUNCIÓN: GENERAR GRÁFICO DE BARRAS PRO ---
def graficar_barras(df_g, x_col, y_col, titulo, color_barras='skyblue'):
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(df_g[x_col], df_g[y_col], color=color_barras)
    
    # Etiquetas encima
    ax.bar_label(bars, fmt='${:,.0f}', padding=3, fontsize=9)
    
    # Formato Ejes
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# --- FUNCIÓN CEREBRO (CHAT) ---
def analizar_con_gpt(df, pregunta, api_key):
    # (Misma lógica de siempre para el chat)
    try:
        client = openai.OpenAI(api_key=api_key)
        buffer = io.StringIO()
        df.head(3).to_csv(buffer, index=False)
        muestra = buffer.getvalue()
        info_cols = df.dtypes.to_string()
        
        prompt_system = """
        Eres Gari. Usa 'df'. Reglas:
        1. Usa 'Fecha' (datetime). Ignora 'FechaCargue'.
        2. Output: 'resultado', 'tabla_resultados' (ordenada cronológicamente), 'fig' (con etiquetas).
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
pagina = st.sidebar.radio("Navegación", ["🧠 Chat con Gari", "📊 Reportes Ejecutivos (Automático)", "🗺️ Mapa"])

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
                    if res_tabla is not None: 
                        st.dataframe(res_tabla.style.format({"Ventas": "${:,.0f}"}))
                    if res_fig: st.pyplot(res_fig)
                    with st.expander("Ver código"): st.code(cod)
            else: st.error("Falta API Key")

# ==============================================================================
# PÁGINA 2: REPORTES EJECUTIVOS (ESTRUCTURA SOLICITADA)
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos (Automático)":
    st.title("📊 Informe Gerencial Automático")
    
    if not df_raw.empty:
        
        # 1. ANÁLISIS GLOBAL (EMPRESA COMPLETA)
        st.header("🏢 Visión Global de la Empresa")
        
        # A) COMPARATIVO POR AÑOS (KPIS)
        st.subheader("1. Comparativo Anual")
        
        df_anual = df_raw.groupby('Año')['Valor'].sum().reset_index().sort_values('Año')
        df_anual['Crecimiento %'] = df_anual['Valor'].pct_change() * 100
        
        # Tarjetas de KPI (Mostramos los últimos 3 años si hay)
        cols = st.columns(len(df_anual))
        for idx, row in df_anual.iterrows():
            year = int(row['Año'])
            val = row['Valor']
            delta = f"{row['Crecimiento %']:.1f}%" if pd.notnull(row['Crecimiento %']) else None
            cols[idx % len(cols)].metric(label=f"Ventas {year}", value=f"${val:,.0f}", delta=delta)
            
        # Gráfico Anual
        fig_anual = graficar_barras(df_anual, 'Año', 'Valor', 'Evolución Anual de Ventas', '#2c3e50')
        st.pyplot(fig_anual)
        
        # Tabla Anual (Debajo del gráfico)
        st.write("Detalle Anual:")
        st.dataframe(df_anual.style.format({"Valor": "${:,.0f}", "Crecimiento %": "{:.2f}%"})
                     .applymap(color_negative_red, subset=['Crecimiento %']), use_container_width=True)
        
        st.markdown("---")
        
        # B) COMPARATIVO MENSUAL (AÑO ACTUAL O SELECCIONADO)
        st.subheader("2. Comparativo Mes a Mes (Año Actual)")
        years = sorted(df_raw['Año'].unique().tolist(), reverse=True)
        year_sel = st.selectbox("Selecciona Año para ver el detalle mensual:", years)
        
        df_mensual = df_raw[df_raw['Año'] == year_sel].groupby('MesNum')['Valor'].sum().reset_index()
        df_mensual['Mes'] = df_mensual['MesNum'].map(meses_es)
        df_mensual['Variación Mensual %'] = df_mensual['Valor'].pct_change() * 100
        
        # Gráfico Mensual
        fig_mes = graficar_barras(df_mensual, 'Mes', 'Valor', f'Ventas Mensuales {year_sel}', '#27ae60')
        st.pyplot(fig_mes)
        
        # Tabla Mensual (Con Variación en ROJO si baja)
        st.write(f"Detalle Mensual {year_sel}:")
        st.dataframe(df_mensual[['Mes', 'Valor', 'Variación Mensual %']].style.format({"Valor": "${:,.0f}", "Variación Mensual %": "{:.2f}%"})
                     .applymap(color_negative_red, subset=['Variación Mensual %']), use_container_width=True)

        st.markdown("---")
        st.header("🏥 Análisis Detallado por Clínica")
        st.info("Despliega cada clínica para ver su rendimiento individual sin necesidad de filtros.")

        # 3. BUCLE POR CLÍNICA (AUTOMÁTICO)
        sucursales = sorted(df_raw['Sucursal'].unique())
        
        for suc in sucursales:
            with st.expander(f"📍 Clínica: {suc}", expanded=False):
                df_suc = df_raw[df_raw['Sucursal'] == suc]
                
                # --- A) POR AÑOS (SUCURSAL) ---
                st.markdown(f"**Comparativo Anual - {suc}**")
                df_suc_anual = df_suc.groupby('Año')['Valor'].sum().reset_index().sort_values('Año')
                df_suc_anual['Crecimiento %'] = df_suc_anual['Valor'].pct_change() * 100
                
                # KPIs Sucursal
                cols_s = st.columns(len(df_suc_anual))
                for idx, row in df_suc_anual.iterrows():
                    val = row['Valor']
                    delta = f"{row['Crecimiento %']:.1f}%" if pd.notnull(row['Crecimiento %']) else None
                    cols_s[idx % len(cols_s)].metric(f"Año {int(row['Año'])}", f"${val:,.0f}", delta)
                
                # Gráfico Sucursal Anual
                fig_sa = graficar_barras(df_suc_anual, 'Año', 'Valor', f'Evolución Anual - {suc}')
                st.pyplot(fig_sa)
                
                # Tabla Sucursal Anual
                st.dataframe(df_suc_anual.style.format({"Valor": "${:,.0f}", "Crecimiento %": "{:.2f}%"})
                             .applymap(color_negative_red, subset=['Crecimiento %']), use_container_width=True)
                
                st.divider()
                
                # --- B) POR MESES (SUCURSAL - Año seleccionado arriba) ---
                st.markdown(f"**Comparativo Mensual ({year_sel}) - {suc}**")
                
                df_suc_mes = df_suc[df_suc['Año'] == year_sel].groupby('MesNum')['Valor'].sum().reset_index()
                
                if not df_suc_mes.empty:
                    df_suc_mes['Mes'] = df_suc_mes['MesNum'].map(meses_es)
                    df_suc_mes['Variación %'] = df_suc_mes['Valor'].pct_change() * 100
                    
                    # Gráfico Sucursal Mensual
                    fig_sm = graficar_barras(df_suc_mes, 'Mes', 'Valor', f'Ventas Mensuales {year_sel} - {suc}', '#e67e22')
                    st.pyplot(fig_sm)
                    
                    # Tabla Sucursal Mensual (Con Rojo)
                    st.dataframe(df_suc_mes[['Mes', 'Valor', 'Variación %']].style.format({"Valor": "${:,.0f}", "Variación %": "{:.2f}%"})
                                 .applymap(color_negative_red, subset=['Variación %']), use_container_width=True)
                else:
                    st.warning(f"No hay ventas registradas en {suc} para el año {year_sel}.")

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
