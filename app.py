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

# --- ESTILOS CSS PARA TABLAS ---
def color_negative_red(val):
    """Pinta de rojo los negativos en las tablas"""
    if isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4b4b; font-weight: bold'
    return 'color: black'

# --- FUNCIÓN GRÁFICA PRO (ESTILO SEMÁFORO) ---
def graficar_barras_pro(df_g, x_col, y_col, titulo, tipo_dato='dinero'):
    """
    Genera gráficos estéticos.
    - tipo_dato: 'dinero' (azul corporativo) o 'variacion' (verde/rojo)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 1. Definir Colores (Lógica Semáforo)
    if tipo_dato == 'variacion':
        colores = ['#2ecc71' if x >= 0 else '#ff4b4b' for x in df_g[y_col]] # Verde / Rojo
    else:
        colores = '#3498db' # Azul Corporativo
        
    # 2. Crear Barras
    bars = ax.bar(df_g[x_col], df_g[y_col], color=colores, edgecolor='none', alpha=0.9)
    
    # 3. Etiquetas de Datos (Encima de la barra)
    if tipo_dato == 'dinero':
        fmt = '${:,.0f}' 
    else:
        fmt = '{:+.1f}%' # Con signo + o -
        
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=10, fontweight='bold', color='#2c3e50')
    
    # 4. Limpieza Visual (Minimalismo)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Quitamos eje Y izquierdo para limpiar
    ax.get_yaxis().set_visible(False)    # Ocultamos números del eje Y (ya están las etiquetas)
    
    # 5. Títulos y Ejes
    ax.set_title(titulo, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
    plt.xticks(rotation=0, fontsize=10, color='#2c3e50') # Texto horizontal si cabe
    
    # Línea base en 0 para variaciones
    if tipo_dato == 'variacion':
        ax.axhline(0, color='grey', linewidth=0.8)
        
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
        df['Año'] = df['Fecha'].dt.year
        df['MesNum'] = df['Fecha'].dt.month
        df['Mes'] = df['MesNum'].map(meses_es)
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
        Eres Gari. Usa 'df' y 'Fecha'. 
        Outputs: 'resultado', 'tabla_resultados' (ordenada cronológicamente), 'fig' (con etiquetas).
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
pagina = st.sidebar.radio("Navegación", ["🧠 Chat con Gari", "📊 Reportes Ejecutivos (Visual Pro)", "🗺️ Mapa"])

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
# PÁGINA 2: REPORTES EJECUTIVOS PRO
# ==============================================================================
elif pagina == "📊 Reportes Ejecutivos (Visual Pro)":
    st.title("📊 Informe Gerencial de Alto Impacto")
    
    if not df_raw.empty:
        
        st.header("🏢 Panorama Global")
        
        # --- A) COMPARATIVO ANUAL ---
        st.subheader("1. Evolución Anual")
        df_anual = df_raw.groupby('Año')['Valor'].sum().reset_index().sort_values('Año')
        df_anual['Crecimiento %'] = df_anual['Valor'].pct_change() * 100
        
        # Gráfico Anual (Dinero)
        fig_anual = graficar_barras_pro(df_anual, 'Año', 'Valor', 'Ventas Totales por Año', tipo_dato='dinero')
        st.pyplot(fig_anual)
        
        # Tabla Anual
        st.dataframe(df_anual.style.format({"Valor": "${:,.0f}", "Crecimiento %": "{:+.2f}%"})
                     .applymap(color_negative_red, subset=['Crecimiento %']), use_container_width=True)
        
        st.markdown("---")
        
        # --- B) COMPARATIVO MENSUAL ---
        st.subheader("2. Evolución Mensual")
        years = sorted(df_raw['Año'].unique().tolist(), reverse=True)
        year_sel = st.selectbox("📅 Selecciona Año:", years)
        
        df_mensual = df_raw[df_raw['Año'] == year_sel].groupby('MesNum')['Valor'].sum().reset_index()
        df_mensual['Mes'] = df_mensual['MesNum'].map(meses_es)
        df_mensual['Variación %'] = df_mensual['Valor'].pct_change() * 100
        
        # Gráfico Mensual (Variación - SEMÁFORO)
        # Aquí graficamos la variación para ver barras verdes/rojas
        # O graficamos ventas si prefieres. Hagamos ventas para consistencia, 
        # y variación en tabla.
        fig_mes = graficar_barras_pro(df_mensual, 'Mes', 'Valor', f'Ventas Mensuales {year_sel}', tipo_dato='dinero')
        st.pyplot(fig_mes)
        
        # Tabla Mensual
        st.dataframe(df_mensual[['Mes', 'Valor', 'Variación %']].style.format({"Valor": "${:,.0f}", "Variación %": "{:+.2f}%"})
                     .applymap(color_negative_red, subset=['Variación %']), use_container_width=True)

        st.markdown("---")
        st.header("🏥 Análisis por Clínica (Smart View)")
        st.info("Despliega para ver el detalle de cada sede.")

        # --- C) BUCLE POR CLÍNICA ---
        sucursales = sorted(df_raw['Sucursal'].unique())
        
        for suc in sucursales:
            with st.expander(f"📍 {suc}", expanded=False):
                df_suc = df_raw[df_raw['Sucursal'] == suc]
                
                col1, col2 = st.columns(2)
                
                # Columna 1: Anual
                with col1:
                    st.markdown("##### 📅 Anual")
                    df_s_a = df_suc.groupby('Año')['Valor'].sum().reset_index()
                    df_s_a['Var %'] = df_s_a['Valor'].pct_change() * 100
                    
                    # Gráfico mini (Solo si hay datos suficientes)
                    if len(df_s_a) > 0:
                        fig_sa = graficar_barras_pro(df_s_a, 'Año', 'Valor', '', tipo_dato='dinero')
                        st.pyplot(fig_sa)
                    
                    st.dataframe(df_s_a.style.format({"Valor": "${:,.0f}", "Var %": "{:+.1f}%"})
                                 .applymap(color_negative_red, subset=['Var %']), use_container_width=True)

                # Columna 2: Mensual (Año seleccionado)
                with col2:
                    st.markdown(f"##### 🗓️ Mensual ({year_sel})")
                    df_s_m = df_suc[df_suc['Año'] == year_sel].groupby('MesNum')['Valor'].sum().reset_index()
                    
                    if not df_s_m.empty:
                        df_s_m['Mes'] = df_s_m['MesNum'].map(meses_es)
                        df_s_m['Var %'] = df_s_m['Valor'].pct_change() * 100
                        
                        # Gráfico Semáforo de Variación (Opcional, aquí dejo ventas)
                        fig_sm = graficar_barras_pro(df_s_m, 'Mes', 'Valor', '', tipo_dato='dinero')
                        st.pyplot(fig_sm)
                        
                        st.dataframe(df_s_m[['Mes', 'Valor', 'Var %']].style.format({"Valor": "${:,.0f}", "Var %": "{:+.1f}%"})
                                     .applymap(color_negative_red, subset=['Var %']), use_container_width=True)
                    else:
                        st.warning("Sin ventas este año.")

# ==============================================================================
# PÁGINA 3: MAPA
# ==============================================================================
elif pagina == "🗺️ Mapa":
    st.title("Mapa SQL")
    try:
        conn = st.connection("sql", type="sql")
        st.dataframe(conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    except: st.error("Error SQL")
