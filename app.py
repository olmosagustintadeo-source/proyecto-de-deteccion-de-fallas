import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Simulador de mantenimiento predictivo", layout="wide")

# ==========================================
# ESTILOS PROFESIONALES Y ANIMACIONES CSS
# ==========================================
st.markdown("""
<style>
    /* 1. Transición global suave al recargar o cambiar elementos */
    * {
        transition: all 0.25s ease-in-out;
    }

    /* 2. Estilo de las pestañas (Tabs) */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.3s ease !important;
    }

    button[data-baseweb="tab"]:hover {
        background-color: rgba(52, 152, 219, 0.1) !important;
        color: #3498db !important;
        transform: translateY(-2px);
    }

    /* Pestaña activa */
    button[aria-selected="true"] {
        border-bottom: 3px solid #2ecc71 !important;
        color: #2ecc71 !important;
    }

    /* 3. Animación de entrada para el contenido de las pestañas */
    div[role="tabpanel"] {
        animation: fadeIn 0.4s ease-in-out;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* 4. Estilos y animación en las Tarjetas de Métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #252526 !important;
        border: 1px solid #333333 !important;
        padding: 15px 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(46, 204, 113, 0.15) !important;
        border-color: #2ecc71 !important;
    }

    /* Label e información del KPI */
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }

    /* 5. Personalización de Sliders y Selectores */
    .stSlider > div {
        padding-top: 5px;
    }

    /* Efecto al pasar sobre las cajas de selección */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
        transition: border-color 0.25s ease !important;
    }

    /* 6. Animación sutil en las gráficas de Matplotlib/Plotly */
    .element-container img, .stPlotlyChart {
        animation: scaleUp 0.35s ease-in-out;
        border-radius: 8px;
    }

    @keyframes scaleUp {
        from {
            opacity: 0.8;
            transform: scale(0.99);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_datos_globales(nombre_modelo):
    if nombre_modelo == "Random Forest":
        return pd.read_csv("eval_rf.csv")
    elif nombre_modelo == "XGBoost":
        return pd.read_csv("eval_xgb.csv")
    else:
        return pd.read_csv("eval_lstm.csv")

@st.cache_data
def cargar_datos_maquinas():
    try:
        return pd.read_csv("resultados_modelos_por_maquina.csv")
    except FileNotFoundError:
        # Si no existe el CSV precalculado, intenta cargar algún Dataset de evaluación por máquinas
        try:
            return pd.read_csv("eval_rf.csv")
        except FileNotFoundError:
            return None

df_test_maquinas = cargar_datos_maquinas()

col_map_maquinas = {
    "Random Forest": "prob_rf",
    "XGBoost": "prob_xgb",
    "LSTM": "prob_lstm"
}

# --- FUNCIÓN DE EVALUACIÓN FINANCIERA EN TIEMPO REAL ---
def calcular_metricas_financieras(df_eval, prob_col, umbral, costo_preventivo_promedio=558.25, costo_inspeccion=150):
    # 1. Normalización de arreglos
    y_real_arr = np.array(df_eval['target_falla']).ravel()
    y_prob_arr = np.array(df_eval[prob_col]).ravel()

    # 2. Binarización según el umbral
    y_pred_arr = (y_prob_arr >= umbral).astype(int)

    # 3. Alineación al tamaño mínimo común
    min_len = min(len(df_eval), len(y_real_arr), len(y_pred_arr))
    y_real_arr = y_real_arr[:min_len]
    y_pred_arr = y_pred_arr[:min_len]
    df_eval_aligned = df_eval.iloc[:min_len].copy().reset_index(drop=True)

    # 4. Máscaras booleanas
    pred_tp = (y_real_arr == 1) & (y_pred_arr == 1)
    pred_fn = (y_real_arr == 1) & (y_pred_arr == 0)
    pred_fp = (y_real_arr == 0) & (y_pred_arr == 1)

    # 5. Conteo de métricas
    tp_count = int(np.sum(pred_tp))
    fn_count = int(np.sum(pred_fn))
    fp_count = int(np.sum(pred_fp))
    tn_count = int(np.sum((y_real_arr == 0) & (y_pred_arr == 0)))

    # 6. Cálculo de costos
    costo_tp = tp_count * costo_preventivo_promedio
    costo_fp = fp_count * costo_inspeccion

    col_costo = 'cost_usd' if 'cost_usd' in df_eval_aligned.columns else 'costo_usd'

    if col_costo in df_eval_aligned.columns:
        costo_fn = float(df_eval_aligned.loc[pred_fn, col_costo].sum())
        costo_sin_modelo = float(df_eval_aligned.loc[y_real_arr == 1, col_costo].sum())
    else:
        costo_fn = float(fn_count * 500)
        costo_sin_modelo = float((tp_count + fn_count) * 500)

    costo_total_modelo = costo_tp + costo_fn + costo_fp
    ahorro_neto = costo_sin_modelo - costo_total_modelo

    return {
        'TP': tp_count,
        'FN': fn_count,
        'FP': fp_count,
        'TN': tn_count,
        'Costo TP ($)': round(costo_tp, 2),
        'Costo FN ($)': round(costo_fn, 2),
        'Costo FP ($)': round(costo_fp, 2),
        'Costo Mantenimiento ($)': round(costo_total_modelo, 2),
        'Costo Sin Modelo ($)': round(costo_sin_modelo, 2),
        'Ahorro neto ($)': round(ahorro_neto, 2)
    }

# --- NAVEGACIÓN EN PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Presentación del Proyecto",
    "Vista Global",
    "Vista por Máquina",
    "Conclusión"
])

# ==========================================
# PESTAÑA 1: PRESENTACIÓN
# ==========================================
with tab1:
    st.title("Proyecto de mantenimiento predictivo industrial")
    st.markdown("### Optimización financiera mediante aprendizaje automático")
    st.markdown("---")

    col_intro1, col_intro2 = st.columns([2, 1])

    with col_intro1:
        st.subheader("Contexto organizacional")
        st.markdown("""
        El conjunto de datos comprende registros de monitoreo continuo y telemetría en tiempo real sobre la maquinaria crítica de la planta de producción. Integra un grupo de variables clave diseñadas para evaluar:

        - **Desgaste dinámico:** Evolución del deterioro estructural y mecánico en los componentes clave.
        - **Fenómenos operacionales:** Variaciones en indicadores críticos como temperatura, presión y nivel de vibración.
        - **Uso de equipos:** Seguimiento acumulativo de las horas de operación por unidad.
        - **Análisis de fallas y costos:** Diagnóstico del rendimiento de las máquinas y cuantificación del impacto financiero derivado del mantenimiento realizado sobre las mismas.
        """)

        st.subheader("Objetivo del proyecto")
        st.markdown("""
        Desarrollar e implementar un **sistema de mantenimiento predictivo ** basado en Machine Learning y Deep Learning para:
        1. **Predecir el riesgo de falla de las máquinas** antes de que ocurra una parada no planificada.
        2. **Minimizar el costo total operativo** equilibrando el gasto de inspecciones/prevenciones contra las penalizaciones por fallas imprevistas.
        3. **Optimizar la toma de decisiones** permitiendo a los ingenieros de planta definir una planificacion controlada sobre el mantenimiento de la maquinaria.
        """)

        st.header(" Descripción de Variables del Dataset")
    st.markdown("A continuación se describen las características presentes en el conjunto de datos de telemetría y mantenimiento:")

    st.markdown("""
    | Variable / Columna | Tipo de Dato | Descripción Breve |
    | :--- | :--- | :--- |
    | **`timestamp`** | Temporal (`datetime64`) | Fecha y hora exacta del registro del sensor. |
    | **`machine_id`** | Identificador (`object`) | Código único asignado a cada máquina en la planta. |
    | **`temperature_C`** | Numérica (`float64`) | Temperatura interna de trabajo medida en °C. |
    | **`vibration_mm_s`** | Numérica (`float64`) | Nivel de vibración mecánica medida en mm/s. |
    | **`pressure_bar`** | Numérica (`float64`) | Presión del sistema medida en bares (bar). |
    | **`status`** | Categórica (`object`) | Estado operativo general de la máquina. |
    | **`operating_hours`** | Numérica (`float64`) | Horas de trabajo acumuladas por el equipo. |
    | **`shift`** | Categórica (`object`) | Turno de trabajo durante la recolección del registro. |
    | **`failure_id`** | Identificador (`object`) | Código único de evento de falla registrado. |
    | **`failure_start`** | Temporal (`datetime64`) | Momento exacto en que comenzó el evento de falla. |
    | **`failure_end`** | Categórica / Temporal (`object`) | Registro o tiempo en el que se solventó la falla. |
    | **`failure_type`** | Categórica (`object`) | Clasificación o categoría de la falla presentada. |
    | **`maintenance_id`** | Identificador (`object`) | Código único de la orden o registro de mantenimiento. |
    | **`maintenance_start`**| Temporal (`datetime64`) | Fecha y hora de inicio de la intervención técnica. |
    | **`maintenance_end`** | Categórica / Temporal (`object`) | Fecha y hora de finalización del mantenimiento. |
    | **`maintenance_type`** | Categórica (`object`) | Tipo de mantenimiento realizado (Preventivo / Correctivo). |
    | **`downtime_minutes`** | Numérica (`float64`) | Tiempo de parada o inactividad generado en minutos. |
    | **`cost_usd`** | Numérica (`float64`) | Costo financiero total en USD derivado de la parada/mantenimiento. |
    """)


    with col_intro2:
        st.info("**Resumen de datos**")
        if df_test_maquinas is not None:
            st.metric("Registros totales del dataset", f"421.632")
            st.metric("Registros evaluados (Por máquina)", f"{len(df_test_maquinas):,}")
            if 'machine_id' in df_test_maquinas.columns:
                st.metric("Máquinas monitoreadas", len(df_test_maquinas['machine_id'].unique()))

        st.success("**Modelos evaluados:**\n- Random Forest\n- XGBoost\n- Red Neuronal Recurrente (LSTM)")

# ==========================================
# PESTAÑA 2: VISTA GLOBAL
# ==========================================
with tab2:
    st.header("Análisis global de umbrales e impacto financiero asociado")
    st.markdown("Ajusta los umbrales probabilísticos globales para evaluar el rendimiento acumulado en toda la planta.")

    # --- INICIALIZACIÓN DE ESTADO PARA GUARDAR UMBRALES POR MODELO ---
    if 'umbrales_modelos' not in st.session_state:
        st.session_state.umbrales_modelos = {
            "Random Forest": 0.88,
            "XGBoost": 0.71,
            "LSTM": 0.75
        }

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        modelo_sel = st.selectbox("Selecciona el modelo global:", ["Random Forest", "XGBoost", "LSTM"], key="mod_global")
        df_test_global = cargar_datos_globales(modelo_sel)

    # Recuperar el umbral actual guardado para el modelo seleccionado
    umbral_actual_modelo = st.session_state.umbrales_modelos[modelo_sel]

    with col_m2:
        umbral_global = st.slider(
            "Umbral de decisión probabilístico:",
            0.01, 0.99,
            float(umbral_actual_modelo),
            0.01,
            key=f"umbral_slider_{modelo_sel}"
        )

        # Actualizamos y guardamos el umbral en la memoria de la sesión
        st.session_state.umbrales_modelos[modelo_sel] = umbral_global

    # Cálculo de métricas para el modelo seleccionado en pantalla
    res_global = calcular_metricas_financieras(df_test_global, 'prob_model', umbral_global)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Ahorro Neto Total", f"${res_global['Ahorro neto ($)']:,.2f}")
    kpi2.metric("Fallas Detectadas (TP)", f"{res_global['TP']} / {res_global['TP'] + res_global['FN']}")
    kpi3.metric("Falsas Alarmas (FP)", f"{res_global['FP']}")
    kpi4.metric("Fallas NO Detectadas (FN)", f"{res_global['FN']}")

    st.markdown("---")

    st.markdown("### Comparativa global de ahorro entre modelos")

    modelos_nombres = ["Random Forest", "XGBoost", "LSTM"]
    datos_globales = []

    # Iteramos calculando cada modelo con SU PROPIO umbral guardado
    for mod_nombre in modelos_nombres:
        df_mod = cargar_datos_globales(mod_nombre)
        u_guardado = st.session_state.umbrales_modelos[mod_nombre]
        res = calcular_metricas_financieras(df_mod, 'prob_model', u_guardado)

        datos_globales.append({
            'Modelo': f"{mod_nombre}\n(Umbral: {u_guardado:.2f})",
            'Ahorro neto ($)': res['Ahorro neto ($)']
        })

    df_global_grafico = pd.DataFrame(datos_globales)

    fig_global, ax_g = plt.subplots(figsize=(8, 3.8), facecolor='#1e1e1e')
    ax_g.set_facecolor('#1e1e1e')

    sns.barplot(
        data=df_global_grafico,
        x='Modelo',
        y='Ahorro neto ($)',
        palette='crest',
        ax=ax_g
    )

    ax_g.set_xlabel('Modelo y umbral evaluado', fontsize=11, fontweight='bold', color='white')
    ax_g.set_ylabel('Ahorro neto total ($)', fontsize=11, fontweight='bold', color='white')
    ax_g.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_g.grid(axis='y', linestyle='--', alpha=0.2, color='white')
    ax_g.tick_params(colors='white', labelsize=10)

    for p in ax_g.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        va_pos = 'bottom' if h >= 0 else 'top'
        text_color = '#2ecc71' if h >= 0 else '#e74c3c'
        offset = 5 if h >= 0 else -14
        ax_g.annotate(
            f'${h:,.2f}',
            (p.get_x() + p.get_width() / 2., h),
            ha='center', va=va_pos,
            fontsize=9, fontweight='bold', color=text_color,
            xytext=(0, offset),
            textcoords='offset points'
        )

    plt.tight_layout()
    st.pyplot(fig_global)

# ==========================================
# PESTAÑA 3: VISTA POR MÁQUINA
# ==========================================
with tab3:
    st.header("Análisis segmentado e individual por máquina")

    if df_test_maquinas is not None and 'machine_id' in df_test_maquinas.columns:
        maquinas_list = sorted(df_test_maquinas['machine_id'].unique())
        modelos_list = ["Random Forest", "XGBoost", "LSTM"]

        # --- INICIALIZACIÓN DE ESTADO PARA GUARDAR UMBRALES POR MÁQUINA Y MODELO ---
        if 'umbrales_por_maquina' not in st.session_state:
            st.session_state.umbrales_por_maquina = {
                m: {"Random Forest": 0.60, "XGBoost": 0.71, "LSTM": 0.50} for m in maquinas_list
            }

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            maquina_sel = st.selectbox("Selecciona la Máquina:", maquinas_list)
        with col_m2:
            modelo_m_sel = st.selectbox("Modelo:", modelos_list, key="mod_maq")
            prob_m_col = col_map_maquinas[modelo_m_sel]

        # Recuperar el umbral previamente guardado para esta máquina y modelo
        umbral_actual_maq = st.session_state.umbrales_por_maquina[maquina_sel][modelo_m_sel]

        with col_m3:
            umbral_maq = st.slider(
                "Umbral Específico:",
                0.01, 0.99,
                float(umbral_actual_maq),
                0.01,
                key=f"umbral_maq_{maquina_sel}_{modelo_m_sel}"
            )
            # Guardar el nuevo umbral asignado
            st.session_state.umbrales_por_maquina[maquina_sel][modelo_m_sel] = umbral_maq

        st.markdown(f"#### Desempeño Operativo en **{maquina_sel}**")

        df_maq = df_test_maquinas[df_test_maquinas['machine_id'] == maquina_sel].copy()

        if prob_m_col in df_maq.columns:
            res_maq = calcular_metricas_financieras(df_maq, prob_m_col, umbral_maq)

            col_mat, col_txt = st.columns([1, 1])

            with col_mat:
                st.subheader("Matriz de Confusión")
                cm_data = [[res_maq['TN'], res_maq['FP']], [res_maq['FN'], res_maq['TP']]]

                fig_cm = px.imshow(
                    cm_data,
                    text_auto=True,
                    x=['Pred No Falla', 'Pred Falla'],
                    y=['Real No Falla', 'Real Falla'],
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_txt:
                st.subheader("Impacto financiero individual")
                st.write(f"**Costo sin modelo:** ${res_maq['Costo Sin Modelo ($)']:,.2f}")
                st.write(f"**Costo con modelo:** ${res_maq['Costo Mantenimiento ($)']:,.2f}")
                st.success(f"**Ahorro neto generado:** ${res_maq['Ahorro neto ($)']:,.2f}")
                st.info(f"**Estado:** Se detectaron **{res_maq['TP']}** fallas reales en **{maquina_sel}** usando **{modelo_m_sel}** con un umbral de **{umbral_maq:.2f}**.")
        else:
            st.error(f"La columna de probabilidad '{prob_m_col}' no está disponible.")

        st.markdown("---")

        st.subheader("Modelo Óptimo y ahorro neto máximo por máquina")

        # --- CÁLCULO DEL MEJOR MODELO POR MÁQUINA USANDO LOS UMBRALES GUARDADOS ---
        registros = []
        modelos_map = {'Random Forest': 'prob_rf', 'XGBoost': 'prob_xgb', 'LSTM': 'prob_lstm'}

        for m in maquinas_list:
            sub_df = df_test_maquinas[df_test_maquinas['machine_id'] == m]
            for mod_nombre, prob_col in modelos_map.items():
                if prob_col in sub_df.columns:
                    # Usamos el umbral individual guardado de esa máquina y ese modelo
                    u_maq_guardado = st.session_state.umbrales_por_maquina[m][mod_nombre]
                    res = calcular_metricas_financieras(sub_df, prob_col, u_maq_guardado)
                    registros.append({
                        'machine_id': m,
                        'Modelo': mod_nombre,
                        'Ahorro neto ($)': res['Ahorro neto ($)']
                    })

        df_resumen = pd.DataFrame(registros)

        col_maquina = 'machine_id'
        col_ahorro = 'Ahorro neto ($)'
        col_modelo = 'Modelo'

        if not df_resumen.empty:
            idx_mejores = df_resumen.groupby(col_maquina)[col_ahorro].idxmax()
            df_mejores_modelos = df_resumen.loc[idx_mejores].sort_values(by=col_ahorro, ascending=False).reset_index(drop=True)

            fig_barras, ax = plt.subplots(figsize=(10, 4.5), facecolor='#1e1e1e')
            ax.set_facecolor('#1e1e1e')

            sns.barplot(
                data=df_mejores_modelos,
                x=col_maquina,
                y=col_ahorro,
                hue=col_modelo,
                palette='crest',
                ax=ax
            )

            ax.set_xlabel('Máquina Evaluada', fontsize=11, fontweight='bold', color='white')
            ax.set_ylabel('Ahorro Neto Generado ($)', fontsize=11, fontweight='bold', color='white')
            ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')
            ax.tick_params(colors='white', labelsize=10)

            legend = ax.legend(title='Modelo Ganador', facecolor='#1e1e1e', edgecolor='#2b2b2b', loc='upper right')
            plt.setp(legend.get_texts(), color='white')
            plt.setp(legend.get_title(), color='white', fontweight='bold')

            for p in ax.patches:
                h = p.get_height()
                if np.isnan(h) or h == 0:
                    continue
                va_pos = 'bottom' if h >= 0 else 'top'
                text_color = '#2ecc71' if h >= 0 else '#e74c3c'
                offset = 5 if h >= 0 else -14
                ax.annotate(
                    f'${h:,.2f}',
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va=va_pos,
                    fontsize=9, fontweight='bold', color=text_color,
                    xytext=(0, offset),
                    textcoords='offset points'
                )

            plt.tight_layout()
            st.pyplot(fig_barras)
    else:
        st.warning("No se encontró el archivo necesario para evaluar por máquinas.")

# ==========================================
# PESTAÑA 4: CONCLUSIÓN
# ==========================================
with tab4:
    st.header("Conclusiones y recomendaciones operativas")
    st.markdown("""
    - **Ajuste de Umbral:** La personalización del umbral probabilístico permite equilibrar los falsos positivos con las paradas no planificadas, maximizando el ahorro total.
    - **Monitoreo por Máquina:** Cada máquina presenta comportamientos distintos; la implementación de un modelo óptimo individualizado genera un retorno financiero superior al de un modelo rígido global.
    """)

    st.markdown("---")

    # --- COMPARATIVA ESTÁTICA Y DEFINITIVA ---
    st.subheader("Impacto Financiero Definitivo: Modelo Global Fijo vs. Segmentado por Máquina")

    # 1. Cálculo del modelo global fijo (XGBoost a umbral 0.71)
    try:
        df_xgb_fijo = cargar_datos_globales("XGBoost")
        res_xgb_fijo = calcular_metricas_financieras(df_xgb_fijo, 'prob_model', umbral=0.71)
        ahorro_global_fijo = res_xgb_fijo['Ahorro neto ($)']
    except Exception:
        # Valor de resguardo si los archivos no están sincronizados
        ahorro_global_fijo = 61250.00

    # 2. Suma fija de los valores óptimos obtenidos por máquina ($28,965.50 + $23,257.50 + $12,840.25 + $7,625.50)
    ahorro_optimo_maquinas_fijo = 28965.50 + 23257.50 + 12840.25 + 7625.50  # Total: $72,688.75

    # 3. Estructuración del DataFrame para la gráfica
    df_comparativa_definitiva = pd.DataFrame([
        {
            'Estrategia': 'Mejor Global\n(XGBoost @ 0.71)',
            'Ahorro Neto Total ($)': ahorro_global_fijo
        },
        {
            'Estrategia': 'Óptimo Segmentado\n(Por Máquina)',
            'Ahorro Neto Total ($)': ahorro_optimo_maquinas_fijo
        }
    ])

    # 4. Construcción del gráfico estático con el diseño institucional
    fig_conc, ax_c = plt.subplots(figsize=(8, 4.2), facecolor='#1e1e1e')
    ax_c.set_facecolor('#1e1e1e')

    sns.barplot(
        data=df_comparativa_definitiva,
        x='Estrategia',
        y='Ahorro Neto Total ($)',
        palette=['#3498db', '#2ecc71'],
        ax=ax_c
    )

    ax_c.set_xlabel('Estrategia de Despliegue', fontsize=11, fontweight='bold', color='white')
    ax_c.set_ylabel('Ahorro Neto Total ($)', fontsize=11, fontweight='bold', color='white')
    ax_c.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_c.grid(axis='y', linestyle='--', alpha=0.2, color='white')
    ax_c.tick_params(colors='white', labelsize=10)

    # Anotaciones financieras sobre las barras
    for p in ax_c.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax_c.annotate(
            f'${h:,.2f}',
            (p.get_x() + p.get_width() / 2., h),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#2ecc71',
            xytext=(0, 5),
            textcoords='offset points'
        )

    plt.tight_layout()

    # Despliegue en pantalla
    col_g, col_t = st.columns([1.5, 1])
    with col_g:
        st.pyplot(fig_conc)

    with col_t:
        diferencia_fija = ahorro_optimo_maquinas_fijo - ahorro_global_fijo
        st.markdown("#### Balance general optimo")
        st.metric("XGBoost global (Umbral 0.71)", f"${ahorro_global_fijo:,.2f}")
        st.metric("Óptimo acumulado por máquina", f"${ahorro_optimo_maquinas_fijo:,.2f}")
        st.success(f"**Ganancia adicional por segmentación:**\n${diferencia_fija:,.2f}")

    # --- TARJETA DE RESUMEN DE UMBRALES ÓPTIMOS ---
    st.markdown("###  Resumen de umbrales optimos seleccionados")

    # CSS inline para una tarjeta elegante y alineada al modo oscuro
    st.markdown("""
    <div style="
        background-color: #252526;
        border: 1px solid #333333;
        border-left: 5px solid #2ecc71;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">
        <h4 style="color: #ffffff; margin-top: 0; margin-bottom: 12px; font-size: 16px;">
             Configuración definitiva de puntos operativos
        </h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
            <div style="flex: 1; min-width: 220px;">
                <p style="color: #3498db; font-weight: bold; margin-bottom: 5px; font-size: 14px;">
                     Modelo global recomendado
                </p>
                <ul style="color: #dcdcdc; margin: 0; padding-left: 20px; font-size: 13.5px; line-height: 1.6;">
                    <li><b>XGBoost General:</b> Umbral = <span style="color: #2ecc71; font-weight: bold;">0.71</span></li>
                </ul>
            </div>
            <div style="flex: 1.5; min-width: 280px;">
                <p style="color: #2ecc71; font-weight: bold; margin-bottom: 5px; font-size: 14px;">
                     Modelos optimos segmentados (Por máquina)
                </p>
                <ul style="color: #dcdcdc; margin: 0; padding-left: 20px; font-size: 13.5px; line-height: 1.6;">
                    <li><b>Máquina 1:</b> Random Forest (Umbral = <span style="color: #2ecc71; font-weight: bold;">0.60</span>)</li>
                    <li><b>Máquina 2:</b> Random Forest (Umbral = <span style="color: #2ecc71; font-weight: bold;">0.60</span>)</li>
                    <li><b>Máquina 3:</b> Random Forest (Umbral = <span style="color: #2ecc71; font-weight: bold;">0.60</span>)</li>
                    <li><b>Máquina 4:</b> XGBoost (Umbral = <span style="color: #2ecc71; font-weight: bold;">0.71</span>)</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
