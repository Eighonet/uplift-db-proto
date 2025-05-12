# streamlit_uplift_dashboard.py
"""
Streamlit dashboard for uplift-model diagnostics (light theme).

Features:
---------
1. Column 1: Task & Data description
2. Column 2: Distribution of individual treatment effects
3. Column 3: Model overview

- Page-wide header showing the period (PERIOD_START–PERIOD_END).
- Columns reordered: Task/Data, Effect, Model.
- Matching background for Column 2 header and content.
- Uses tooltips for hyperparameters, Plotly figures, and Streamlit expanders.
- Typography: Montserrat; all tables display two decimal places.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx 
from pyvis.network import Network
import streamlit.components.v1 as components

# Display floats with 2 decimals
pd.options.display.float_format = "{:,.2f}".format

# ----------------------- Placeholder Task Params ------------------------
TREATMENT_VARIABLE = "has_credit"
TARGET_VARIABLE    = "pnl"
LAG_MONTHS         = 3
PERIOD_START       = "2021-01"
PERIOD_END         = "2021-12"
SAMPLING_DESC      = "Исключены неактивные клиенты"

# Data params
TREATMENT_TYPE     = "binary"
DISCRETIZATION_DESC= "Квантили: 0–25%, 25–50%, 50–75%, 75–100%"
SOURCE_VIEW        = "sales"
NAN_FILLING        = "Замена NaN на 0 для пропущенных значений"

X_FEATURES = [
    {"source": "sales", "variable": "has_credit",          "description": "Флаг наличия кредита"},
    {"source": "sales", "variable": "has_sbol",       "description": "Флаг наличия приложения"},
    {"source": "pnl_data", "variable": "pnl", "description": "Доходность клиента"},
    {"source": "clusters", "variable": "cluster_2_flag", "description": "Флаг принадложености ко 2 кластеру"},
]
X_FEATURES_DF = pd.DataFrame(X_FEATURES)

# ----------------------- Helpers --------------------------------------------
def calc_stats(values: np.ndarray) -> dict[str, float]:
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    return {
        "Mean":     float(np.mean(values)),
        "Median":   float(q2),
        "Q1":       float(q1),
        "Q3":       float(q3),
        "Std":      float(np.std(values, ddof=1)),
        "IQR":      float(iqr),
        "Skewness": float(pd.Series(values).skew()),
        "Kurtosis": float(pd.Series(values).kurt()),
        "Lower":    float(q2 - 1.5 * iqr),
        "Upper":    float(q2 + 1.5 * iqr),
    }


def make_tooltip(label: str, df: pd.DataFrame, width: int = 260) -> str:
    table_html = df.to_html(classes="hp-table", border=0)
    return (
        f"<span class='tooltip'>{label}"
        f"  <span class='tooltiptext' style='width:{width}px'>{table_html}</span>"
        f"</span>"
    )

# ----------------------- Plotly Figures -------------------------------------
def histogram_fig(values: np.ndarray, stats: dict[str, float]) -> go.Figure:
    bins = [10, 20, 50, 100]
    fig = go.Figure()
    for nb in bins:
        fig.add_histogram(
            x=values, nbinsx=nb,
            marker_color="#4e79a7", opacity=0.8,
            visible=(nb == 20), showlegend=False
        )
    fig.add_shape(type="rect",
        x0=min(values.min(), stats["Lower"]), x1=0,
        y0=0, y1=1, xref="x", yref="paper",
        fillcolor="rgba(255,255,255,0.23)", line_width=0, layer="below"
    )
    fig.add_shape(type="line",
        x0=stats["Median"], x1=stats["Median"], y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#FFFFFF", dash="dash", width=4), layer="above"
    )
    buttons = []
    for i, nb in enumerate(bins):
        vis = [False]*len(bins); vis[i]=True
        buttons.append(dict(label=f"{nb} bins", method="update", args=[{"visible":vis},{}]))
    span = max(values.max(), stats["Upper"]) - min(values.min(), stats["Lower"])
    fig.update_layout(
        template="simple_white",
        margin=dict(l=20, r=20, t=60, b=20),
        bargap=0.125,
        font=dict(family="Montserrat, sans-serif"),
        updatemenus=[dict(
            x=1.03, y=1.02,
            xanchor="right", yanchor="top",
            buttons=buttons
        )],
        showlegend=False
    )
    fig.update_xaxes(title_text="", dtick=np.round(span/7, 2), showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(title="количество клиентов", showline=True, linewidth=1, linecolor="black")
    return fig

def boxplot_fig(values: np.ndarray, stats: dict[str, float]) -> go.Figure:
    span = max(values.max(), stats["Upper"]) - min(values.min(), stats["Lower"])
    fig = go.Figure(go.Box(
        x=values, orientation='h', boxpoints=False,
        marker_color="#59a14f", hovertemplate="значение: %{x:.4f}<extra></extra>", showlegend=False
    ))
    fig.update_layout(
        template="simple_white", height=180,
        margin=dict(l=20, r=20, t=10, b=30),
        xaxis_title="эффект", font=dict(family="Montserrat, sans-serif"), showlegend=False
    )
    fig.update_xaxes(title="эффект", dtick=np.round(span/7, 2), showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", visible=False)
    return fig

def bar_chart_fig(effects: np.ndarray, flags: np.ndarray) -> go.Figure:
    boundaries = np.percentile(effects, np.arange(0, 101, 10))
    means, diffs = [], []
    for i in range(10):
        mask = (effects >= boundaries[i]) & (effects < boundaries[i+1])
        seg = effects[mask]; seg_flags = flags[mask]
        means.append(seg.mean() if seg.size else 0)
        t = seg[seg_flags==1]; c = seg[seg_flags==0]
        diffs.append((t.mean() if t.size else 0) - (c.mean() if c.size else 0))
    labels = [f"{i*10}–{(i+1)*10}%" for i in range(10)][::-1]
    means, diffs = means[::-1], diffs[::-1]
    fig = go.Figure()
    fig.add_bar(x=labels, y=means, name="Среднее", marker_color="#4e79a7")
    fig.add_bar(x=labels, y=diffs, name="Δ эффект (t–c)", marker_color="#e15759")
    dropdown=[dict(type="dropdown", x=0.99, y=0.99, xanchor="right", yanchor="top",
        buttons=[
            dict(label="коммуникация", method="relayout", args=[{"title": "Детальный анализ — коммуникация"}]),
            dict(label="покупка", method="relayout", args=[{"title": "Детальный анализ — покупка"}])
        ]
    )]
    fig.update_layout(
        barmode="group", template="simple_white",
        margin=dict(l=20, r=20, t=60, b=40),
        font=dict(family="Montserrat, sans-serif"),
        legend=dict(x=0, y=0, xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.5)"),
        updatemenus=dropdown, title="Детальный анализ — коммуникация"
    )
    fig.update_xaxes(tickangle=-45); fig.update_yaxes(title="значение эффекта")
    return fig

def ranking_curve_fig(kind: str = "uplift gain") -> go.Figure:
    x = np.linspace(0,1,101)
    raw = {"uplift gain":  x + 5*(1-x)*x*(1-2*x),
           "qini":         x + 3*(1-x)*x*(1-2*x),
           "adjusted qini":x + 4*(1-x)*x*(1-2*x)}
    curves = {n: y/y[-1] for n,y in raw.items()}
    fig = go.Figure()
    for idx,(name,y) in enumerate(curves.items()):
        fig.add_scatter(x=x, y=y, mode="lines", name=name, line=dict(width=3), visible=(name==kind), showlegend=False)
    baseline = x
    fig.add_scatter(x=x, y=baseline, mode="lines", name="y = x", line=dict(color="black", dash="dash"), showlegend=False)
    total=2*len(curves)+1
    for idx,(name,y) in enumerate(curves.items()):
        fig.add_scatter(x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([y, baseline[::-1]]),
                        fill="toself", fillcolor="rgba(78,121,167,0.15)", line=dict(width=0), visible=(name==kind), showlegend=False)
    buttons=[]
    for i,label in enumerate(curves.keys()):
        vis=[False]*total; vis[i]=True; vis[len(curves)]=True; vis[len(curves)+1+i]=True
        buttons.append(dict(label=label, method="update", args=[{"visible":vis}, {"title":f"Ранжирующая кривая – {label}"}]))
    fig.update_layout(template="simple_white", height=300, margin=dict(l=20,r=20,t=40,b=20),
        updatemenus=[dict(x=1.03, y=0.98, xanchor="right", yanchor="top", buttons=buttons, showactive=True)], showlegend=False)
    fig.update_xaxes(title="доля охваченной аудитории", range=[0,1])
    fig.update_yaxes(title="метрика", range=[0,1])
    return fig

# ----------------------- Main -----------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Uplift Dashboard", layout="wide")
        # Header
    header_html = f"""
    <div style="position:relative; left:50%; transform:translateX(-47.5%); width:100vw; padding:16px 32px; text-align:left;">
        <h1 style="margin:0; font-family:Montserrat, sans-serif; font-weight:400;">
            Оценка uplift эффекта приобретения кредитной карты на PnL <br> {PERIOD_START} – {PERIOD_END}
        </h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    # CSS & tooltips
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-weight: 600; }
        .stTable { width: 100% !important; }
        @media (max-width: 992px) { div[data-testid='column'] { width: 100% !important; flex: 1 0 100% !important; }}
        .tooltip { position: relative; display: inline-block; cursor: pointer; color: #2b8acf; }
        .tooltip .tooltiptext { visibility: hidden; opacity: 0; transition: opacity .2s;
                                position: absolute; z-index: 10; top: 135%; left: 0;
                                background: #fff; color: #000; border: 1px solid #ccc;
                                border-radius: 6px; padding: 8px; box-shadow: 0 4px 12px rgba(0,0,0,.15);
                                max-height:200px; overflow:auto; }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        .hp-table { font-size: 12px; border-collapse: collapse; width: 100%; }
        .hp-table th, .hp-table td { border: 1px solid #d3d3d3; padding: 4px 6px; text-align: left; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Data & setup
    np.random.seed(42)
    effects = 10000 * np.random.normal(loc=0.02, scale=0.04, size=50_000)
    flags   = np.random.binomial(1, 0.5, size=effects.shape[0])
    stats   = calc_stats(effects)
    global_params = pd.DataFrame({"Параметр":["iterations","has_uncertainty"],"Значение":[300,True]})
    sub1_params   = pd.DataFrame({"Параметр":["depth","learning_rate"],"value":[6,0.043]})
    sub2_params   = pd.DataFrame({"Параметр":["depth","learning_rate"],"value":[8,0.0019]})
    sub1_metrics  = pd.DataFrame({"Метрика":["MAE","RMSE","MAPE"],"Значение":[325.42,21432.38,12.34]}).set_index("Метрика")
    sub2_metrics  = pd.DataFrame({"Метрика":["MAE","RMSE","MAPE"],"Значение":[509.39,30451.03,15.67]}).set_index("Метрика")
    tooltip_model  = make_tooltip("XLearner", global_params)
    tooltip_sub1   = make_tooltip("CatBoost 1", sub1_params)
    tooltip_sub2   = make_tooltip("CatBoost 2", sub2_params)
    # Columns reordered + spacing
    col_old1, col_old2, col_old3 = st.columns(3, gap="large")
    col1, col2, col3 = col_old3, col_old1, col_old2

    # Column 1: Task & Data
    with col1:
        st.markdown("## Задача")
        with st.expander("Постановка задачи", expanded=True):
            st.markdown(f"<div><strong>Переменная воздействия:</strong> {TREATMENT_VARIABLE}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Целевая переменная:</strong> {TARGET_VARIABLE}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Временной лаг:</strong> {LAG_MONTHS} месяцев</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Период:</strong> {PERIOD_START} – {PERIOD_END}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Семплирование:</strong> {SAMPLING_DESC}</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
        st.markdown("## Данные")
        with st.expander("Treatment & outcome", expanded=True):
            st.markdown(f"<div><strong>Тип тритмента:</strong> {TREATMENT_TYPE}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Дискретизация:</strong> {DISCRETIZATION_DESC}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Источник:</strong> {SOURCE_VIEW}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Заполнение NaN:</strong> {NAN_FILLING}</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
        with st.expander("X", expanded=True):
            st.table(X_FEATURES_DF)
            csv = X_FEATURES_DF.to_csv(index=False)
            st.download_button(label="Скачать переменные", data=csv, file_name="x_features.csv", mime="text/csv")
            
    # Column 2: Effect distribution (with matching background)
    with col2:
        st.markdown("## Эффект")
        with st.expander("Распределение эффекта", expanded=True):
            st.plotly_chart(histogram_fig(effects, stats), use_container_width=True)
            st.plotly_chart(boxplot_fig(effects, stats),    use_container_width=True)
        with st.expander("Статистики эффекта", expanded=True):
            left_df = pd.DataFrame({
                "Статистика": ["Mean","Median","Q1","Q3"],
                "Значение":   [stats["Mean"],stats["Median"],stats["Q1"],stats["Q3"]]
            }).set_index("Статистика")
            right_df = pd.DataFrame({
                "Статистика": ["IQR","Std","Skewness","Kurtosis"],
                "Значение":   [stats["IQR"],stats["Std"],stats["Skewness"],stats["Kurtosis"]]
            }).set_index("Статистика")
            c1, c2 = st.columns(2)
            with c1: st.table(left_df)
            with c2: st.table(right_df)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("Причинно-следственный граф", expanded=True):
            feature_names = ["has_credit", "has_sbol", "pnl", "client_cluster_2"]
            edge_list = [
                ("has_credit", "pnl"),
                ("has_sbol", "has_credit"),
                ("has_sbol", "pnl"),
                ("client_cluster_2", "pnl"),
            ]
            edge_correlation = {
                ("has_credit", "pnl"):       0.75,
                ("has_sbol", "has_credit"):   0.65,
                ("has_sbol", "pnl"):         0.82,
                ("client_cluster_2", "pnl"): 0.68,
            }
            treatment = "has_credit"
            outcome   = "pnl"

            # Создаём pyvis-сеть
            net = Network(
                height="470px",
                width="100%",
                directed=True,
                bgcolor="#0e1117",    # прозрачный фон
                font_color="white"
            )
            net.force_atlas_2based()

            # Добавляем узлы с цветом
            for node in feature_names:
                if node == treatment:
                    color = "#2b8acf"
                elif node == outcome:
                    color = "#66cc66"
                else:
                    color = "#B0BEC5"
                net.add_node(node, label=node, color=color, borderWidth=1, size=15)

            # Добавляем ориентированные рёбра с подписью корреляции
            for src, tgt in edge_list:
                corr = edge_correlation[(src, tgt)]
                net.add_edge(
                    src,
                    tgt,
                    arrows="to",
                    label=f"{corr:.2f}",
                    title=f"ρₛ = {corr:.2f}",
                    width=corr * 5,
                    font={
                        "color": "white",       # цвет текста
                        "strokeWidth": 5,       # ширина обводки
                        "strokeColor": "black"  # цвет обводки
                    }
                )

            # Сохраняем и встраиваем в Streamlit
            net.save_graph("pyvis_graph.html")
            html = open("pyvis_graph.html", "r", encoding="utf-8").read()

            html += "<style>:root {background-color: #0e1117;} .card {border: 0px} #mynetwork {border: 0px}</style>"
            components.html(html, height=450)
 #           st.markdown(f"<div><strong>Переменная воздействия:</strong> {TREATMENT_VARIABLE}</div>", unsafe_allow_html=True)
        
    # Column 3: Model overview
    with col3:
        st.markdown("## Модель")
        with st.expander("Общая информация", expanded=True):
            st.markdown(f"<div><strong>Модель:</strong> {tooltip_model}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Подмодели:</strong> {tooltip_sub1} | {tooltip_sub2}</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
        with st.expander("Ранжирование", expanded=True):
            st.plotly_chart(ranking_curve_fig("uplift gain"), use_container_width=True)
        with st.expander("Детальный анализ перцентилей", expanded=True):
            st.plotly_chart(bar_chart_fig(effects, flags), use_container_width=True)

        with st.expander("Метрики submodel 1", expanded=True):
            st.table(sub1_metrics)
        with st.expander("Метрики submodel 2", expanded=True):
            st.table(sub2_metrics)

if __name__ == "__main__":
    main()
