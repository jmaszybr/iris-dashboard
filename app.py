import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Konfiguracja ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Iris Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Premium ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f72585, #b5179e, #7209b7, #4361ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.kpi-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-number { font-size: 2rem; font-weight: 700; color: #f72585; }
.kpi-label  { font-size: 0.8rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px; }

.tab-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: white;
    margin-bottom: 0.3rem;
}
.tab-desc {
    color: rgba(255,255,255,0.5);
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}

.insight-box {
    background: linear-gradient(135deg, rgba(247,37,133,0.15), rgba(67,97,238,0.15));
    border-left: 3px solid #f72585;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1.2rem;
    color: rgba(255,255,255,0.85);
    font-size: 0.92rem;
    margin-top: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.6);
    font-weight: 600;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f72585, #7209b7) !important;
    color: white !important;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.8rem;
    border: 1px solid rgba(255,255,255,0.1);
}
div[data-testid="stMetricValue"] { color: #f72585 !important; }
div[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; }
</style>
""", unsafe_allow_html=True)

# ── Dane ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=["Dł. kielicha", "Szer. kielicha", "Dł. płatka", "Szer. płatka"])
    df["Gatunek"] = pd.Categorical.from_codes(iris.target, ["Setosa", "Versicolor", "Virginica"])
    return df

df = load_data()
CECHY = ["Dł. kielicha", "Szer. kielicha", "Dł. płatka", "Szer. płatka"]

KOLORY = {"Setosa": "#f72585", "Versicolor": "#4cc9f0", "Virginica": "#f8961e"}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="rgba(255,255,255,0.8)", family="Inter"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.06)",
        bordercolor="rgba(255,255,255,0.12)",
        borderwidth=1,
        font=dict(size=12)
    ),
    margin=dict(t=50, b=40, l=40, r=20),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0'>
        <div style='font-size:3rem'>🌸</div>
        <div style='font-size:1.3rem; font-weight:700; color:white'>Iris Explorer</div>
        <div style='font-size:0.8rem; color:rgba(255,255,255,0.4); margin-top:4px'>Dashboard Edukacyjny</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='color:rgba(255,255,255,0.6); font-size:0.85rem; margin-bottom:8px'>🎯 FILTRUJ GATUNKI</div>", unsafe_allow_html=True)

    gatunki = []
    for g, kolor in KOLORY.items():
        if st.checkbox(g, value=True, key=g):
            gatunki.append(g)

    st.divider()
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05); border-radius:12px; padding:1rem; font-size:0.85rem; color:rgba(255,255,255,0.6)'>
        <div style='color:white; font-weight:600; margin-bottom:8px'>📖 O zbiorze Iris</div>
        Klasyczny zbiór <b style='color:#f72585'>Ronalda Fishera</b> z 1936 roku.<br><br>
        🔢 <b style='color:white'>150</b> próbek<br>
        🌸 <b style='color:white'>3</b> gatunki<br>
        📏 <b style='color:white'>4</b> cechy mierzone w cm
    </div>
    """, unsafe_allow_html=True)

if not gatunki:
    st.warning("Wybierz co najmniej jeden gatunek!")
    st.stop()

df_f = df[df["Gatunek"].isin(gatunki)]
kolor_map = {g: KOLORY[g] for g in gatunki}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌸 Iris Dataset Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Odkryj wzorce w klasycznym zbiorze danych Fishera</div>', unsafe_allow_html=True)

# ── KPI ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("🌱", str(len(df_f)), "Próbek"),
    ("🌸", str(df_f["Gatunek"].nunique()), "Gatunków"),
    ("📏", "4", "Cechy"),
    ("✅", "100%", "Kompletność"),
]
for col, (icon, val, label) in zip([k1,k2,k3,k4], kpis):
    col.markdown(f"""
    <div class="kpi-card">
        <div style='font-size:1.6rem'>{icon}</div>
        <div class="kpi-number">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Zakładki ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Rozkłady", "🔵 Scatter", "🌡️ Korelacje", "📦 Boxploty", "🔬 PCA"
])

# ══ TAB 1 — Rozkłady ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="tab-header">Rozkłady cech</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Jak rozkładają się wartości każdej cechy? Czy gatunki się nakładają?</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=2, subplot_titles=CECHY,
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    pos = [(1,1),(1,2),(2,1),(2,2)]

    for (r,c), cecha in zip(pos, CECHY):
        for g in gatunki:
            dane = df_f[df_f["Gatunek"]==g][cecha]
            fig.add_trace(go.Histogram(
                x=dane, name=g, legendgroup=g,
                showlegend=(r==1 and c==1),
                marker_color=KOLORY[g],
                opacity=0.75, nbinsx=15,
                hovertemplate=f"<b>{g}</b><br>{cecha}: %{{x:.2f}}<br>Liczba: %{{y}}<extra></extra>"
            ), row=r, col=c)

    fig.update_layout(barmode="overlay", height=550,
                      title_text="Rozkłady cech Iris",
                      title_font=dict(size=16, color="white"),
                      **PLOTLY_THEME)
    for ann in fig.layout.annotations:
        ann.font = dict(color="rgba(255,255,255,0.7)", size=12)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Wniosek:</b> Setosa jest wyraźnie oddzielona od pozostałych gatunków — szczególnie w cechach płatka. Versicolor i Virginica częściowo się nakładają.</div>', unsafe_allow_html=True)

# ══ TAB 2 — Scatter ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tab-header">Wykres rozrzutu</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Wybierz dwie cechy i obserwuj jak gatunki się separują w przestrzeni 2D.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        osx = st.selectbox("Oś X", CECHY, index=2)
    with col_b:
        osy = st.selectbox("Oś Y", CECHY, index=3)

    fig = px.scatter(
        df_f, x=osx, y=osy, color="Gatunek",
        color_discrete_map=kolor_map,
        symbol="Gatunek",
        hover_data=CECHY,
        marginal_x="violin", marginal_y="violin",
        template="plotly_dark",
        title=f"<b>{osx}</b> vs <b>{osy}</b>",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85,
                                  line=dict(width=0.5, color="rgba(255,255,255,0.3)")))
    fig.update_layout(height=580, **PLOTLY_THEME)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Wskazówka:</b> Wybierz <b>Dł. płatka</b> i <b>Szer. płatka</b> — zobaczysz niemal idealne rozdzielenie trzech gatunków!</div>', unsafe_allow_html=True)

# ══ TAB 3 — Korelacje ═════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="tab-header">Macierz korelacji</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Jak silnie powiązane są ze sobą cechy? Wartości od -1 (odwrotna) do +1 (silna pozytywna).</div>', unsafe_allow_html=True)

    corr = df_f[CECHY].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.index,
        colorscale=[[0,"#4361ee"],[0.5,"#f8f9fa"],[1,"#f72585"]],
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hovertemplate="%{x} × %{y}<br>Korelacja: <b>%{z}</b><extra></extra>",
        showscale=True,
        colorbar=dict(
            title="Korelacja",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            bordercolor="rgba(255,255,255,0.1)"
        )
    ))
    fig.update_layout(
        title="Macierz korelacji Pearsona",
        title_font=dict(size=15, color="white"),
        height=480, **PLOTLY_THEME
    )
    fig.update_xaxes(tickfont=dict(color="rgba(255,255,255,0.7)"))
    fig.update_yaxes(tickfont=dict(color="rgba(255,255,255,0.7)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Wniosek:</b> Długość i szerokość płatka są <b>bardzo silnie skorelowane (0.96)</b> — można powiedzieć, że prawie zawsze rosną razem. Kielich nie wykazuje tak silnych zależności.</div>', unsafe_allow_html=True)

# ══ TAB 4 — Boxploty ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="tab-header">Boxploty</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Mediana, kwartyle i wartości odstające — wszystko w jednym wykresie.</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=4, subplot_titles=CECHY,
                        horizontal_spacing=0.06)

    for i, cecha in enumerate(CECHY, 1):
        for g in gatunki:
            dane = df_f[df_f["Gatunek"]==g][cecha]
            fig.add_trace(go.Violin(
                y=dane, name=g, legendgroup=g,
                showlegend=(i==1),
                fillcolor=KOLORY[g],
                line_color=KOLORY[g],
                opacity=0.75,
                box_visible=True,
                meanline_visible=True,
                meanline_color="white",
                points="outliers",
                marker=dict(color=KOLORY[g], size=4),
                hovertemplate=f"<b>{g}</b><br>{cecha}: %{{y:.2f}}<extra></extra>"
            ), row=1, col=i)

    fig.update_layout(
        violinmode="group", height=500,
        title_text="Violin + Box ploty cech Iris",
        title_font=dict(size=15, color="white"),
        **PLOTLY_THEME
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color="rgba(255,255,255,0.7)", size=11)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Wniosek:</b> Violin plot łączy boxplot z rozkładem gęstości. Setosa ma wąski, skupiony rozkład płatków — Virginica jest najbardziej zróżnicowana.</div>', unsafe_allow_html=True)

# ══ TAB 5 — PCA ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="tab-header">PCA — Redukcja wymiarowości</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">PCA kompresuje 4 cechy do 2 wymiarów zachowując maksimum informacji.</div>', unsafe_allow_html=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_f[CECHY])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    df_pca = pd.DataFrame({
        "PC1": coords[:,0], "PC2": coords[:,1],
        "Gatunek": df_f["Gatunek"].values,
        **{c: df_f[c].values for c in CECHY}
    })

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(
            df_pca, x="PC1", y="PC2", color="Gatunek",
            color_discrete_map=kolor_map,
            symbol="Gatunek",
            hover_data=CECHY,
            template="plotly_dark",
            title=f"PCA — {var[0]:.1f}% + {var[1]:.1f}% = <b>{var[0]+var[1]:.1f}%</b> wariancji",
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85,
                                      line=dict(width=0.5, color="rgba(255,255,255,0.3)")))
        fig.update_layout(
            height=480,
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            **PLOTLY_THEME
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric("PC1", f"{var[0]:.1f}%", "wariancji")
        st.metric("PC2", f"{var[1]:.1f}%", "wariancji")
        st.metric("Łącznie", f"{var[0]+var[1]:.1f}%", "zachowane")

        loadings = pd.DataFrame(
            pca.components_.T, index=CECHY, columns=["PC1","PC2"]
        ).round(2)
        st.markdown("<br>**Ładunki PCA:**", unsafe_allow_html=True)
        st.dataframe(loadings, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Wniosek:</b> Setosa jest wyraźnie odizolowana. Dwie pierwsze składowe wyjaśniają ponad 95% zmienności — cztery cechy można skutecznie zastąpić dwiema!</div>', unsafe_allow_html=True)

# ── Stopka ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:1.5rem; 
     background:rgba(255,255,255,0.03); border-radius:16px;
     border:1px solid rgba(255,255,255,0.07);
     color:rgba(255,255,255,0.35); font-size:0.82rem'>
    🌸 Iris Dataset Explorer &nbsp;•&nbsp; Dane: R.A. Fisher (1936) &nbsp;•&nbsp; Zbudowany z ❤️ w Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
