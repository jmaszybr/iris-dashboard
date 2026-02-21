import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="🌸 Iris Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Bootstrap 5 + CSS ─────────────────────────────────────────────────────────
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<style>
    .stApp { background-color: #f0f4f8; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dee2e6; }
    .stTabs [data-baseweb="tab-list"] { background: #ffffff; border-radius: 10px; padding: 4px; border: 1px solid #dee2e6; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; color: #495057; }
    .stTabs [aria-selected="true"] { background: #0d6efd !important; color: white !important; }
    div[data-testid="stMetricValue"] { color: #0d6efd !important; font-weight: 700; }
    h1,h2,h3 { color: #212529; }
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
KOLORY = {"Setosa": "#0d6efd", "Versicolor": "#198754", "Virginica": "#dc3545"}

THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#212529", family="system-ui, sans-serif"),
    legend=dict(bgcolor="white", bordercolor="#dee2e6", borderwidth=1),
    margin=dict(t=50, b=40, l=40, r=20),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="text-center py-3">
        <span style="font-size:2.5rem">🌸</span>
        <h4 class="mt-2 mb-0 fw-bold text-primary">Iris Explorer</h4>
        <small class="text-muted">Dashboard Edukacyjny</small>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**🎯 Filtruj gatunki**")
    gatunki = []
    for g, kolor in KOLORY.items():
        if st.checkbox(g, value=True, key=g):
            gatunki.append(g)

    st.markdown("""<hr>
    <div class="card border-0 bg-light rounded-3 p-3">
        <h6 class="fw-bold text-primary"><i class="bi bi-info-circle"></i> O zbiorze Iris</h6>
        <small class="text-muted">
            Klasyczny zbiór <strong>Ronalda Fishera</strong> z 1936 r.<br><br>
            🔢 <strong>150</strong> próbek<br>
            🌸 <strong>3</strong> gatunki<br>
            📏 <strong>4</strong> cechy (cm)
        </small>
    </div>""", unsafe_allow_html=True)

if not gatunki:
    st.warning("Wybierz co najmniej jeden gatunek!")
    st.stop()

df_f = df[df["Gatunek"].isin(gatunki)]
kolor_map = {g: KOLORY[g] for g in gatunki}

# ── Nagłówek ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card border-0 shadow-sm rounded-4 p-4 mb-3" style="background:linear-gradient(135deg,#0d6efd,#6610f2)">
    <h1 class="text-white fw-bold mb-1">🌸 Iris Dataset Explorer</h1>
    <p class="text-white-50 mb-0">Odkryj wzorce w klasycznym zbiorze danych Fishera • Edukacyjny Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("bi-collection", "0d6efd", str(len(df_f)), "Próbek"),
    ("bi-flower1",    "198754", str(df_f["Gatunek"].nunique()), "Gatunków"),
    ("bi-rulers",     "dc3545", "4", "Cechy"),
    ("bi-check-circle","fd7e14","100%", "Kompletność"),
]
for col, (icon, kolor, val, label) in zip([k1,k2,k3,k4], kpis):
    col.markdown(f"""
    <div class="card border-0 shadow-sm rounded-4 p-3 text-center h-100">
        <i class="bi {icon} text-{kolor if kolor in ['primary','success','danger','warning'] else ''}"
           style="font-size:1.8rem;color:#{kolor}"></i>
        <div class="fw-bold fs-3 mt-1" style="color:#{kolor}">{val}</div>
        <div class="text-muted small text-uppercase">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Zakładki ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Rozkłady", "🔵 Scatter", "🌡️ Korelacje", "📦 Violin", "🔬 PCA"
])

def insight(txt):
    st.markdown(f"""
    <div class="alert alert-primary border-0 rounded-3 mt-3 d-flex align-items-start gap-2">
        <i class="bi bi-lightbulb-fill fs-5 text-warning mt-1"></i>
        <div><strong>Wniosek:</strong> {txt}</div>
    </div>""", unsafe_allow_html=True)

def card_header(title, desc):
    st.markdown(f"""
    <div class="mb-3">
        <h4 class="fw-bold mb-1">{title}</h4>
        <p class="text-muted mb-0">{desc}</p>
    </div>""", unsafe_allow_html=True)

# ══ TAB 1 ═════════════════════════════════════════════════════════════════════
with tab1:
    card_header("Rozkłady cech", "Jak rozkładają się wartości każdej cechy? Czy gatunki się nakładają?")
    fig = make_subplots(rows=2, cols=2, subplot_titles=CECHY,
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    for (r,c), cecha in zip([(1,1),(1,2),(2,1),(2,2)], CECHY):
        for g in gatunki:
            fig.add_trace(go.Histogram(
                x=df_f[df_f["Gatunek"]==g][cecha], name=g, legendgroup=g,
                showlegend=(r==1 and c==1),
                marker_color=KOLORY[g], opacity=0.7, nbinsx=15,
                hovertemplate=f"<b>{g}</b><br>{cecha}: %{{x:.2f}}<br>Liczba: %{{y}}<extra></extra>"
            ), row=r, col=c)
    fig.update_layout(barmode="overlay", height=520, **THEME)
    fig.update_xaxes(gridcolor="#e9ecef", zeroline=False)
    fig.update_yaxes(gridcolor="#e9ecef", zeroline=False)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#495057")
    st.plotly_chart(fig, use_container_width=True)
    insight("Setosa jest wyraźnie oddzielona od pozostałych — szczególnie w cechach płatka. Versicolor i Virginica częściowo się nakładają.")

# ══ TAB 2 ═════════════════════════════════════════════════════════════════════
with tab2:
    card_header("Wykres rozrzutu", "Wybierz dwie cechy i obserwuj jak gatunki separują się w przestrzeni 2D.")
    col_a, col_b = st.columns(2)
    osx = col_a.selectbox("Oś X", CECHY, index=2)
    osy = col_b.selectbox("Oś Y", CECHY, index=3)
    fig = px.scatter(df_f, x=osx, y=osy, color="Gatunek",
                     color_discrete_map=kolor_map, symbol="Gatunek",
                     hover_data=CECHY, marginal_x="violin", marginal_y="violin",
                     title=f"<b>{osx}</b> vs <b>{osy}</b>")
    fig.update_traces(marker=dict(size=9, opacity=0.85,
                                  line=dict(width=0.5, color="white")))
    fig.update_layout(height=560, **THEME)
    fig.update_xaxes(gridcolor="#e9ecef")
    fig.update_yaxes(gridcolor="#e9ecef")
    st.plotly_chart(fig, use_container_width=True)
    insight("Wybierz <b>Dł. płatka</b> i <b>Szer. płatka</b> — zobaczysz niemal idealne rozdzielenie trzech gatunków!")

# ══ TAB 3 ═════════════════════════════════════════════════════════════════════
with tab3:
    card_header("Macierz korelacji", "Jak silnie powiązane są ze sobą cechy? Od -1 (odwrotna) do +1 (silna pozytywna).")
    corr = df_f[CECHY].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#0d6efd"],[0.5,"#f8f9fa"],[1,"#dc3545"]],
        zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=13, color="#212529"),
        hovertemplate="%{x} × %{y}<br>Korelacja: <b>%{z}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Korelacja", font=dict(color="#212529")),
            tickfont=dict(color="#212529")
        )
    ))
    fig.update_layout(height=460,
                      title="Macierz korelacji Pearsona",
                      title_font=dict(size=15, color="#212529"),
                      **THEME)
    st.plotly_chart(fig, use_container_width=True)
    insight("Długość i szerokość płatka są <b>bardzo silnie skorelowane (0.96)</b> — prawie zawsze rosną razem.")

# ══ TAB 4 ═════════════════════════════════════════════════════════════════════
with tab4:
    card_header("Violin + Box", "Rozkład gęstości, mediana, kwartyle i wartości odstające w jednym wykresie.")
    fig = make_subplots(rows=1, cols=4, subplot_titles=CECHY, horizontal_spacing=0.06)
    for i, cecha in enumerate(CECHY, 1):
        for g in gatunki:
            fig.add_trace(go.Violin(
                y=df_f[df_f["Gatunek"]==g][cecha], name=g, legendgroup=g,
                showlegend=(i==1),
                fillcolor=KOLORY[g], line_color=KOLORY[g],
                opacity=0.7, box_visible=True,
                meanline_visible=True, meanline_color="#212529",
                points="outliers",
                hovertemplate=f"<b>{g}</b><br>{cecha}: %{{y:.2f}}<extra></extra>"
            ), row=1, col=i)
    fig.update_layout(violinmode="group", height=480, **THEME)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#495057")
    fig.update_yaxes(gridcolor="#e9ecef", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    insight("Setosa ma wąski, skupiony rozkład płatków. Virginica jest najbardziej zróżnicowana.")

# ══ TAB 5 ═════════════════════════════════════════════════════════════════════
with tab5:
    card_header("PCA — Redukcja wymiarowości", "PCA kompresuje 4 cechy do 2 wymiarów zachowując maksimum informacji.")
    X = StandardScaler().fit_transform(df_f[CECHY])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    df_pca = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1],
                            "Gatunek": df_f["Gatunek"].values,
                            **{c: df_f[c].values for c in CECHY}})

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_pca, x="PC1", y="PC2", color="Gatunek",
                         color_discrete_map=kolor_map, symbol="Gatunek",
                         hover_data=CECHY,
                         title=f"PCA — łącznie <b>{var[0]+var[1]:.1f}%</b> wariancji")
        fig.update_traces(marker=dict(size=10, opacity=0.85,
                                      line=dict(width=0.5, color="white")))
        fig.update_layout(height=460,
                          xaxis_title=f"PC1 ({var[0]:.1f}%)",
                          yaxis_title=f"PC2 ({var[1]:.1f}%)",
                          **THEME)
        fig.update_xaxes(gridcolor="#e9ecef")
        fig.update_yaxes(gridcolor="#e9ecef")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("PC1", f"{var[0]:.1f}%", "wariancji")
        st.metric("PC2", f"{var[1]:.1f}%", "wariancji")
        st.metric("Łącznie", f"{var[0]+var[1]:.1f}%", "zachowane")
        st.markdown("**Ładunki PCA:**")
        loadings = pd.DataFrame(pca.components_.T, index=CECHY,
                                 columns=["PC1","PC2"]).round(2)
        st.dataframe(loadings, use_container_width=True)

    insight(f"Dwie pierwsze składowe wyjaśniają <b>{var[0]+var[1]:.1f}%</b> zmienności. Setosa jest idealnie odizolowana od pozostałych gatunków!")

# ── Stopka ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card border-0 bg-white shadow-sm rounded-4 p-3 mt-3 text-center text-muted small">
    🌸 Iris Dataset Explorer &nbsp;•&nbsp; Dane: R.A. Fisher (1936) &nbsp;•&nbsp; Zbudowany z ❤️ w Streamlit + Plotly + Bootstrap 5
</div>
""", unsafe_allow_html=True)
