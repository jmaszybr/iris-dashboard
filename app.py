import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Konfiguracja strony ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Iris Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f4ff; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    h1 { color: #5b2d8e; }
    h2, h3 { color: #7b3db5; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Dane ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.columns = ["Dł. kielicha (cm)", "Szer. kielicha (cm)", "Dł. płatka (cm)", "Szer. płatka (cm)"]
    df["Gatunek"] = pd.Categorical.from_codes(iris.target, ["Setosa", "Versicolor", "Virginica"])
    return df

df = load_data()

KOLORY = {
    "Setosa":     "#e63946",
    "Versicolor": "#2a9d8f",
    "Virginica":  "#e9c46a"
}
PALETA = list(KOLORY.values())

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg", use_container_width=True)
    st.markdown("## 🌸 Iris Explorer")
    st.markdown("**Edukacyjny dashboard** do eksploracji klasycznego zbioru danych Iris.")
    st.divider()

    gatunki = st.multiselect(
        "Filtruj gatunki:",
        options=["Setosa", "Versicolor", "Virginica"],
        default=["Setosa", "Versicolor", "Virginica"]
    )

    st.divider()
    st.markdown("### 📖 O zbiorze Iris")
    st.info("""
    Zbiór stworzony przez **Ronalda Fishera** w 1936 roku.
    Zawiera **150 próbek** trzech gatunków irysów z 4 pomiarami każda.
    Jeden z najbardziej znanych zbiorów w ML!
    """)

df_filtered = df[df["Gatunek"].isin(gatunki)]

# ── Nagłówek ─────────────────────────────────────────────────────────────────
st.title("🌸 Iris Dataset — Edukacyjny Dashboard")
st.markdown("Eksploruj klasyczny zbiór danych Fishera i odkryj różnice między gatunkami irysów.")
st.divider()

# ── Metryki ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌱 Próbek łącznie", len(df_filtered))
c2.metric("🌸 Gatunków", df_filtered["Gatunek"].nunique())
c3.metric("📏 Cech", 4)
c4.metric("📊 Brak danych", "0")

st.divider()

# ── Zakładki ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Rozkłady", "🔵 Scatter Plot", "🌡️ Korelacje", "📦 Boxploty", "🔬 PCA"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Rozkłady (histogramy)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Rozkłady cech dla każdego gatunku")
    st.markdown("Histogramy pokazują jak rozkładają się wartości każdej cechy. Czy gatunki się nakładają?")

    cechy = df_filtered.columns[:-1].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#f8f4ff")

    for ax, cecha in zip(axes.flatten(), cechy):
        for gatunek, kolor in KOLORY.items():
            if gatunek in gatunki:
                dane = df_filtered[df_filtered["Gatunek"] == gatunek][cecha]
                ax.hist(dane, bins=15, alpha=0.6, color=kolor, label=gatunek, edgecolor="white")
        ax.set_title(cecha, fontsize=11, fontweight="bold", color="#5b2d8e")
        ax.set_xlabel("Wartość (cm)", fontsize=9)
        ax.set_ylabel("Liczba próbek", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_facecolor("#fdfbff")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Rozkłady cech Iris", fontsize=14, fontweight="bold", color="#5b2d8e", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("💡 **Wniosek:** Setosa jest wyraźnie oddzielona od pozostałych gatunków, szczególnie w cechach płatka.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Scatter Plot
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Wykres rozrzutu — porównanie dwóch cech")

    col_a, col_b = st.columns(2)
    with col_a:
        osx = st.selectbox("Oś X:", cechy, index=2)
    with col_b:
        osy = st.selectbox("Oś Y:", cechy, index=3)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#f8f4ff")
    ax.set_facecolor("#fdfbff")

    for gatunek, kolor in KOLORY.items():
        if gatunek in gatunki:
            sub = df_filtered[df_filtered["Gatunek"] == gatunek]
            ax.scatter(sub[osx], sub[osy], c=kolor, label=gatunek,
                       alpha=0.8, s=80, edgecolors="white", linewidths=0.5)

    ax.set_xlabel(osx, fontsize=11)
    ax.set_ylabel(osy, fontsize=11)
    ax.set_title(f"{osx} vs {osy}", fontsize=13, fontweight="bold", color="#5b2d8e")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("💡 **Wskazówka:** Wybierz cechy płatka — zobaczysz idealne rozdzielenie gatunków!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Heatmapa korelacji
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Macierz korelacji cech")
    st.markdown("Korelacja pokazuje jak silnie dwie cechy są ze sobą powiązane (od -1 do 1).")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#f8f4ff")
    corr = df_filtered[cechy].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        vmin=-1, vmax=1, square=True
    )
    ax.set_title("Korelacja między cechami", fontsize=13, fontweight="bold", color="#5b2d8e", pad=15)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("💡 **Wniosek:** Długość i szerokość płatka są silnie skorelowane (0.96) — im dłuższy płatek, tym szerszy.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Boxploty
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Boxploty — rozkład i wartości odstające")
    st.markdown("Boxplot pokazuje medianę, kwartyle i wartości odstające dla każdego gatunku.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#f8f4ff")

    for ax, cecha in zip(axes.flatten(), cechy):
        dane_box = [df_filtered[df_filtered["Gatunek"] == g][cecha].values
                    for g in gatunki]
        bp = ax.boxplot(dane_box, labels=gatunki, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, g in zip(bp["boxes"], gatunki):
            patch.set_facecolor(KOLORY[g])
            patch.set_alpha(0.7)
        ax.set_title(cecha, fontsize=11, fontweight="bold", color="#5b2d8e")
        ax.set_ylabel("Wartość (cm)", fontsize=9)
        ax.set_facecolor("#fdfbff")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Boxploty cech Iris", fontsize=14, fontweight="bold", color="#5b2d8e", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("💡 **Wniosek:** Setosa ma wyraźnie mniejsze płatki — pudełka w ogóle się nie nakładają!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("PCA — redukcja wymiarowości do 2D")
    st.markdown("""
    **PCA (Principal Component Analysis)** redukuje 4 cechy do 2 głównych składowych,
    zachowując jak najwięcej informacji. Dzięki temu możemy zobaczyć dane w 2D.
    """)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[cechy])
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#f8f4ff")
    ax.set_facecolor("#fdfbff")

    for gatunek, kolor in KOLORY.items():
        if gatunek in gatunki:
            idx = df_filtered["Gatunek"].values == gatunek
            ax.scatter(components[idx, 0], components[idx, 1],
                       c=kolor, label=gatunek, alpha=0.85, s=90,
                       edgecolors="white", linewidths=0.5)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% wariancji)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% wariancji)", fontsize=11)
    ax.set_title("PCA — Iris w przestrzeni 2D", fontsize=13, fontweight="bold", color="#5b2d8e")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("PC1 wyjaśnia", f"{var[0]:.1f}% wariancji")
    col2.metric("PC2 wyjaśnia", f"{var[1]:.1f}% wariancji")

    st.info(f"💡 **Wniosek:** Dwie pierwsze składowe wyjaśniają łącznie **{var[0]+var[1]:.1f}%** całej zmienności danych — to bardzo dużo!")

# ── Stopka ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#aaa; font-size:0.85rem'>🌸 Iris Dashboard • Dane: R.A. Fisher (1936) • Zbudowany w Streamlit</center>",
    unsafe_allow_html=True
)
