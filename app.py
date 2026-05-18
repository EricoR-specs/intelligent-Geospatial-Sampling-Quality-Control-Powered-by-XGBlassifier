import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — mencegah blank putih
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely import wkt
from shapely.geometry import Point

# Suppress sklearn InconsistentVersionWarning (minor 1.7.1 → 1.8.0)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GeoValid — Spatial QC Validator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK GEOSPATIAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       #111827;
    --bg-card2:      #1a2235;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff88;
    --accent-orange: #ff6b35;
    --accent-purple: #8b5cf6;
    --text-primary:  #e8edf5;
    --text-muted:    #6b7a99;
    --border:        rgba(0, 212, 255, 0.15);
    --glow:          0 0 20px rgba(0, 212, 255, 0.15);
}

/* ── App Background ── */
.stApp {
    background-image:
        url("https://huggingface.co/spaces/EricoR/SGC_Petak_QC_Validator/resolve/main/Gemini_Generated_Image_7b5zv37b5zv37b5z.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Dark overlay di atas background image ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(5, 8, 18, 0.72);
    z-index: 0;
    pointer-events: none;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(8, 12, 24, 0.98) !important;
    border-right: 1px solid var(--border);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Main Content Overlay — solid dark card ── */
.main .block-container {
    background: rgba(8, 12, 24, 0.97);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    box-shadow: 0 8px 48px rgba(0,0,0,0.6);
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

/* ── Header Badge ── */
.hero-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--accent-cyan), #5eead4, var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.hero-author {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.3rem 1rem;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 99px;
    font-size: 0.8rem;
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: var(--glow);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 0.3rem;
}
.metric-tag {
    font-size: 0.7rem;
    margin-top: 0.25rem;
    padding: 0.15rem 0.6rem;
    border-radius: 99px;
    display: inline-block;
}
.tag-excellent { background: rgba(0,255,136,0.12); color: var(--accent-green); border: 1px solid rgba(0,255,136,0.3); }
.tag-superior  { background: rgba(139,92,246,0.12); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.3); }
.tag-stable    { background: rgba(0,212,255,0.12);  color: var(--accent-cyan);   border: 1px solid rgba(0,212,255,0.3); }

/* ── Section Headers ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent-cyan);
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

/* ── Insight Cards ── */
.insight-card {
    background: var(--bg-card);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
    color: #b0bcd4;
    line-height: 1.6;
}
.insight-icon {
    font-size: 1.1rem;
    margin-right: 0.5rem;
}

/* ── Result Boxes ── */
.result-valid {
    background: rgba(0, 255, 136, 0.07);
    border: 1px solid rgba(0, 255, 136, 0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-invalid {
    background: rgba(255, 107, 53, 0.07);
    border: 1px solid rgba(255, 107, 53, 0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.result-prob {
    font-size: 0.9rem;
    margin-top: 0.4rem;
    color: var(--text-muted);
}

/* ── Feature Table ── */
.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
}
.feature-row:last-child { border-bottom: none; }
.feature-name { color: var(--text-muted); font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.feature-val  { color: var(--text-primary); font-weight: 500; }

/* ── Point Status Badges ── */
.point-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
}
.badge-inside  { background: rgba(0,255,136,0.12); color: var(--accent-green); border: 1px solid rgba(0,255,136,0.3); }
.badge-outside { background: rgba(255,107,53,0.12); color: var(--accent-orange); border: 1px solid rgba(255,107,53,0.3); }

/* ── Streamlit Overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: rgba(10, 15, 30, 0.98) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
}

/* Pastikan semua widget wrapper punya background solid */
.stTextInput, .stTextArea, .stNumberInput, .stSelectbox {
    background: transparent !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #8892a4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.8) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(0, 212, 255, 0.4) !important;
}
label, .stSelectbox label, .stNumberInput label {
    color: #8892a4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}
h1, h2, h3, h4 { color: var(--text-primary) !important; }
p { color: #c8d3e8 !important; }

/* Expander — solid dark ── */
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: rgba(8, 12, 24, 0.98) !important;
    backdrop-filter: blur(12px) !important;
}
.stExpander summary {
    background: rgba(8, 12, 24, 0.98) !important;
    color: var(--text-primary) !important;
}
[data-testid="stExpanderDetails"] {
    background: rgba(8, 12, 24, 0.98) !important;
}

/* Metric cards — solid ── */
[data-testid="stMetric"] {
    background: rgba(15, 22, 42, 0.98) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent-cyan) !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #8892a4 !important; }

/* Tabs — solid ── */
[data-testid="stTabs"] {
    background: rgba(8, 12, 24, 0.97) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }
hr { border-color: var(--border) !important; }

/* Divider ── */
[data-testid="stDivider"] { border-color: rgba(0,212,255,0.15) !important; }</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🛰️ GeoValid</p>
    <p class="hero-subtitle">Intelligent Geospatial Sampling Quality Control — Powered by XGB Classifier</p>
    <span class="hero-author">✦ Author: Muhammad Erico Ricardo</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-family: 'Space Mono', monospace; font-size:1.1rem; color:#00d4ff;">📊 Model Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Performance Metrics
    st.markdown('<p class="section-label">Performance Metrics</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy", "95.15%", help="Test set accuracy")
    col_b.metric("ROC AUC", "0.9931", help="Near-perfect class separation")
    col_a.metric("F1-Score", "95.14%", help="Macro average")
    col_b.metric("Precision", "95.96%", help="Optimized via hyperparameter tuning")

    st.divider()

    # Model Info
    st.markdown('<p class="section-label">Model Information</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card" style="margin:0; border-left-color: #8b5cf6;">
        <b style="color:#8b5cf6;">Algorithm</b><br>
        XGBClassifier<br>
        <small style="color:#6b7a99;">n_estimators=200, max_depth=7, lr=0.1</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card" style="margin-top:0.5rem; border-left-color: #00ff88;">
        <b style="color:#00ff88;">Projection</b><br>
        UTM Zone 48S (EPSG:32748)<br>
        <small style="color:#6b7a99;">Meter-accurate geospatial ops</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card" style="margin-top:0.5rem; border-left-color: #00d4ff;">
        <b style="color:#00d4ff;">Status</b><br>
        🟢 Production Ready<br>
        <small style="color:#6b7a99;">Validated on unseen test data</small>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; color:#3d4f6e; text-align:center; line-height:1.6;">
        <b>Key Features Used:</b><br>
        UTM 48S Distance · Movement Angle<br>
        Inside/Outside Status · Device Accuracy<br>
        Satellite Count · Centroid Distance
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('best_xgboost_spatial_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))

def render_feature_table(df_features):
    feature_info = {
        "in_1": "Point QC-1 in Polygon",
        "in_2": "Point QC-2 in Polygon",
        "in_3": "Point QC-3 in Polygon",
        "all_inside": "All Points Inside",
        "dist_c1": "Distance to Centroid QC-1 (m)",
        "dist_c2": "Distance to Centroid QC-2 (m)",
        "dist_c3": "Distance to Centroid QC-3 (m)",
        "angle_1": "Angle from Centroid → QC-1 (°)",
        "angle_2": "Angle from Centroid → QC-2 (°)",
        "angle_3": "Angle from Centroid → QC-3 (°)",
        "move_angle_12": "Movement Angle QC1→QC2 (°)",
        "move_angle_23": "Movement Angle QC2→QC3 (°)",
        "AKURASI": "Device Accuracy (m)",
        "JUMLAH_SATELIT": "Satellite Count",
    }
    rows = ""
    for col in df_features.columns:
        val = df_features.iloc[0][col]
        label = feature_info.get(col, col)
        if isinstance(val, float):
            display = f"{val:.4f}"
        else:
            display = str(int(val))
        rows += f'<div class="feature-row"><span class="feature-name">{label}</span><span class="feature-val">{display}</span></div>'
    return f'<div style="background:var(--bg-card); border:1px solid var(--border); border-radius:10px; overflow:hidden;">{rows}</div>'


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div style="background: rgba(8,12,24,0.97); border: 1px solid rgba(0,212,255,0.18);
            border-radius: 14px; padding: 1.5rem 1.8rem 0.5rem;
            box-shadow: 0 4px 32px rgba(0,0,0,0.5); margin-bottom: 0.5rem;">
    <p class="section-label" style="margin-bottom:0.3rem;">Input & Prediksi</p>
    <p class="section-title" style="margin-bottom:0;">📥 Data Lapangan</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    shape_input = st.text_area(
        "Polygon Area (WKT Format)",
        "POLYGON ((105.279121611669 -4.61294944550871, 105.282742642802 -4.61295160307554, 105.28273552638 -4.61305213663778, 105.282628246063 -4.61321015621269, 105.282556732449 -4.613332257707, 105.282478051271 -4.61342563898184, 105.28245258899 -4.61352565822589, 105.282406671129 -4.61364817075038, 105.282397946226 -4.61384785040607, 105.282324510892 -4.61422994589588, 105.282202859793 -4.61424076542341, 105.282013192225 -4.61416903314357, 105.281791294663 -4.61407902912378, 105.281587341926 -4.61402558687888, 105.280223999308 -4.61380352262068, 105.280037894649 -4.61369229416355, 105.279969926283 -4.61373181519669, 105.279927042557 -4.61386826669822, 105.279873430047 -4.61402014241115, 105.279110981559 -4.61402007137196, 105.279121611669 -4.61294944550871))",
        height=90,
        help="Masukkan polygon dalam format WKT. Gunakan koordinat WGS84 (lon lat)."
    )

    st.markdown('<p style="color:#8892a4; font-size:0.8rem; font-weight:600; margin-bottom:0.3rem;">📍 Titik Sampling (Format: lat, lon)</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    qc1 = c1.text_input("QC Point 1", "-4.613253116607666,105.2796859741211", help="Koordinat pengambilan sampel pertama")
    qc2 = c2.text_input("QC Point 2", "-4.613449573516846,105.280639648437", help="Koordinat pengambilan sampel kedua")
    qc3 = c3.text_input("QC Point 3", "-4.61384391784668,105.28195190429688", help="Koordinat pengambilan sampel ketiga")

    st.markdown('<p style="color:#8892a4; font-size:0.8rem; font-weight:600; margin: 0.5rem 0 0.3rem;">📡 Metadata Perangkat</p>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    acc = d1.number_input("Akurasi GPS (m)", value=1.0, step=0.1, help="Tingkat akurasi perangkat GNSS dalam meter")
    sat = d2.number_input("Jumlah Satelit", value=39, step=1, help="Jumlah satelit yang terkunci saat pengukuran")

    run_btn = st.button("🚀 Jalankan Prediksi QC", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if run_btn:
    try:
        l1, n1 = map(float, qc1.split(","))
        l2, n2 = map(float, qc2.split(","))
        l3, n3 = map(float, qc3.split(","))

        df = pd.DataFrame([{
            "SHAPE": shape_input, "AKURASI": acc, "JUMLAH_SATELIT": sat,
            "Latitude_1": l1, "Longitude_1": n1,
            "Latitude_2": l2, "Longitude_2": n2,
            "Latitude_3": l3, "Longitude_3": n3,
        }])

        df["geometry"] = df["SHAPE"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        gdf['GPS_PRECISION_SCORE'] = np.clip(100 - (gdf['AKURASI'] * 10), 0, 100)
        gdf_meter = gdf.to_crs(epsg=32748)
        centroids_meter = gdf_meter.geometry.centroid
        p_meters = []

        for i in range(1, 4):
            p_geom = [Point(xy) for xy in zip(gdf[f"Longitude_{i}"], gdf[f"Latitude_{i}"])]
            p_gdf = gpd.GeoDataFrame(geometry=p_geom, crs="EPSG:4326", index=gdf.index).to_crs(epsg=32748)
            p_meters.append(p_gdf.geometry)
            gdf[f"in_{i}"] = p_gdf.within(gdf_meter.geometry).astype(int)
            gdf[f"dist_c{i}"] = p_gdf.distance(centroids_meter)
            gdf[f"angle_{i}"] = [calculate_angle(c, p) for c, p in zip(centroids_meter, p_gdf.geometry)]

        gdf["all_inside"] = ((gdf["in_1"] == 1) & (gdf["in_2"] == 1) & (gdf["in_3"] == 1)).astype(int)
        gdf["move_angle_12"] = [calculate_angle(p1, p2) for p1, p2 in zip(p_meters[0], p_meters[1])]
        gdf["move_angle_23"] = [calculate_angle(p2, p3) for p2, p3 in zip(p_meters[1], p_meters[2])]

        features = [
            "in_1", "in_2", "in_3", "all_inside",
            "dist_c1", "dist_c2", "dist_c3",
            "angle_1", "angle_2", "angle_3",
            "move_angle_12", "move_angle_23",
            "GPS_PRECISION_SCORE", "JUMLAH_SATELIT",
        ]

        X_feat = gdf[features]
        prob = model.predict_proba(X_feat)[0][1]   # probabilitas kelas VALID
        THRESHOLD = 0.70
        pred = 1 if prob >= THRESHOLD else 0
        conf = max(prob, 1 - prob)

        # ── Hasil ditampilkan di bawah input (bukan di dalam col2) ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.markdown('<p class="section-label">Hasil Analisis</p>', unsafe_allow_html=True)

        # ── Result Banner ──
        if pred == 1:
            st.markdown(f"""
            <div class="result-valid">
                <div class="result-label" style="color:#00ff88;">✅ SAMPLING VALID</div>
                <div class="result-prob">
                    Probabilitas Valid: <b style="color:#00ff88;">{prob:.2%}</b>
                    &nbsp;·&nbsp; Threshold: <b style="color:#00ff88;">≥ 70%</b>
                    &nbsp;·&nbsp; Confidence: <b style="color:#00ff88;">{conf:.2%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-invalid">
                <div class="result-label" style="color:#ff6b35;">❌ SAMPLING TIDAK VALID</div>
                <div class="result-prob">
                    Probabilitas Valid: <b style="color:#ff6b35;">{prob:.2%}</b>
                    &nbsp;·&nbsp; Threshold: <b style="color:#ff6b35;">≥ 70%</b>
                    &nbsp;·&nbsp; Butuh: <b style="color:#ff6b35;">{max(0, 0.70 - prob):.2%} lagi</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Point Status + Visualisasi side by side ──
        res_left, res_right = st.columns([1, 1], gap="large")

        with res_left:
            st.markdown('<p style="font-size:0.82rem; color:#6b7a99; margin-bottom:0.6rem;">STATUS TITIK SAMPLING</p>', unsafe_allow_html=True)
            for i, label in enumerate(["QC-1", "QC-2", "QC-3"], 1):
                inside = gdf.iloc[0][f"in_{i}"] == 1
                badge  = "badge-inside" if inside else "badge-outside"
                status = "✔ Dalam Polygon" if inside else "✘ Luar Polygon"
                dist   = gdf.iloc[0][f"dist_c{i}"]
                angle  = gdf.iloc[0][f"angle_{i}"]
                st.markdown(f"""
                <div style="background:var(--bg-card2); border:1px solid var(--border); border-radius:10px;
                            padding:0.8rem 1rem; margin-bottom:0.5rem; display:flex;
                            justify-content:space-between; align-items:center;">
                    <span style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#e8edf5;">{label}</span>
                    <span class="point-badge {badge}">{status}</span>
                    <span style="font-size:0.72rem; color:#6b7a99;">{dist:.1f} m &nbsp;|&nbsp; {angle:.1f}°</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.82rem; color:#6b7a99; margin-bottom:0.4rem;">MOVEMENT ANGLES</p>', unsafe_allow_html=True)
            ma1, ma2 = st.columns(2)
            ma1.metric("QC1 → QC2", f"{gdf.iloc[0]['move_angle_12']:.2f}°")
            ma2.metric("QC2 → QC3", f"{gdf.iloc[0]['move_angle_23']:.2f}°")

        with res_right:
            st.markdown('<p style="font-size:0.82rem; color:#6b7a99; margin-bottom:0.4rem;">VISUALISASI SPASIAL</p>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#0d1526")

            poly = gdf.iloc[0].geometry
            if hasattr(poly, "exterior"):
                px, py = poly.exterior.xy
                ax.fill(px, py, alpha=0.15, color="#00d4ff")
                ax.plot(px, py, color="#00d4ff", linewidth=1.5, label="Polygon Petak")

            cx, cy = poly.centroid.x, poly.centroid.y
            ax.plot(cx, cy, marker="+", color="#ffffff", markersize=10, markeredgewidth=1.5, zorder=5)

            for i, (lon, lat, c, m, lbl) in enumerate(zip(
                [n1, n2, n3], [l1, l2, l3],
                ["#00ff88", "#ff6b35", "#8b5cf6"],
                ["o", "s", "^"],
                ["QC-1", "QC-2", "QC-3"]
            )):
                inside_i = gdf.iloc[0][f"in_{i+1}"] == 1
                ec = "#ffffff" if inside_i else "#ff4444"
                ax.scatter(lon, lat, color=c, edgecolors=ec, linewidths=1.2, s=90, zorder=6, marker=m, label=lbl)
                ax.annotate(lbl, (lon, lat), textcoords="offset points", xytext=(6, 6),
                            fontsize=7.5, color=c, fontfamily="monospace")

            ax.plot([n1, n2, n3], [l1, l2, l3], color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.35, zorder=4)
            ax.set_xlabel("Longitude", color="#6b7a99", fontsize=8)
            ax.set_ylabel("Latitude",  color="#6b7a99", fontsize=8)
            ax.tick_params(colors="#4a5568", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d45")
            ax.grid(True, color="#1e2d45", linewidth=0.5, alpha=0.7)
            ax.legend(loc="upper right", fontsize=7.5, facecolor="#111827",
                      edgecolor="#1e2d45", labelcolor="#b0bcd4")
            ax.set_title("Spatial Distribution of QC Points", color="#8892a4",
                         fontsize=9, pad=8, fontfamily="monospace")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)   # ← WAJIB: cegah memory leak & blank screen

        # ── Feature Table ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Computed Features & Diagnostics</p>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📋 Feature Values", "📐 Angle Analysis"])

        with tab1:
            st.markdown(render_feature_table(X_feat), unsafe_allow_html=True)

        with tab2:
            st.markdown('<p style="color:#6b7a99; font-size:0.8rem;">Sudut dari centroid ke setiap titik QC</p>', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            a1.metric("Angle QC-1", f"{gdf.iloc[0]['angle_1']:.2f}°")
            a2.metric("Angle QC-2", f"{gdf.iloc[0]['angle_2']:.2f}°")
            a3.metric("Angle QC-3", f"{gdf.iloc[0]['angle_3']:.2f}°")

    except Exception as e:
        import traceback
        st.error(f"⚠️ Terjadi kesalahan: **{e}**")
        with st.expander("🔍 Detail Error (untuk debugging)"):
            st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────
# MODEL DOCUMENTATION — COLLAPSIBLE
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()

with st.expander("📖 Model Documentation & Technical Report", expanded=False):
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value" style="color:#00ff88;">95.15%</div>
            <span class="metric-tag tag-excellent">Excellent</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1-Score (Macro)</div>
            <div class="metric-value" style="color:#00d4ff;">95.14%</div>
            <span class="metric-tag tag-stable">Highly Robust</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">ROC AUC</div>
            <div class="metric-value" style="color:#8b5cf6;">0.9931</div>
            <span class="metric-tag tag-superior">Superior</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Key Performance Summary")
    st.markdown("""
    Based on the evaluation results, the **XGBClassifier** demonstrated outstanding predictive performance
    and exceptional generalization on the **Test Set** — data the model had **never seen before**.
    Hyperparameter tuning adjusted the model's prediction behavior, optimizing *precision* to **95.96%**
    while maintaining a highly robust *recall* of **94.06%** for Class 1, with **ROC AUC reaching 0.9931**.
    The model shows **no bias toward a specific category**, confirmed by the near-identical macro-averaged
    precision and recall scores across both Class 0 and Class 1.
    """)

    st.markdown("### 2. Hyperparameter Tuning: Precision Optimization Shift")
    st.markdown("""
    Tuning via optimization yielded the following best parameter combination:
    - `colsample_bytree = 0.8`
    - `learning_rate = 0.1`
    - `max_depth = 7`
    - `n_estimators = 200`
    - `subsample = 0.8`

    Increasing estimators to 200 and applying feature/row sub-sampling significantly boosted validation
    accuracy (from 94.63% → **96.59%**). Global test accuracy stabilized at **95.15%**, indicating the
    model reached its performance ceiling on unseen data — a healthy sign of **true generalization**, not underfitting.
    While tuning did not change raw global accuracy on the test set, it refined the **internal calibration**
    of the model, enhanced **ROC AUC (0.9928 → 0.9931)**, and shifted the model into a more conservative,
    precision-optimized state — successfully minimizing **False Positives**.
    """)

    st.markdown("### 3. Technical Insights: Why Is Performance So High?")
    st.markdown("""
    <div class="insight-card"><span class="insight-icon">🎯</span><b>Projection Precision:</b>
    Transforming coordinates into the <b>UTM 48S (Meters)</b> system enabled highly accurate distance
    (<code>dist_c</code>) and geometric calculations for all spatial features.</div>
    <div class="insight-card"><span class="insight-icon">📐</span><b>Feature Quality:</b>
    High-signal features like <i>Movement Angles</i> and <i>Inside/Outside</i> polygon status provided
    clear mathematical separation that XGBoost's gradient-boosted trees exploit optimally.</div>
    <div class="insight-card"><span class="insight-icon">🛡️</span><b>Built-in Regularization (L1 & L2):</b>
    XGBoost's native regularization, combined with <code>subsample=0.8</code> and <code>max_depth=7</code>,
    penalizes overly complex trees — keeping test accuracy highly competitive and resilient against noise.</div>
    <div class="insight-card"><span class="insight-icon">📈</span><b>Sequential Boosting Power:</b>
    Unlike bagging methods (Random Forest, Extra Trees), XGBoost builds trees sequentially — each tree
    correcting the residual errors of its predecessor via gradient descent — delivering a highly precise
    and adaptive decision boundary.</div>
    """, unsafe_allow_html=True)

    st.markdown("### 4. Final Conclusion & Recommendations")
    st.info("""
    ✅ **Production-Ready.** The post-tuning XGBClassifier is the recommended deployment model.

    **Model Selection Trade-Off:**
    - Choose the **Pre-Tuning Model (Default)** if your priority is computational efficiency and an equal risk distribution between False Positives and False Negatives (Precision ≈ Recall ≈ 95.05%).
    - Choose the **Post-Tuning Model** if your use case strictly penalizes False Positives (Precision = 95.96%), or if you plan to manually calibrate classification probability thresholds in production.

    **Next Steps:**
    - **Error Analysis:** Manually inspect the remaining ~4.85% of misclassified instances — focus on data points located exactly on or near polygon boundaries where movement angles may become ambiguous (4 False Positives and 6 False Negatives in the test confusion matrix).
    - **Deployment:** Export the selected pipeline using `joblib` or `pickle` for direct integration into field applications or analytical pipelines.
    """)

    st.markdown("""
    <div style="margin-top: 1rem; padding: 1rem; background: rgba(0,212,255,0.05);
                border: 1px solid rgba(0,212,255,0.2); border-radius: 10px; font-size:0.85rem; color:#8892a4;">
    > <b>Final Note:</b> This project confirms that the regularized boosting mechanics of XGBoost are
    exceptionally well-suited for this geospatial dataset. Hyperparameter tuning successfully refined
    the model's inner calibration — delivering a <b>stable, dependable, and precision-optimized</b> deployment asset.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem; border-top: 1px solid rgba(0,212,255,0.1);">
    <span style="font-family: 'Space Mono', monospace; font-size:0.75rem; color:#3d4f6e;">
        🛰️ GeoValid v2.0 &nbsp;·&nbsp; Built by <b style="color:#00d4ff;">Muhammad Erico Ricardo</b>
        &nbsp;·&nbsp; XGBClassifier · UTM 48S · EPSG:32748
    </span>
</div>
""", unsafe_allow_html=True)
