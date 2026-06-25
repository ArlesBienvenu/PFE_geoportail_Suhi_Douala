"""
============================================================
GEOPORTAIL SUHI DOUALA (2000-2025) -- OS5
============================================================
Application Streamlit de visualisation interactive, en complement des
cartes statiques validees du memoire (Livrables_Etape4/).

Lancement local :
    streamlit run geoportail_SUHI_Douala.py
(a executer depuis C:\\SUHI_Douala_Etape2\\, pour que les dossiers
Livrables_Etape3 et Livrables_Etape4 soient trouves via les chemins
relatifs ci-dessous)

Deploiement Streamlit Community Cloud : voir le fichier
INSTRUCTIONS_DEPLOIEMENT.md fourni a part.
============================================================
"""

import os

# ------------------------------------------------------------------
# CORRECTIF PROJ : evite un conflit avec une installation PostgreSQL/PostGIS
# presente sur le systeme, qui peut imposer une variable d'environnement
# PROJ_LIB ou PROJ_DATA pointant vers une base proj.db incompatible avec
# celle utilisee par rasterio. On supprime ces variables AVANT tout import
# rasterio, pour forcer l'utilisation de la base PROJ livree avec rasterio.
# ------------------------------------------------------------------
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from affine import Affine
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# 0. CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Geoportail SUHI Douala",
    page_icon="🌡️",
    layout="wide",
)

DATA_DIR = "Livrables_Etape4"
DATA_DIR_ETAPE3 = "Livrables_Etape3"

PERIODES = ["2000_2004", "2005_2009", "2010_2014", "2015_2019", "2020_2025"]

# Seules les 3 couches les plus pertinentes sont proposees a l'utilisateur :
# Delta_T observe (preuve factuelle), Hotspots (conclusion statistique
# validee, basee sur le modele lisse), LULC (contexte d'occupation du sol).
COUCHES = {
    "Delta_T observe (mesure)": {
        "fichier": "SUHI_observe_{p}.tif",
        "type": "continu",
        "cmap": "RdBu_r", "vmin": -5, "vmax": 5,
        "label": "Delta_T (degC)",
    },
    "Hotspots SUHI (conclusion statistique)": {
        "fichier": "Hotspots_{p}.tif",
        "type": "categoriel",
        "couleurs": {-2: (33, 102, 172), -1: (146, 197, 222), 0: (240, 240, 240),
                     1: (253, 219, 199), 2: (178, 24, 43)},
        "noms": {-2: "Coldspot 99%", -1: "Coldspot 95%", 0: "Non significatif",
                 1: "Hotspot 95%", 2: "Hotspot 99%"},
        "alpha_classes": {0: 0},
    },
    "Occupation du sol (LULC)": {
        "fichier": "LULC_{p}.tif",
        "type": "categoriel",
        "couleurs": {0: (49, 130, 189), 1: (222, 45, 38), 2: (49, 163, 84), 3: (223, 194, 125)},
        "noms": {0: "Eau", 1: "Bati", 2: "Vegetation", 3: "Sol nu / autre"},
    },
}

FONDS_DE_CARTE = {
    "Satellite (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
    },
    "Plan (OpenStreetMap)": {
        "tiles": "OpenStreetMap",
        "attr": None,
    },
}

VARIABLES_MODELE = pd.DataFrame([
    {"Variable": "NDVI", "Type": "Spectral", "Description": "Indice de végétation (densité/santé de la végétation)"},
    {"Variable": "NDBI", "Type": "Spectral", "Description": "Indice de bâti (surfaces construites)"},
    {"Variable": "MNDWI", "Type": "Spectral", "Description": "Indice d'eau modifié (plans d'eau, estuaire)"},
    {"Variable": "Albedo", "Type": "Spectral", "Description": "Réflectance de surface (matériaux clairs/sombres)"},
    {"Variable": "densite_bati", "Type": "Morphologique", "Description": "Densité du bâti dans un voisinage de 150m"},
    {"Variable": "dist_centre_m", "Type": "Morphologique", "Description": "Distance au centre urbain de référence"},
    {"Variable": "texture_LST", "Type": "Morphologique", "Description": "Hétérogénéité spatiale locale de la température"},
])

# ============================================================
# 1. CHARGEMENT + REPROJECTION (EPSG:32632 -> EPSG:4326), mis en cache
# ============================================================

@st.cache_data(show_spinner="Chargement et reprojection de la couche...")
def charger_raster_wgs84(chemin_tif, decimation=1):
    with rasterio.open(chemin_tif) as src:
        data = src.read(1)
        if decimation > 1:
            data = data[::decimation, ::decimation]
            transform_src = src.transform * Affine.scale(decimation, decimation)
        else:
            transform_src = src.transform

        transform_dst, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", data.shape[1], data.shape[0], *src.bounds
        )
        dest = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=data, destination=dest,
            src_transform=transform_src, src_crs=src.crs,
            dst_transform=transform_dst, dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
        west, south, east, north = array_bounds(height, width, transform_dst)

    return dest, (south, west, north, east)


# ============================================================
# 2. CONVERSION EN IMAGE RGBA (fond transparent hors ROI)
# ============================================================

def vers_image_continu(array, cmap_nom, vmin, vmax):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_nom)
    valides = ~np.isnan(array)
    couleurs = cmap(norm(np.where(valides, array, vmin)))
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    rgba[..., :3] = (couleurs[..., :3] * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valides, 210, 0).astype(np.uint8)
    return rgba


def vers_image_categoriel(array, couleurs_dict, alpha_classes=None):
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    alpha_classes = alpha_classes or {}
    for code, (r, g, b) in couleurs_dict.items():
        masque = array == code
        alpha = alpha_classes.get(code, 210)
        rgba[masque] = [r, g, b, alpha]
    return rgba


# ============================================================
# 3. BARRE LATERALE
# ============================================================
with st.sidebar:
    st.header("🌡️ Geoportail SUHI Douala")
    st.caption("PFE Master Geomatique -- 2000-2025")
    st.markdown(
        "Visualisation interactive complementaire aux cartes statiques "
        "validees, presentees dans le memoire."
    )
    st.divider()

    fond_carte_nom = st.radio("🗺️ Fond de carte", list(FONDS_DE_CARTE.keys()))

    mode_perf = st.checkbox(
        "Mode performance (sous-echantillonnage)",
        value=False,
        help="A activer si l'affichage est lent (recommande sur deploiement en ligne)."
    )
    decimation = 4 if mode_perf else 1

    st.divider()
    st.caption("Saison etudiee : saison seche (Decembre - Fevrier).")
    st.caption("Donnees : Landsat 7/8/9, GEE, modele XGBoost (R² ≈ 0.49, validation spatiale et temporelle).")

# ============================================================
# 4. CORPS PRINCIPAL -- ONGLETS
# ============================================================
st.title("🌡️ Géoportail interactif -- SUHI Douala (2000–2025)")
st.markdown(
    "Explorez l'évolution spatio-temporelle de l'îlot de chaleur urbain de surface (SUHI) "
    "de Douala, par période de 5 ans, à partir des résultats du modèle de Machine Learning."
)

onglet_carte, onglet_stats = st.tabs(["🗺️ Carte interactive", "📊 Statistiques & indicateurs clés"])

# ------------------------------------------------------------
# ONGLET 1 -- CARTE INTERACTIVE
# ------------------------------------------------------------
with onglet_carte:
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col1:
        periode = st.selectbox("📅 Période", PERIODES, index=len(PERIODES) - 1)
    with col2:
        couche_nom = st.selectbox("🛰️ Couche à afficher", list(COUCHES.keys()))
    with col3:
        opacite = st.slider("🎚️ Opacité", 0.0, 1.0, 0.8, 0.05)

    config = COUCHES[couche_nom]
    chemin = os.path.join(DATA_DIR, config["fichier"].format(p=periode))

    if not os.path.exists(chemin):
        st.error(f"Fichier introuvable : `{chemin}`. Vérifiez que les livrables de l'Étape 4 "
                 f"sont bien présents dans le dossier `{DATA_DIR}`.")
    else:
        array, (south, west, north, east) = charger_raster_wgs84(chemin, decimation=decimation)

        if config["type"] == "continu":
            rgba = vers_image_continu(array, config["cmap"], config["vmin"], config["vmax"])
        else:
            rgba = vers_image_categoriel(array, config["couleurs"], config.get("alpha_classes"))

        centre_lat = (south + north) / 2
        centre_lon = (west + east) / 2

        fond = FONDS_DE_CARTE[fond_carte_nom]
        m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles=fond["tiles"],
            attr=fond["attr"] if fond["attr"] else "OpenStreetMap",
            name=fond_carte_nom,
            control=False,
        ).add_to(m)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[south, west], [north, east]],
            opacity=opacite,
            interactive=False,
            cross_origin=False,
        ).add_to(m)

        st_folium(m, width=None, height=620, returned_objects=[])

        if config["type"] == "categoriel":
            st.markdown("**Légende**")
            noms = config["noms"]
            cols_legende = st.columns(len(noms))
            for i, (code, nom) in enumerate(noms.items()):
                r, g, b = config["couleurs"][code]
                with cols_legende[i]:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<div style="width:16px;height:16px;background-color:rgb({r},{g},{b});'
                        f'border:1px solid #333;border-radius:3px;"></div>'
                        f'<span style="font-size:0.85em;">{nom}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption(f"Échelle : {config['label']} (de {config['vmin']} à {config['vmax']})")

    st.divider()
    st.caption(
        "⚠️ Note méthodologique : les périodes 2005-2009 et 2010-2014 portent la signature "
        "résiduelle de l'artefact SLC-off de Landsat 7. Les hotspots officiels du mémoire ne "
        "sont retenus que pour 2015-2019 et 2020-2025 (cf. discussion méthodologique)."
    )

# ------------------------------------------------------------
# ONGLET 2 -- STATISTIQUES & INDICATEURS CLES
# ------------------------------------------------------------
with onglet_stats:
    st.subheader("Indicateurs clés du modèle et de l'étude")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Période d'étude", "2000–2025", "26 ans")
    c2.metric("R² (XGBoost, validation spatiale)", "0,484")
    c3.metric("R² (validation temporelle)", "0,491")
    c4.metric("Indice de Moran (résidus)", "0,078", help="Proche de 0 = résidus spatialement aléatoires (bon signe)")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Croissance superficie bâtie", "+89%", "2000-2004 → 2020-2025")
    c6.metric("Résolution spatiale", "30 m", "Landsat 7/8/9")
    c7.metric("Tendance LST globale", "+0,15 °C/an")
    c8.metric("Saison étudiée", "Déc–Fév", "Saison sèche")

    st.divider()

    st.subheader("Variables explicatives utilisées par le modèle")
    st.dataframe(VARIABLES_MODELE, hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("Graphiques et figures clés")

    col_g1, col_g2 = st.columns(2)

    chemin_fig1 = os.path.join(DATA_DIR, "fig1_LST_serie_temporelle.png")
    with col_g1:
        if os.path.exists(chemin_fig1):
            st.image(chemin_fig1, caption="Série temporelle de la LST moyenne (2000-2025) — tendance +0,15°C/an",
                      use_container_width=True)
        else:
            st.info(f"Figure non trouvée : `{chemin_fig1}`")

    chemin_evolution = os.path.join(DATA_DIR, "evolution_superficie_bati.png")
    with col_g2:
        if os.path.exists(chemin_evolution):
            st.image(chemin_evolution, caption="Évolution de la superficie bâtie (classification LULC)",
                      use_container_width=True)
        else:
            st.info(f"Figure non trouvée : `{chemin_evolution}`")

    chemin_shap = os.path.join(DATA_DIR_ETAPE3, "shap_summary_plot.png")
    if os.path.exists(chemin_shap):
        st.image(chemin_shap,
                  caption="Importance et effet des variables explicatives (analyse SHAP, modèle XGBoost)",
                  width=700)
    else:
        st.info(f"Figure non trouvée : `{chemin_shap}`")