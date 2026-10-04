# -*- coding: utf-8 -*-
"""
خريطة نصيب الفرد من المساحات الخضراء - محافظة الإسكندرية
Streamlit app (same core stack as NEVIS_release.py:
streamlit, streamlit_extras, plotly.graph_objects, geopandas, pandas, numpy, json)

Run (Anaconda PowerShell Prompt, from this folder):
    streamlit run greenspace_app.py
"""

import json
import math
import re
from os import path as os_path

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st
from shapely import force_2d
from streamlit_extras.metric_cards import style_metric_cards

## SETUP
#region
st.set_page_config(page_title="المساحات الخضراء - الإسكندرية", page_icon="🌳", layout="wide",
                   initial_sidebar_state="collapsed")

dirname = os_path.dirname(os_path.abspath(__file__))
DATA_FILE = os_path.join(dirname, "Greenspaces_Alexandria.geojson")

# Main column to visualise (نصيب الفرد)
VALUE_COL = "نصيب"

# The shapefile export truncated the column names to 10 bytes, so we map them back
# to readable Arabic labels. Edit the labels/units here if they are not right.
COLUMN_LABELS = {
    "إسم_ا": "اسم الحي",
    "مساحة": "المساحة (كم²)",
    "عدد_س": "عدد السكان (ألف نسمة)",
    "حدائق": "الحدائق",
    "إجمال": "إجمالي المساحات الخضراء (ألف م²)",
    "نصيب": "نصيب الفرد من المساحات الخضراء (م²/فرد)",
    "نسبة": "النسبة (%)",
}
NAME_COL = "إسم_ا"
POPUP_COLS = [c for c in COLUMN_LABELS if c != NAME_COL]

# ---- Map size (tuned for a 15-inch laptop screen) ----
MAP_HEIGHT = 520          # pixels
MAP_WIDTH_GUESS = 1250    # approximate map width in pixels, used only to fit the zoom

# ---- Client (HCSR) colour palette: change the hex codes here to fine-tune ----
NAVY = "#0B2E59"          # titles, legend text
BLUE = "#1565C0"          # accents, card borders
LIGHT_BLUE = "#EAF2FB"    # light backgrounds
BORDER_BLUE = "#C9DCF2"
TEXT_DARK = "#1F2937"

# Map colours: green only
GREEN_SCALE = [
    [0.0, "#E5F5E0"],
    [0.25, "#A1D99B"],
    [0.5, "#41AB5D"],
    [0.75, "#238B45"],
    [1.0, "#00441B"],
]

# Satellite basemap (Esri World Imagery) + optional labels overlay
SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LABEL_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
ESRI_CREDIT = "Imagery © Esri, Maxar, Earthstar Geographics"

# Right-to-left layout, Arabic font, HCSR blues, compact laptop layout
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText, .stMetric, .stSelectbox, .stSlider,
    .stCheckbox, .stRadio, h1, h2, h3, h4, p, label, div[role="dialog"] {{
        font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
    }}
    .main .block-container, section[data-testid="stSidebar"], div[role="dialog"] {{
        direction: rtl; text-align: right;
    }}
    /* keep the map itself left-to-right so controls/colorbar are not mirrored */
    .stPlotlyChart, .js-plotly-plot {{ direction: ltr; }}

    /* compact page so everything fits on a laptop screen */
    .main .block-container, [data-testid="stMainBlockContainer"] {{
        padding-top: 2.4rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;
        max-width: 100%;
    }}
    [data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}

    /* title bar in HCSR blue */
    .app-title {{
        background: linear-gradient(90deg, {NAVY}, {BLUE});
        color: #FFFFFF; font-size: 1.2rem; font-weight: 700;
        padding: 0.45rem 1rem; border-radius: 8px; margin-bottom: 0.2rem;
    }}

    /* metric cards: compact, dark text on white, blue accent */
    [data-testid="stMetric"] {{ background-color: #FFFFFF !important; padding: 0.35rem 0.8rem !important; }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
        color: {NAVY} !important; font-size: 0.85rem !important; font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
        color: {TEXT_DARK} !important; font-size: 1.35rem !important; font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div {{
        color: {BLUE} !important; font-size: 0.8rem !important;
    }}
    [data-testid="stMetricDelta"] svg {{ display: none; }}

    /* expanders and sidebar in light blue */
    [data-testid="stExpander"] details {{ border: 1px solid {BORDER_BLUE}; border-radius: 8px; }}
    [data-testid="stExpander"] summary {{ background: {LIGHT_BLUE}; color: {NAVY}; font-weight: 600; }}
    section[data-testid="stSidebar"] {{ background: {LIGHT_BLUE}; }}

    /* popup table */
    .info-table {{ width: 100%; border-collapse: collapse; direction: rtl; }}
    .info-table td {{ padding: 6px 10px; border-bottom: 1px solid {BORDER_BLUE}; color: {TEXT_DARK}; }}
    .info-table td:first-child {{ font-weight: 600; width: 60%; color: {NAVY}; }}
    .highlight-row td {{ background: #E5F5E0; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)
#endregion

## READ CACHED DATA FUNCTIONS
#region
def to_number(value):
    """Convert strings like '56.58' or '30.26%' to float."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_name(value):
    """'_______[حى أول المنتزة]_______'  ->  'حى أول المنتزة'"""
    return re.sub(r"[_\[\]]+", " ", str(value)).strip()


@st.cache_data
def readgreenspacedata(file_path):
    gdf = gpd.read_file(file_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf.geometry.apply(force_2d)  # drop the Z values

    gdf[NAME_COL] = gdf[NAME_COL].apply(clean_name)
    for col in POPUP_COLS:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(to_number)

    gdf["id"] = gdf.index.astype(str)  # unique id used as plotly 'locations'
    return gdf
#endregion

## MAP FUNCTIONS
#region
def fit_zoom_center(bounds, width_px, height_px, padding=0.9):
    """Zoom level and centre that fit the data bounds inside a map of the given pixel size."""
    minx, miny, maxx, maxy = bounds

    def merc(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    def inv_merc(y):
        return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)

    zoom_x = math.log2(width_px * padding * 360 / (512 * (maxx - minx)))
    zoom_y = math.log2(height_px * padding * 2 * math.pi / (512 * (merc(maxy) - merc(miny))))
    center = dict(lat=inv_merc((merc(miny) + merc(maxy)) / 2), lon=(minx + maxx) / 2)
    return min(zoom_x, zoom_y), center


def fmt(value, col):
    if pd.isna(value):
        return "—"
    if col == "نسبة":
        return f"{value:.1f}%"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def info_table_html(row):
    rows = []
    for col in POPUP_COLS:
        css = ' class="highlight-row"' if col == VALUE_COL else ""
        rows.append(f"<tr{css}><td>{COLUMN_LABELS[col]}</td><td>{fmt(row[col], col)}</td></tr>")
    return f'<table class="info-table">{"".join(rows)}</table>'


def build_map(gdf, opacity, show_labels):
    geojson = json.loads(gdf[["id", "geometry"]].to_json())
    customdata = gdf[POPUP_COLS].to_numpy()

    hover = "<b>%{text}</b><br>"
    for i, col in enumerate(POPUP_COLS):
        if col == VALUE_COL:
            hover += f"<b>{COLUMN_LABELS[col]}: %{{customdata[{i}]:.2f}}</b><br>"
        else:
            hover += f"{COLUMN_LABELS[col]}: %{{customdata[{i}]:,}}<br>"
    hover += "<i>انقر لعرض التفاصيل</i><extra></extra>"

    fig = go.Figure(go.Choroplethmap(
        geojson=geojson,
        featureidkey="properties.id",
        locations=gdf["id"],
        z=gdf[VALUE_COL],
        colorscale=GREEN_SCALE,
        marker_opacity=opacity,
        marker_line_width=1.5,
        marker_line_color="white",
        text=gdf[NAME_COL],
        customdata=customdata,
        hovertemplate=hover,
        # Legend: white box inside the map, large dark text
        colorbar=dict(
            title=dict(text="<b>نصيب الفرد<br>(م²/فرد)</b>", side="top",
                       font=dict(size=15, color=NAVY, family="Cairo, Tahoma, sans-serif")),
            tickfont=dict(size=14, color=NAVY, family="Cairo, Tahoma, sans-serif"),
            tickformat=".1f",
            x=0.985, xanchor="right",
            y=0.5, yanchor="middle",
            len=0.65, thickness=20,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=NAVY, borderwidth=1,
            outlinecolor=NAVY, outlinewidth=1,
            xpad=12, ypad=12,
        ),
    ))

    layers = [dict(sourcetype="raster", source=[SATELLITE_TILES], below="traces")]
    if show_labels:
        layers.append(dict(sourcetype="raster", source=[LABEL_TILES]))

    zoomval, cent = fit_zoom_center(gdf.total_bounds, MAP_WIDTH_GUESS, MAP_HEIGHT)

    fig.update_layout(
        map_style="white-bg",
        map_layers=layers,
        map_zoom=zoomval,
        map_center=cent,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=MAP_HEIGHT,
        hoverlabel=dict(font_family="Cairo, Tahoma, sans-serif", font_size=13, align="right",
                        bgcolor="white", bordercolor=BLUE, font_color=TEXT_DARK),
        clickmode="event+select",
        uirevision=True,  # keep zoom/pan when the app reruns
        # Small Esri credit inside the bottom-left corner of the map
        annotations=[dict(
            text=ESRI_CREDIT, showarrow=False,
            xref="paper", yref="paper", x=0.005, y=0.005, xanchor="left", yanchor="bottom",
            font=dict(size=10, color=TEXT_DARK), bgcolor="rgba(255,255,255,0.7)", borderpad=2,
        )],
    )
    return fig


@st.dialog("تفاصيل الحي", width="large")
def show_popup(row):
    st.markdown(f"<h3 style='color:{NAVY};margin-top:0'>{row[NAME_COL]}</h3>", unsafe_allow_html=True)
    st.markdown(info_table_html(row), unsafe_allow_html=True)
#endregion

## APP
#region
if not os_path.exists(DATA_FILE):
    st.error(f"لم يتم العثور على الملف: {DATA_FILE}\n\nضع ملف Greenspaces_Alexandria.geojson في نفس مجلد هذا البرنامج.")
    st.stop()

gdf = readgreenspacedata(DATA_FILE)

# Sidebar controls (click the > arrow at the top to open)
with st.sidebar:
    st.header("إعدادات الخريطة")
    opacity = st.slider("شفافية المضلعات", 0.1, 1.0, 0.65, 0.05)
    show_labels = st.checkbox("إظهار أسماء الأماكن فوق صور الأقمار الصناعية", value=False)

st.markdown('<div class="app-title">🌳 نصيب الفرد من المساحات الخضراء — محافظة الإسكندرية</div>',
            unsafe_allow_html=True)

# Metric cards (boxes on top)
m1, m2, m3, m4 = st.columns(4)
best = gdf.loc[gdf[VALUE_COL].idxmax()]
worst = gdf.loc[gdf[VALUE_COL].idxmin()]
total_green = gdf["إجمال"].sum()
total_pop = gdf["عدد_س"].sum()
m1.metric("عدد الأحياء", f"{len(gdf)}")
m2.metric("المتوسط العام (م²/فرد)", f"{total_green / total_pop:.2f}" if total_pop else "—")
m3.metric("الأعلى (م²/فرد)", f"{best[VALUE_COL]:.2f}", best[NAME_COL], delta_color="off")
m4.metric("الأدنى (م²/فرد)", f"{worst[VALUE_COL]:.2f}", worst[NAME_COL], delta_color="off")
style_metric_cards(
    background_color="#FFFFFF",
    border_size_px=1,
    border_color=BORDER_BLUE,
    border_radius_px=8,
    border_left_color=BLUE,
    box_shadow=True,
)

# Selection state
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "last_popup_id" not in st.session_state:
    st.session_state.last_popup_id = None

# Full-width map
fig = build_map(gdf, opacity, show_labels)
event = st.plotly_chart(
    fig,
    key="green_map",
    on_select="rerun",
    selection_mode="points",
    config={"scrollZoom": True, "displaylogo": False},
    use_container_width=True,
)

# Work out which polygon was clicked (only the main trace, curve 0)
clicked_id = None
if event and event.selection and event.selection.points:
    for p in event.selection.points:
        if p.get("curve_number", 0) == 0:
            loc = p.get("location")
            if loc is None and p.get("point_index") is not None:
                loc = gdf["id"].iloc[p["point_index"]]
            clicked_id = str(loc)
            break

st.session_state.selected_id = clicked_id  # plotly dims the other polygons automatically

# Open the popup once per new click
if clicked_id is not None and clicked_id != st.session_state.last_popup_id:
    st.session_state.last_popup_id = clicked_id
    show_popup(gdf[gdf["id"] == clicked_id].iloc[0])
if clicked_id is None:
    st.session_state.last_popup_id = None

# Extra views below the map (collapsed so the map stays the focus)
with st.expander("ترتيب الأحياء حسب نصيب الفرد"):
    ranked = gdf.sort_values(VALUE_COL)
    bar_colors = [BLUE if i == st.session_state.selected_id else "#238B45" for i in ranked["id"]]
    bar = go.Figure(go.Bar(
        x=ranked[VALUE_COL],
        y=ranked[NAME_COL],
        orientation="h",
        marker_color=bar_colors,
        text=ranked[VALUE_COL].map(lambda v: f"{v:.2f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.2f} م²/فرد<extra></extra>",
    ))
    bar.update_layout(
        height=380,
        margin={"r": 10, "t": 10, "l": 10, "b": 30},
        xaxis_title="م²/فرد",
        yaxis=dict(side="right"),
        font=dict(family="Cairo, Tahoma, sans-serif", color=NAVY),
        template="plotly_white",
    )
    st.plotly_chart(bar, config={"displayModeBar": False}, use_container_width=True)

with st.expander("جدول البيانات الكامل"):
    table = gdf[[NAME_COL] + POPUP_COLS].rename(columns=COLUMN_LABELS)
    st.dataframe(table.sort_values(COLUMN_LABELS[VALUE_COL], ascending=False),
                 hide_index=True, use_container_width=True)
#endregion
