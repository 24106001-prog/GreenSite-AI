import io
import json
import math
from datetime import date, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
from shapely.geometry import Point, shape

st.set_page_config(page_title="GreenSite AI | Renewable Site Suitability", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.stApp {background:#f4f8f5} .block-container{padding-top:1.2rem;padding-bottom:2rem}
h1,h2,h3{color:#174d38}
[data-testid="stMetric"]{background:white;padding:14px;border-radius:12px;border:1px solid #dce9df}
section[data-testid="stSidebar"]{background:#eaf3ec}
</style>
""", unsafe_allow_html=True)

st.title("🌱 GreenSite AI")
st.subheader("Solar and Wind Site Suitability Engine")
st.caption("Solar and wind datasets + weighted site ranking + suitability heat map + exportable report")
st.info("This version has exactly two upload fields: one Solar CSV and one Wind CSV. Land, grid and road scores are illustrative placeholders. Candidate points are screening locations, not surveyed parcels.")

# ------------------------- HELPERS -------------------------
def safe_num(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def score_range(value, low, high):
    if pd.isna(value):
        return 0.0
    return round(float(np.clip((float(value) - low) / (high - low) * 100, 0, 100)), 1)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1, math.sqrt(a)))


@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "shortwave_radiation,wind_speed_10m,wind_speed_100m",
        "timezone": "Asia/Kolkata", "wind_speed_unit": "ms",
    }
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("hourly", {}).get("time"):
        raise ValueError("No hourly weather data returned for this location/date range.")
    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly = hourly.rename(columns={
        "time": "Time", "shortwave_radiation": "Solar radiation (W/m²)",
        "wind_speed_10m": "Wind at 10 m (m/s)", "wind_speed_100m": "Wind at 100 m (m/s)"
    })
    hourly["Date"] = hourly["Time"].dt.date
    # Hourly mean W/m² summed over hourly records / 1000 approximates daily kWh/m².
    solar = hourly.groupby("Date")["Solar radiation (W/m²)"].sum(min_count=1) / 1000
    wind = hourly.groupby("Date")[["Wind at 10 m (m/s)", "Wind at 100 m (m/s)"]].mean()
    daily = pd.concat([solar.rename("Solar energy (kWh/m²/day)"), wind], axis=1).reset_index()
    return hourly, daily


def load_point_csv(uploaded, label):
    """CSV must have latitude and longitude columns; optional name column."""
    if uploaded is None:
        return None
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"lat": "latitude", "lon": "longitude", "lng": "longitude", "long": "longitude"}
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError(f"{label} CSV needs latitude and longitude columns.")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if "name" not in df.columns:
        df["name"] = [f"{label} {i+1}" for i in range(len(df))]
    return df


def load_resource_csv(uploaded, label, accepted_value_columns):
    """Load a solar or wind CSV with latitude/longitude and a resource-value column."""
    if uploaded is None:
        return None, None
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    aliases = {"lat": "latitude", "lon": "longitude", "lng": "longitude", "long": "longitude"}
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError(f"{label} CSV needs latitude and longitude columns.")
    value_col = next((c for c in accepted_value_columns if c in df.columns), None)
    if value_col is None:
        raise ValueError(f"{label} CSV needs one resource column: {', '.join(accepted_value_columns)}.")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", value_col])
    if df.empty:
        raise ValueError(f"{label} CSV has no valid numeric rows.")
    return df, value_col


def nearest_resource_value(lat, lon, df, value_col):
    if df is None or df.empty or not value_col:
        return np.nan
    nearest = min(df.itertuples(index=False), key=lambda r: haversine_km(lat, lon, r.latitude, r.longitude))
    return safe_num(getattr(nearest, value_col))


def nearest_distance(lat, lon, points):
    if points is None or points.empty:
        return np.nan
    return min(haversine_km(lat, lon, r.latitude, r.longitude) for r in points.itertuples())


def load_land_geojson(uploaded):
    """Read a GeoJSON FeatureCollection of land-use polygons with optional land_score/land_use properties."""
    if uploaded is None:
        return []
    payload = json.load(uploaded)
    features = payload.get("features", [])
    parsed = []
    for feature in features:
        try:
            geom = shape(feature.get("geometry"))
            props = feature.get("properties") or {}
            parsed.append((geom, props))
        except Exception:
            continue
    if not parsed:
        raise ValueError("GeoJSON did not contain readable polygon features.")
    return parsed


def land_score_for_site(lat, lon, land_df, land_polygons):
    """Use a containing GeoJSON polygon when available; otherwise nearest point CSV score."""
    point = Point(lon, lat)  # GeoJSON coordinate order is longitude, latitude.
    for geom, props in land_polygons:
        if geom.geom_type in ("Polygon", "MultiPolygon") and geom.covers(point):
            category = str(props.get("land_use", props.get("class_name", props.get("class", "Mapped land use"))))
            raw_score = props.get("land_score", props.get("score", None))
            class_scores = {
                "barren": 90, "barren land": 90, "open land": 85, "scrub": 75,
                "grassland": 70, "agriculture": 55, "cropland": 55, "farmland": 55,
                "built-up": 20, "built up": 20, "urban": 15, "forest": 0,
                "water": 0, "wetland": 0, "protected": 0,
            }
            val = safe_num(raw_score, np.nan) if raw_score is not None else class_scores.get(category.strip().lower(), np.nan)
            if not pd.isna(val):
                return float(np.clip(val, 0, 100)), f"GeoJSON land-use polygon: {category}"
            return np.nan, f"Polygon matched ({category}) but no numeric land_score or recognized class was provided"
    if land_df is not None and not land_df.empty and "land_score" in land_df.columns:
        nearest = min(land_df.itertuples(), key=lambda r: haversine_km(lat, lon, r.latitude, r.longitude))
        val = safe_num(getattr(nearest, "land_score", np.nan))
        if not pd.isna(val):
            category = str(getattr(nearest, "land_use", getattr(nearest, "class_name", "Uploaded land-use sample")))
            return float(np.clip(val, 0, 100)), f"Nearest uploaded land-use sample: {category}"
    return np.nan, "No matching land-use polygon or scored point sample supplied"


def report_html(df, location_name, energy, start, end, weights, land_mode, grid_mode, road_mode):
    cols = ["Rank", "Site", "Score", "Solar average (kWh/m²/day)", "Wind at 100 m average (m/s)", "Land score", "Grid score", "Road score"]
    table = df[cols].to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>GreenSite AI Report</title>
    <style>body{{font-family:Arial,sans-serif;margin:32px;color:#24352b}}h1,h2{{color:#176b45}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d5e1d8;padding:7px;text-align:left}}th{{background:#eaf3ec}}.note{{background:#f3f7f4;padding:12px;border-left:4px solid #176b45}}</style></head><body>
    <h1>GreenSite AI — Site Suitability Report</h1><p><b>Location:</b> {location_name}<br><b>Energy type:</b> {energy}<br><b>Weather period:</b> {start} to {end}</p>
    <h2>Ranking</h2>{table}<h2>Weight settings</h2><ul><li>Resource: {weights[0]:.1f}%</li><li>Land: {weights[1]:.1f}%</li><li>Grid: {weights[2]:.1f}%</li><li>Road: {weights[3]:.1f}%</li></ul>
    <h2>Assumptions</h2><div class='note'><b>Important:</b> Land, grid and road scores are illustrative placeholders because this version accepts only two uploads (solar and wind datasets). Candidate points are generated screening locations, not verified land parcels. Check land ownership, zoning, protected areas, grid capacity, road access and on-site resource measurements before investment decisions.</div>
    <p>Weather fallback source: Open-Meteo Historical Weather API. Geocoding: OpenStreetMap Nominatim.</p></body></html>"""

# ------------------------- SIDEBAR INPUTS -------------------------
if "base_location" not in st.session_state:
    st.session_state.base_location = {"name": "Coimbatore, Tamil Nadu, India", "lat": 11.0168, "lon": 76.9558}

st.sidebar.header("1. Location")
place = st.sidebar.text_input("City / district / place in India", value=st.session_state.base_location["name"])
if st.sidebar.button("Search location"):
    if not place.strip():
        st.sidebar.error("Enter a location name.")
    else:
        try:
            loc = Nominatim(user_agent="GreenSiteAI_Suitability_App").geocode(f"{place.strip()}, India", timeout=20)
            if loc:
                st.session_state.base_location = {"name": loc.address, "lat": float(loc.latitude), "lon": float(loc.longitude)}
                st.rerun()
            else:
                st.sidebar.error("Location not found. Try a city or district name.")
        except Exception as exc:
            st.sidebar.error(f"Location search failed: {exc}")
base = st.session_state.base_location
st.sidebar.caption(f"Selected: {base['name']} ({base['lat']:.4f}, {base['lon']:.4f})")

st.sidebar.header("2. Analysis")
energy = st.sidebar.selectbox("Energy type", ["Solar", "Wind"])
latest = date.today() - timedelta(days=5)
def_start = date(max(2017, latest.year - 2), 1, 1)
def_end = min(date(latest.year - 1, 12, 31), latest)
start = st.sidebar.date_input("Historical start date", value=def_start, min_value=date(2017,1,1), max_value=latest)
end = st.sidebar.date_input("Historical end date", value=def_end, min_value=date(2017,1,1), max_value=latest)
if start > end:
    st.sidebar.error("Start date must be on or before end date.")
    st.stop()

st.sidebar.header("3. User-set weights")
st.sidebar.caption("Weights are normalized automatically. Set at least one weight above zero.")
w_resource = st.sidebar.slider("Solar / wind resource", 0, 100, 40)
w_land = st.sidebar.slider("Land suitability", 0, 100, 25)
w_grid = st.sidebar.slider("Substation proximity", 0, 100, 20)
w_road = st.sidebar.slider("Road accessibility", 0, 100, 15)
raw_weights = [w_resource, w_land, w_grid, w_road]
if sum(raw_weights) == 0:
    st.error("At least one weight must be greater than zero.")
    st.stop()
weights = [x / sum(raw_weights) for x in raw_weights]

st.sidebar.header("4. Upload resource datasets")
st.sidebar.caption("Upload only these two CSV files. Both need latitude and longitude columns.")
st.sidebar.markdown("**Solar CSV** — resource column can be `solar_irradiance` (W/m²), `solar_energy` (kWh/m²/day), or `shortwave_radiation` (W/m²).")
solar_file = st.sidebar.file_uploader("☀️ Upload Solar Dataset (CSV)", type=["csv"], key="solar_dataset")
st.sidebar.markdown("**Wind CSV** — resource column can be `wind_speed_100m` (m/s), `wind_speed` (m/s), or `wind_speed_10m` (m/s).")
wind_file = st.sidebar.file_uploader("💨 Upload Wind Dataset (CSV)", type=["csv"], key="wind_dataset")

try:
    solar_data, solar_value_col = load_resource_csv(solar_file, "Solar", ["solar_irradiance", "solar_energy", "shortwave_radiation", "solar_radiation"])
    wind_data, wind_value_col = load_resource_csv(wind_file, "Wind", ["wind_speed_100m", "wind_speed", "wind_speed_10m", "wind_speed_ms"])
except Exception as exc:
    st.sidebar.error(str(exc))
    st.stop()

# ------------------------- CANDIDATE SITES & WEATHER -------------------------
# Generated screening points; they are not cadastral parcels or verified land boundaries.
offsets = [("Site A", 0.000, 0.000), ("Site B", 0.045, 0.000), ("Site C", -0.045, 0.000), ("Site D", 0.000, 0.045), ("Site E", 0.000, -0.045), ("Site F", 0.032, 0.032)]
st.write(f"**Selected location:** {base['name']}  \n**Historical weather period:** {start.isoformat()} to {end.isoformat()}")
records, weather_daily = [], None
errors = []
with st.spinner("Loading Open-Meteo historical solar and wind data for candidate points..."):
    for site_name, lat_off, lon_off in offsets:
        lat, lon = base["lat"] + lat_off, base["lon"] + lon_off
        try:
            daily = None
            # Open-Meteo is used as fallback for whichever dataset was not uploaded.
            if solar_data is None or wind_data is None:
                _, daily = get_weather(round(lat, 5), round(lon, 5), start.isoformat(), end.isoformat())
            if solar_data is not None:
                solar_raw = nearest_resource_value(lat, lon, solar_data, solar_value_col)
                solar_avg = solar_raw * 24.0 / 1000.0 if solar_value_col in ["solar_irradiance", "shortwave_radiation", "solar_radiation"] else solar_raw
            else:
                solar_avg = float(daily["Solar energy (kWh/m²/day)"].mean())
            if wind_data is not None:
                wind_raw = nearest_resource_value(lat, lon, wind_data, wind_value_col)
                wind10_avg = wind_raw
                wind100_avg = wind_raw
            else:
                wind10_avg = float(daily["Wind at 10 m (m/s)"].mean())
                wind100_avg = float(daily["Wind at 100 m (m/s)"].mean())
            # Land/grid/road data are not upload inputs in this simplified two-file version.
            land_val = {"Site A": 85, "Site B": 80, "Site C": 50, "Site D": 40, "Site E": 65, "Site F": 72}[site_name]
            land_explanation = f"Illustrative placeholder score {land_val}/100; this version only accepts solar and wind files"
            grid_score = {"Site A": 90, "Site B": 70, "Site C": 85, "Site D": 60, "Site E": 75, "Site F": 68}[site_name]
            grid_explanation = f"Illustrative placeholder score {grid_score}/100; substation data is not uploaded in this version"
            road_score = {"Site A": 80, "Site B": 90, "Site C": 70, "Site D": 60, "Site E": 75, "Site F": 82}[site_name]
            road_explanation = f"Illustrative placeholder score {road_score}/100; road data is not uploaded in this version"
            sub_dist = np.nan
            road_dist = np.nan
            solar_score = score_range(solar_avg, 2, 7)
            wind_score = score_range(wind100_avg, 3, 10)
            resource_score = solar_score if energy == "Solar" else wind_score
            contribution_resource = resource_score * weights[0]
            contribution_land = land_val * weights[1]
            contribution_grid = grid_score * weights[2]
            contribution_road = road_score * weights[3]
            total = round(contribution_resource + contribution_land + contribution_grid + contribution_road, 1)
            records.append({
                "Site": site_name, "Latitude": lat, "Longitude": lon,
                "Solar average (kWh/m²/day)": solar_avg,
                "Wind at 10 m average (m/s)": wind10_avg,
                "Wind at 100 m average (m/s)": wind100_avg,
                "Solar resource score": solar_score, "Wind resource score": wind_score,
                "Resource score used": resource_score, "Land score": land_val,
                "Grid score": grid_score, "Road score": road_score,
                "Distance to nearest substation (km)": sub_dist,
                "Distance to nearest road point (km)": road_dist,
                "Resource contribution": contribution_resource,
                "Land contribution": contribution_land,
                "Grid contribution": contribution_grid,
                "Road contribution": contribution_road,
                "Score": total, "Land explanation": land_explanation,
                "Grid explanation": grid_explanation, "Road explanation": road_explanation,
            })
            if site_name == "Site A" and daily is not None:
                weather_daily = daily.copy()
        except Exception as exc:
            errors.append(f"{site_name}: {exc}")
if errors:
    st.warning("Some candidate weather requests failed:\n\n" + "\n".join(errors))
if not records:
    st.error("No site data could be calculated. Upload valid solar/wind CSV files or check internet access for Open-Meteo fallback.")
    st.stop()

df = pd.DataFrame(records).sort_values("Score", ascending=False).reset_index(drop=True)
df["Rank"] = np.arange(1, len(df) + 1)
best = df.iloc[0]
land_mode = "Illustrative placeholder scores (land-use upload not included in two-file version)"
grid_mode = "Illustrative placeholder scores (substation upload not included in two-file version)"
road_mode = "Illustrative placeholder scores (road upload not included in two-file version)"

# ------------------------- SUMMARY & RANKED LIST -------------------------
st.markdown("---")
st.header("Output 1 — Ranked Site List")
a,b,c,d = st.columns(4)
a.metric("Top-ranked site", best["Site"])
b.metric("Top suitability score", f"{best['Score']:.1f}/100")
c.metric("Candidate sites", len(df))
d.metric("Energy type", energy)
st.dataframe(df[["Rank", "Site", "Score", "Solar average (kWh/m²/day)", "Wind at 100 m average (m/s)", "Land score", "Grid score", "Road score"]].style.format({"Score":"{:.1f}", "Solar average (kWh/m²/day)":"{:.2f}", "Wind at 100 m average (m/s)":"{:.2f}", "Land score":"{:.1f}", "Grid score":"{:.1f}", "Road score":"{:.1f}"}), use_container_width=True, hide_index=True)
fig = px.bar(df.sort_values("Score"), x="Score", y="Site", orientation="h", color="Score", text="Score", range_color=[0,100], color_continuous_scale=["#d95f59", "#f2c14e", "#2e8b57"], labels={"Score":"Suitability score (0–100)"}, title="Candidate site comparison")
fig.update_layout(height=360, coloraxis_showscale=False, xaxis_range=[0,100])
st.plotly_chart(fig, use_container_width=True)

# ------------------------- HEAT MAP -------------------------
st.markdown("---")
st.header("Output 2 — Suitability Heat Map")
st.caption("Colored candidate points show each site's weighted score. This is a point-based screening heat map, not a continuous raster surface or a land parcel boundary map.")
map_obj = folium.Map(location=[base["lat"], base["lon"]], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
folium.Marker([base["lat"], base["lon"]], tooltip="Searched reference location", icon=folium.Icon(color="blue", icon="info-sign")).add_to(map_obj)
heat_points = [[float(r.Latitude), float(r.Longitude), float(r.Score)] for r in df.itertuples()]
HeatMap(heat_points, radius=35, blur=24, min_opacity=0.35, max_zoom=13).add_to(map_obj)
for _, r in df.iterrows():
    col = "green" if r["Rank"] == 1 else ("orange" if r["Score"] >= 60 else "red")
    folium.CircleMarker([r["Latitude"], r["Longitude"]], radius=8, color=col, fill=True, fill_opacity=0.9,
        tooltip=f"{r['Site']}: {r['Score']:.1f}/100 (Rank {int(r['Rank'])})",
        popup=f"<b>{r['Site']}</b><br>Rank: {int(r['Rank'])}<br>Score: {r['Score']:.1f}/100<br>Solar: {r['Solar average (kWh/m²/day)']:.2f} kWh/m²/day<br>Wind at 100m: {r['Wind at 100 m average (m/s)']:.2f} m/s").add_to(map_obj)
map_obj.fit_bounds([[base["lat"], base["lon"]]] + [[float(r.Latitude), float(r.Longitude)] for r in df.itertuples()], padding=(25,25))
st_folium(map_obj, height=520, use_container_width=True, key="greensite_suitability_heatmap")

# ------------------------- PER-SITE EXPLANATION -------------------------
st.markdown("---")
st.header("Output 3 — Per-site Score Explanation")
st.caption("Weighted contribution = criterion score × normalized weight. Contributions add up to the overall score.")
with st.expander("Show current normalized weights"):
    st.write({"Resource": f"{weights[0]*100:.1f}%", "Land": f"{weights[1]*100:.1f}%", "Grid": f"{weights[2]*100:.1f}%", "Road": f"{weights[3]*100:.1f}%"})
for _, r in df.iterrows():
    with st.expander(f"Rank {int(r['Rank'])}: {r['Site']} — {r['Score']:.1f}/100"):
        lcol, rcol = st.columns([1,1])
        with lcol:
            st.write(f"**Solar resource:** {r['Solar resource score']:.1f}/100 from {r['Solar average (kWh/m²/day)']:.2f} kWh/m²/day")
            st.write(f"**Wind resource:** {r['Wind resource score']:.1f}/100 from {r['Wind at 100 m average (m/s)']:.2f} m/s at 100 m")
            st.write(f"**Land:** {r['Land score']:.1f}/100 — {r['Land explanation']}")
            st.write(f"**Grid:** {r['Grid score']:.1f}/100 — {r['Grid explanation']}")
            st.write(f"**Road:** {r['Road score']:.1f}/100 — {r['Road explanation']}")
        with rcol:
            comp = pd.DataFrame({"Criterion":["Resource", "Land", "Grid", "Road"], "Weighted contribution":[r['Resource contribution'], r['Land contribution'], r['Grid contribution'], r['Road contribution']]})
            st.bar_chart(comp.set_index("Criterion"))
            st.write(f"**Final score:** {r['Score']:.1f}/100")
            st.write(f"**Coordinates:** {r['Latitude']:.5f}, {r['Longitude']:.5f}")

# ------------------------- WEATHER TRENDS -------------------------
st.markdown("---")
st.header("Historical Weather Trends")
if weather_daily is not None:
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Daily solar energy — Site A reference point**")
        st.line_chart(weather_daily.set_index("Date")["Solar energy (kWh/m²/day)"])
    with rcol:
        st.markdown("**Daily mean wind speed — Site A reference point**")
        st.line_chart(weather_daily.set_index("Date")[["Wind at 10 m (m/s)", "Wind at 100 m (m/s)"]])
    with st.expander("View daily weather table"):
        st.dataframe(weather_daily, use_container_width=True, hide_index=True)

# ------------------------- EXPORTABLE REPORT -------------------------
st.markdown("---")
st.header("Output 4 — Exportable Report")
st.write("Download the ranked results as CSV, the full explanation table as CSV, or an HTML report that can be opened in a browser and printed to PDF.")
rank_cols = ["Rank", "Site", "Latitude", "Longitude", "Score", "Solar average (kWh/m²/day)", "Wind at 10 m average (m/s)", "Wind at 100 m average (m/s)", "Solar resource score", "Wind resource score", "Land score", "Grid score", "Road score", "Distance to nearest substation (km)", "Distance to nearest road point (km)"]
st.download_button("⬇️ Download ranked sites CSV", df[rank_cols].to_csv(index=False).encode("utf-8-sig"), "greensite_ranked_sites.csv", "text/csv")
st.download_button("⬇️ Download per-site explanations CSV", df.to_csv(index=False).encode("utf-8-sig"), "greensite_site_explanations.csv", "text/csv")
if weather_daily is not None:
    st.download_button("⬇️ Download historical weather CSV", weather_daily.to_csv(index=False).encode("utf-8-sig"), "greensite_historical_weather.csv", "text/csv")
html = report_html(df, base["name"], energy, start.isoformat(), end.isoformat(), [x*100 for x in weights], land_mode, grid_mode, road_mode)
st.download_button("⬇️ Download HTML report (print/save as PDF)", html.encode("utf-8"), "greensite_suitability_report.html", "text/html")

st.markdown("---")
st.subheader("Data sources and assumptions")
st.markdown(f"""
- **Solar dataset:** {('User-uploaded CSV (' + solar_value_col + ')' if solar_data is not None else 'Open-Meteo Historical Weather API fallback')}.
- **Wind dataset:** {('User-uploaded CSV (' + wind_value_col + ')' if wind_data is not None else 'Open-Meteo Historical Weather API fallback')}.
- **Geocoding:** OpenStreetMap Nominatim.
- **Land, grid and road:** illustrative placeholder scores because this version intentionally has only two upload fields.
- **Scoring:** solar resource scaled across 2–7 kWh/m²/day; wind resource scaled across 3–10 m/s. These are prototype thresholds, not official approval standards.
- **Interpretation:** candidate sites are generated around the searched point and do not represent verified land parcels. Uploaded values are matched to the nearest coordinate in each CSV.
""")

st.warning("Before a real investment decision, validate land ownership and zoning, protected areas, terrain/slope, environmental restrictions, grid capacity, connection cost, road access and on-site resource measurements. The current map is a candidate-point heat map, not a parcel-level continuous GIS suitability raster.")
