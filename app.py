import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="ATMOS · Weather Intelligence",
    layout="wide",
    page_icon="🌦️",
    initial_sidebar_state="collapsed"
)

def load_css(path):
    css = Path(path).read_text(encoding="utf-8")
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)

load_css("style.css")

@st.cache_resource
def load_ann_model():
    return load_model("weather_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model  = load_ann_model()
scaler = load_scaler()

# Correct order matching training label encoding (alphabetical from get_dummies / LabelEncoder)
LABELS_RAW  = ["Cloudy", "Rainy", "Sunny", "Snowy"]
LABELS_ICON = ["☁️", "🌧️", "☀️", "❄️"]

TRAIN_COLUMNS = [
    "Cloud Cover_clear", "Cloud Cover_cloudy", "Cloud Cover_overcast",
    "Cloud Cover_partly cloudy", "Season_Autumn", "Season_Spring",
    "Season_Summer", "Season_Winter", "Location_coastal", "Location_inland",
    "Location_mountain", "Temperature", "Humidity", "Wind Speed",
    "Precipitation (%)", "Atmospheric Pressure", "UV Index", "Visibility (km)"
]

def predict_weather(inputs_dict):
    input_df = pd.DataFrame([inputs_dict])
    cat_cols = ["Cloud Cover", "Season", "Location"]
    encoded  = pd.get_dummies(input_df[cat_cols])
    input_df = input_df.drop(columns=cat_cols)
    final    = pd.concat([encoded, input_df], axis=1)
    final    = final.replace(True, 1).replace(False, 0)
    final    = final.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    scaled   = scaler.transform(final)
    preds    = model.predict(scaled, verbose=0)
    idx      = int(np.argmax(preds))
    return idx, preds[0]


# ─── HERO ──────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero'>"
    "<div class='hero-eyebrow'>Atmospheric Neural Intelligence</div>"
    "<h1 class='hero-title'>ATMOS</h1>"
    "<p class='hero-sub'>ANN-powered weather prediction &nbsp;·&nbsp; Real-time confidence analysis</p>"
    "<div class='hero-line'></div>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ─── LAYOUT ────────────────────────────────────────────────────────────────
left, spacer, right = st.columns([5, 0.4, 4])

with left:
    st.markdown("<div class='section-label'>📡 Input Parameters</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        temp       = st.number_input("🌡 Temperature (°C)",     value=25.0, step=0.5)
        wind       = st.number_input("🌬 Wind Speed (km/h)",     value=10.0, step=0.5)
        pressure   = st.number_input("🧭 Atmospheric Pressure", value=1013.0, step=0.5)
        uv         = st.number_input("UV Index",                 value=3.0,   step=0.1)
    with c2:
        humidity   = st.number_input("💧 Humidity (%)",          value=60.0, step=1.0)
        precip     = st.number_input("Precipitation (%)",        value=0.0,  step=1.0)
        visibility = st.number_input("👁 Visibility (km)",       value=10.0, step=0.5)

    st.markdown("<div class='my-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>🌍 Conditions</div>", unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        cloud    = st.selectbox("Cloud Cover", ["clear", "partly cloudy", "cloudy", "overcast"])
    with c4:
        season   = st.selectbox("Season",      ["Winter", "Spring", "Summer", "Autumn"])
    with c5:
        location = st.selectbox("Location",    ["coastal", "inland", "mountain"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Run Prediction")

    chips = (
        "<div class='chips-row'>"
        + "<span class='chip'>🌡 <strong>" + str(temp) + " C</strong></span>"
        + "<span class='chip'>💧 <strong>" + str(humidity) + "%</strong></span>"
        + "<span class='chip'>🌬 <strong>" + str(wind) + " km/h</strong></span>"
        + "<span class='chip'>🧭 <strong>" + str(pressure) + " hPa</strong></span>"
        + "<span class='chip'>UV <strong>" + str(uv) + "</strong></span>"
        + "<span class='chip'>👁 <strong>" + str(visibility) + " km</strong></span>"
        + "</div>"
    )
    st.markdown(chips, unsafe_allow_html=True)


with right:
    st.markdown("<div class='section-label'>🧠 Prediction Output</div>", unsafe_allow_html=True)

    if predict_clicked:
        try:
            inputs = {
                "Temperature": temp, "Humidity": humidity, "Wind Speed": wind,
                "Precipitation (%)": precip, "Atmospheric Pressure": pressure,
                "UV Index": uv, "Visibility (km)": visibility,
                "Cloud Cover": cloud, "Season": season, "Location": location
            }
            idx, probs = predict_weather(inputs)

            st.markdown(
                "<div class='result-card'>"
                + "<div class='result-icon'>" + LABELS_ICON[idx] + "</div>"
                + "<div class='result-label'>" + LABELS_RAW[idx] + "</div>"
                + "<div class='result-sub'>Predicted Weather Type</div>"
                + "</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div class='section-label' style='margin-top:1.2rem'>📊 Confidence Breakdown</div>",
                unsafe_allow_html=True
            )

            bars_html = ""
            for prob, label, icon in sorted(zip(probs, LABELS_RAW, LABELS_ICON), reverse=True):
                pct = float(prob) * 100
                bars_html += (
                    "<div class='conf-row'>"
                    + "<div class='conf-name'>" + icon + " " + label + "</div>"
                    + "<div class='conf-track'>"
                    + "<div class='conf-fill' style='width:" + f"{pct:.1f}" + "%'></div>"
                    + "</div>"
                    + "<div class='conf-pct'>" + f"{pct:.1f}" + "%</div>"
                    + "</div>"
                )
            st.markdown(bars_html, unsafe_allow_html=True)

            top_pct = float(max(probs)) * 100
            st.markdown(
                "<div class='conf-note'>Model confidence: <strong>"
                + f"{top_pct:.1f}%"
                + "</strong> certainty on this prediction.</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error("Prediction failed: " + str(e))

    else:
        st.markdown(
            "<div class='idle-state'>"
            "<div class='big-icon'>🌐</div>"
            "<p>Configure atmospheric parameters<br>and run the neural prediction engine.</p>"
            "</div>",
            unsafe_allow_html=True
        )


# ─── FOOTER ────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>ATMOS · Powered by ANN + Streamlit · Meteorological Intelligence System</div>",
    unsafe_allow_html=True
)