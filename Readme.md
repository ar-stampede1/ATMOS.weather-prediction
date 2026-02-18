# 🌦️ ATMOS — Weather Type Prediction

A sleek, ANN-powered weather prediction web app built with **Streamlit** and **TensorFlow**. Enter atmospheric parameters and get an instant prediction of the weather type — Rainy, Cloudy, Sunny, or Snowy — with a full confidence breakdown.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)

---

## Features

- **ANN Model** trained on meteorological data
- **Real-time confidence breakdown** with animated bars
- **Glassmorphism UI** with neon cyan/indigo palette
- Supports 4 weather types: Rainy 🌧️, Cloudy ☁️, Sunny ☀️, Snowy ❄️

---

## Project Structure

```
weather_prediction_project/
├── app.py                  # Main Streamlit application
├── style.css               # Custom UI styling
├── weather_model.h5        # Trained ANN model
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/weather-prediction.git
cd weather-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploy on Streamlit Community Cloud (Free)

1. Push your project to a **public GitHub repository**
2. Make sure `requirements.txt` is present
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
4. Click **"New app"** → select your repo, branch, and set `app.py` as the main file
5. Click **Deploy** — your app will be live in ~2 minutes!

> **Note:** The `weather_model.h5` file must be committed to the repo. If it's large (>100MB), use [Git LFS](https://git-lfs.github.com/).

---

## Deploy on Render

1. Push project to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Deploy**

---

## Deploy on Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Choose **Streamlit** as the SDK
3. Upload your files: `app.py`, `style.css`, `weather_model.h5`, `requirements.txt`
4. The space will build and deploy automatically

---

## Input Parameters

| Parameter | Description |
|---|---|
| Temperature (°C) | Ambient air temperature |
| Humidity (%) | Relative humidity |
| Wind Speed (km/h) | Surface wind speed |
| Precipitation (%) | Chance of precipitation |
| Atmospheric Pressure (hPa) | Sea-level pressure |
| UV Index | Solar UV radiation index |
| Visibility (km) | Horizontal visibility |
| Cloud Cover | clear / partly cloudy / cloudy / overcast |
| Season | Winter / Spring / Summer / Autumn |
| Location | coastal / inland / mountain |

---

## Requirements

```
streamlit
tensorflow
pandas
numpy
```

---

## Model Details

- **Architecture:** Artificial Neural Network (ANN)
- **Input:** 18 features (numerical + one-hot encoded categoricals)
- **Output:** 4-class softmax (Rainy, Cloudy, Sunny, Snowy)
- **Format:** Keras `.h5`

---

## License

MIT License — free to use and modify.