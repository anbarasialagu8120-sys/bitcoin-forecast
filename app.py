import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Bitcoin Price Forecasting",
    page_icon="₿",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("₿ Bitcoin Multi-Horizon Price Forecasting")

st.write(
    "RNN / LSTM Based Bitcoin Price Forecasting"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select Forecast Horizon",
    ["1 Day", "3 Days", "7 Days"]
)


# =========================================================
# MODEL SELECTION
# =========================================================

if horizon == "1 Day":

    model_file = "rnn_1day.keras"
    history_file = "history_rnn_1d.pkl"
    forecast_days = 1
    selected_model_name = "RNN"

elif horizon == "3 Days":

    model_file = "lstm_3day.keras"
    history_file = "history_lstm_3d.pkl"
    forecast_days = 3
    selected_model_name = "LSTM"

else:

    model_file = "lstm_7day.keras"
    history_file = "history_lstm_7d.pkl"
    forecast_days = 7
    selected_model_name = "LSTM"


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model(model_file)

except Exception as e:

    st.error(
        f"❌ Model loading error: {e}"
    )

    st.stop()


# =========================================================
# LOAD SCALER
# =========================================================

try:

    scaler = joblib.load("scaler.pkl")

except Exception as e:

    st.error(
        f"❌ Scaler loading error: {e}"
    )

    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

st.info(
    f"Selected Horizon: **{horizon}** | "
    f"Model: **{model_file}**"
)


# =========================================================
# CSV UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload Bitcoin Historical CSV",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "👆 Upload your Bitcoin historical CSV "
        "to start forecasting."
    )

    st.stop()


# =========================================================
# READ CSV
# =========================================================

df = pd.read_csv(uploaded_file)

st.success(
    "✅ CSV uploaded successfully!"
)


# =========================================================
# CHECK DATE COLUMN
# =========================================================

if "Date" not in df.columns:

    st.error(
        "❌ 'Date' column not found in CSV."
    )

    st.stop()


# =========================================================
# DATE CONVERSION
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date"]
).copy()


# =========================================================
# 2015 FILTER
# =========================================================

original_rows = len(df)

df = df[
    df["Date"] >= pd.Timestamp("2015-01-01")
].copy()

removed_rows = (
    original_rows - len(df)
)


# =========================================================
# SORT DATE
# =========================================================

df = df.sort_values(
    by="Date",
    ascending=True
).reset_index(drop=True)


st.info(
    f"ℹ️ {removed_rows} rows before "
    f"2015-01-01 were excluded to match "
    f"the training data."
)


# =========================================================
# REQUIRED FEATURES
# =========================================================

features = [
    "Price",
    "Open",
    "High",
    "Low",
    "Vol.",
    "Change %"
]


# =========================================================
# CHECK FEATURES
# =========================================================

missing_columns = [
    col
    for col in features
    if col not in df.columns
]


if missing_columns:

    st.error(
        f"❌ Missing columns: {missing_columns}"
    )

    st.stop()


# =========================================================
# DATASET INFO
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        len(df)
    )

with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )

with col3:

    st.metric(
        "Forecast",
        horizon
    )


# =========================================================
# VALUE CONVERSION
# =========================================================

def convert_value(value):

    if pd.isna(value):

        return np.nan

    value = str(value).strip()

    value = value.replace(
        ",",
        ""
    )

    try:

        if value.upper().endswith("K"):

            return (
                float(value[:-1]) * 1000
            )

        elif value.upper().endswith("M"):

            return (
                float(value[:-1]) * 1000000
            )

        elif value.upper().endswith("B"):

            return (
                float(value[:-1]) * 1000000000
            )

        elif value.endswith("%"):

            return float(
                value[:-1]
            )

        else:

            return float(value)

    except:

        return np.nan


# =========================================================
# CONVERT FEATURES
# =========================================================

for col in features:

    df[col] = df[col].apply(
        convert_value
    )


# =========================================================
# REMOVE INF
# =========================================================

df[features] = df[
    features
].replace(
    [np.inf, -np.inf],
    np.nan
)


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

df[features] = df[
    features
].ffill()

df[features] = df[
    features
].bfill()


# =========================================================
# CHECK NULL
# =========================================================

if df[features].isnull().sum().sum() > 0:

    st.error(
        "❌ Some feature values could not "
        "be converted to numeric values."
    )

    st.stop()


# =========================================================
# REQUIRED ROWS
# =========================================================

required_rows = 60 + forecast_days


if len(df) < required_rows:

    st.error(
        f"❌ At least {required_rows} rows are required "
        f"for {horizon} validation."
    )

    st.stop()


# =========================================================
# SHOW RECENT DATA
# =========================================================

st.subheader(
    "📊 Historical Bitcoin Data"
)

st.dataframe(
    df.tail(10)[
        [
            "Date",
            "Price",
            "Open",
            "High",
            "Low",
            "Vol.",
            "Change %"
        ]
    ],
    use_container_width=True
)


# =========================================================
# LAST 60 + FUTURE VALIDATION ROWS
# =========================================================

validation_data = df.tail(
    required_rows
).copy()


# =========================================================
# FIRST 60 ROWS = MODEL INPUT
# =========================================================

input_60 = validation_data.iloc[
    :60
].copy()


# =========================================================
# NEXT FORECAST_DAYS = ACTUAL TARGETS
# =========================================================

actual_future = validation_data.iloc[
    60:60 + forecast_days
].copy()


# =========================================================
# CURRENT PRICE
# =========================================================

current_price = float(
    input_60["Price"].iloc[-1]
)

current_date = input_60[
    "Date"
].iloc[-1]


# =========================================================
# VALIDATION INFORMATION
# =========================================================

st.subheader(
    "🔍 Validation Input"
)

col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Current / 60th Row**"
    )

    st.dataframe(
        input_60.tail(1)[
            [
                "Date",
                "Price",
                "Open",
                "High",
                "Low",
                "Vol.",
                "Change %"
            ]
        ],
        use_container_width=True
    )


with col2:

    st.write(
        f"**Actual Next {forecast_days} Day(s)**"
    )

    st.dataframe(
        actual_future[
            [
                "Date",
                "Price",
                "Open",
                "High",
                "Low",
                "Vol.",
                "Change %"
            ]
        ],
        use_container_width=True
    )


# =========================================================
# SHOW EXACT 60 INPUT ROWS
# =========================================================

st.subheader(
    "📅 Latest 60 Rows Used for Prediction"
)

st.dataframe(
    input_60[
        [
            "Date",
            "Price",
            "Open",
            "High",
            "Low",
            "Vol.",
            "Change %"
        ]
    ],
    use_container_width=True
)


# =========================================================
# CHRONOLOGICAL CHECK
# =========================================================

if not input_60[
    "Date"
].is_monotonic_increasing:

    st.error(
        "❌ Dates are not in chronological order."
    )

    st.stop()

else:

    st.success(
        "✅ 60 input rows are in chronological order."
    )


# =========================================================
# INPUT CHECK
# =========================================================

st.subheader(
    "🔍 Input Check"
)

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Current Price (60th Row)",
        f"{current_price:,.2f}"
    )


with col2:

    st.metric(
        "Current Date",
        current_date.strftime(
            "%Y-%m-%d"
        )
    )


# =========================================================
# SCALER CHECK
# =========================================================

st.write(
    "Scaler expects features:",
    scaler.n_features_in_
)


if scaler.n_features_in_ != len(features):

    st.error(
        f"❌ Scaler expects "
        f"{scaler.n_features_in_} features, "
        f"but CSV has {len(features)} required features."
    )

    st.stop()


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

input_features = input_60[
    features
].copy()


# =========================================================
# SCALE INPUT
# =========================================================

try:

    scaled_data = scaler.transform(
        input_features
    )

except Exception as e:

    st.error(
        "❌ Scaler error"
    )

    st.error(
        str(e)
    )

    st.stop()


# =========================================================
# SHOW SCALED DATA
# =========================================================

st.write(
    "Scaled Latest Row:",
    scaled_data[-1]
)


# =========================================================
# MODEL INPUT
# =========================================================

X_input = scaled_data.reshape(
    1,
    60,
    6
)


st.write(
    "Model Input Shape:",
    X_input.shape
)


# =========================================================
# MODEL PREDICTION
# =========================================================

try:

    prediction = model.predict(
        X_input,
        verbose=0
    )

except Exception as e:

    st.error(
        "❌ Prediction error"
    )

    st.error(
        str(e)
    )

    st.stop()


# =========================================================
# DEBUG
# =========================================================

st.write(
    "Model Output Shape:",
    prediction.shape
)

st.write(
    "Scaled Prediction:",
    prediction.flatten()
)


# =========================================================
# GET ALL PREDICTIONS
# =========================================================

predictions_scaled = (
    prediction.flatten()
)


# =========================================================
# CHECK OUTPUT COUNT
# =========================================================

if len(predictions_scaled) != forecast_days:

    st.error(
        f"❌ Model returned "
        f"{len(predictions_scaled)} predictions, "
        f"but {forecast_days} were expected."
    )

    st.stop()


# =========================================================
# INVERSE TRANSFORM ALL PREDICTIONS
# =========================================================

inverse_input = np.zeros(
    (forecast_days, len(features))
)


# Price is first feature

inverse_input[:, 0] = (
    predictions_scaled
)


inverse_df = pd.DataFrame(
    inverse_input,
    columns=features
)


try:

    inverse_predictions = (
        scaler.inverse_transform(
            inverse_df
        )
    )

except Exception as e:

    st.error(
        "❌ Inverse scaling error"
    )

    st.error(
        str(e)
    )

    st.stop()


predicted_prices = (
    inverse_predictions[:, 0]
)


# =========================================================
# FORECAST SUMMARY
# =========================================================

st.subheader(
    "🎯 Forecast Summary"
)

final_predicted_price = float(
    predicted_prices[-1]
)

total_price_difference = (
    final_predicted_price -
    current_price
)

total_percentage_change = (
    total_price_difference /
    current_price
) * 100


summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)


with summary_col1:

    st.metric(
        "Current Price",
        f"{current_price:,.2f}"
    )


with summary_col2:

    st.metric(
        f"{horizon} Predicted Price",
        f"{final_predicted_price:,.2f}"
    )


with summary_col3:

    st.metric(
        "Expected Change",
        f"{total_percentage_change:.2f}%"
    )


with summary_col4:

    st.metric(
        "Model Used",
        selected_model_name
    )


st.write(
    f"📌 **Forecast Horizon:** {horizon}"
)

st.write(
    f"📌 **Forecast Model:** {selected_model_name}"
)

st.write(
    f"📌 **Current Date:** "
    f"{current_date.strftime('%Y-%m-%d')}"
)


# =========================================================
# ACTUAL PRICES
# =========================================================

actual_prices = (
    actual_future["Price"]
    .astype(float)
    .to_numpy()
)


# =========================================================
# ERROR CALCULATION
# =========================================================

absolute_errors = np.abs(
    actual_prices -
    predicted_prices
)


percentage_errors = (
    absolute_errors /
    actual_prices
) * 100


# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

comparison_df = pd.DataFrame({

    "Forecast Day": [
        f"Day {i}"
        for i in range(
            1,
            forecast_days + 1
        )
    ],

    "Date": actual_future[
        "Date"
    ].values,

    "Actual Price": actual_prices,

    "Predicted Price": predicted_prices,

    "Absolute Error": absolute_errors,

    "Percentage Error (%)":
        percentage_errors

})


# =========================================================
# FORECAST RESULT TABLE
# =========================================================

st.subheader(
    f"📅 {forecast_days}-Day Forecast Result"
)


st.dataframe(
    comparison_df.style.format({

        "Actual Price": "{:,.2f}",

        "Predicted Price": "{:,.2f}",

        "Absolute Error": "{:,.2f}",

        "Percentage Error (%)": "{:.2f}%"

    }),
    use_container_width=True
)


# =========================================================
# VALIDATION SUMMARY
# =========================================================

st.subheader(
    "📊 Prediction Validation"
)


average_absolute_error = (
    absolute_errors.mean()
)


average_percentage_error = (
    percentage_errors.mean()
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Average Absolute Error",
        f"{average_absolute_error:,.2f}"
    )


with col2:

    st.metric(
        "Average Percentage Error",
        f"{average_percentage_error:.2f}%"
    )


# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader(
    "📈 Actual vs Predicted Price"
)

graph_df = comparison_df[
    [
        "Date",
        "Actual Price",
        "Predicted Price"
    ]
].copy()


fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    graph_df["Date"],
    graph_df["Actual Price"],
    marker="o",
    label="Actual Price"
)

ax.plot(
    graph_df["Date"],
    graph_df["Predicted Price"],
    marker="o",
    label="Predicted Price"
)


ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b")
)

ax.set_xlabel("Date")
ax.set_ylabel("Bitcoin Price")

ax.legend()

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig)

# =========================================================
# TRAINING LOSS CURVE
# =========================================================

st.subheader(
    "📉 Training Loss Curve"
)


if os.path.exists(history_file):

    try:

        with open(
            history_file,
            "rb"
        ) as f:

            history = pickle.load(f)


        if "loss" in history:

            loss_df = pd.DataFrame({

                "Training Loss":
                    history["loss"]

            })


            if "val_loss" in history:

                loss_df["Validation Loss"] = (
                    history["val_loss"]
                )


            loss_df.index = np.arange(
                1,
                len(loss_df) + 1
            )

            loss_df.index.name = "Epoch"


            st.line_chart(
                loss_df
            )


            st.caption(
                f"Training history for {horizon} "
                f"({model_file})"
            )

        else:

            st.warning(
                "⚠️ Loss data was not found."
            )

    except Exception as e:

        st.warning(
            f"⚠️ Unable to load loss history: {e}"
        )

else:

    st.warning(
        f"⚠️ {history_file} not found. "
        "Loss curve cannot be displayed."
    )


# =========================================================
# HORIZON-WISE MODEL COMPARISON
# =========================================================

st.subheader(
    "📊 Horizon-wise Model Comparison"
)

st.write(
    "Performance comparison of the selected best model "
    "for each forecasting horizon."
)


# =========================================================
# YOUR BEST MODEL METRICS
# =========================================================

comparison_metrics = pd.DataFrame({

    "Horizon": [
        "1-Day",
        "3-Day",
        "7-Day"
    ],

    "Model": [
        "RNN",
        "LSTM",
        "LSTM"
    ],

    "MAE": [
        0.015496,
        0.0144600,
        0.020358
    ],

    "RMSE": [
        0.026361,
        0.0245545,
        0.030872
    ],

    "MAPE (%)": [
        2.975,
        np.nan,
        np.nan
    ],

    "R²": [
        0.975056,
        0.978066,
        0.964431
    ]

})


# =========================================================
# DISPLAY METRICS TABLE
# =========================================================

st.dataframe(
    comparison_metrics.style.format({

        "MAE": "{:.6f}",

        "RMSE": "{:.6f}",

        "MAPE (%)": "{:.2f}",

        "R²": "{:.6f}"

    }),
    use_container_width=True
)


# =========================================================
# COMPARISON BAR CHART - MAE
# =========================================================

st.write(
    "### MAE Comparison"
)

mae_chart = comparison_metrics[
    [
        "Horizon",
        "MAE"
    ]
].set_index(
    "Horizon"
)

st.bar_chart(
    mae_chart
)


# =========================================================
# COMPARISON BAR CHART - RMSE
# =========================================================

st.write(
    "### RMSE Comparison"
)

rmse_chart = comparison_metrics[
    [
        "Horizon",
        "RMSE"
    ]
].set_index(
    "Horizon"
)

st.bar_chart(
    rmse_chart
)


# =========================================================
# COMPARISON BAR CHART - R2
# =========================================================

st.write(
    "### R² Comparison"
)

r2_chart = comparison_metrics[
    [
        "Horizon",
        "R²"
    ]
].set_index(
    "Horizon"
)

st.bar_chart(
    r2_chart
)


# =========================================================
# RECENT PRICE TREND
# =========================================================

st.subheader(
    "📈 Recent Bitcoin Price Trend"
)


chart_data = input_60[
    [
        "Date",
        "Price"
    ]
].copy()


chart_data = chart_data.set_index(
    "Date"
)


st.line_chart(
    chart_data
)

# =========================================================
# OVERALL BITCOIN PRICE TREND
# =========================================================

st.subheader(
    "📊 Overall Historical Bitcoin Price Trend"
)

overall_chart = df[
    [
        "Date",
        "Price"
    ]
].copy()

overall_chart = overall_chart.set_index(
    "Date"
)

st.line_chart(
    overall_chart
)