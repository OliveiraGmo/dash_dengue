import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from keras.layers import Dense
from keras.models import Sequential
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


@st.cache_resource(show_spinner="Treinando e prevendo modelos...")
def _cached_fit_all_models_with_future(df):
    return _fit_all_models_with_future(df)


def show_forecast(df_rmvp):
    st.subheader("Previsão de casos de dengue (Comparativo de Modelos )")
    df, _, test = _preprocess_and_split(df_rmvp)
    results, future_results, future_dates = _cached_fit_all_models_with_future(df)
    # Gráficos lado a lado agrupados por tipo
    classic_models = [
        ("Regressão Linear", "Regressão Linear"),
        ("SARIMA", "SARIMA"),
        ("Prophet", "Prophet"),
    ]
    ml_models = [
        ("Random Forest", "Random Forest"),
        ("XGBoost", "XGBoost"),
        ("MLP", "MLP (Neural Net)"),
        ("LSTM", "LSTM (Neural Net)"),
    ]

    st.markdown("### Modelos Clássicos")
    cols1 = st.columns(len(classic_models))
    for i, (model_key, model_label) in enumerate(classic_models):
        if model_key in results:
            with cols1[i]:
                if model_key == "LSTM":
                    _, lstm_pred = results[model_key]
                    plot_real_vs_pred(df, test, lstm_pred, model_label)
                else:
                    plot_real_vs_pred(df, test, results[model_key], model_label)

    st.markdown("### Modelos Machine Learning / Redes Neurais")
    cols2 = st.columns(len(ml_models))
    for i, (model_key, model_label) in enumerate(ml_models):
        if model_key in results:
            with cols2[i]:
                if model_key == "LSTM":
                    _, lstm_pred = results[model_key]
                    plot_real_vs_pred(df, test, lstm_pred, model_label)
                else:
                    plot_real_vs_pred(df, test, results[model_key], model_label)
    st.subheader("Previsão para 2025 (12 meses futuros)")
    _plot_future_chart(future_results, future_dates)


def plot_real_vs_pred(df, test, y_pred, model_name):
    """
    Plota gráfico com pontos reais (azul) e linha de previsão (vermelha) para o período de teste.
    df: DataFrame completo (com coluna 'data' e 'casos')
    test: DataFrame do período de teste
    y_pred: previsões do modelo para o período de teste
    model_name: nome do modelo para título
    """
    fig = go.Figure()
    # Pontos reais (azul)
    fig.add_trace(
        go.Scatter(
            x=test["data"],
            y=test["casos"],
            mode="markers",
            marker=dict(color="blue", size=7),
            name="Real",
        )
    )
    # Linha de previsão (vermelha)
    fig.add_trace(
        go.Scatter(
            x=test["data"], y=y_pred, mode="lines", line=dict(color="red", width=2), name="Previsto"
        )
    )
    fig.update_layout(
        title=f"Previsão de casos de dengue ({model_name})",
        xaxis_title="Ano",
        yaxis_title="Casos",
        legend_title="Legenda",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _preprocess_and_split(df_rmvp):
    meses_pt = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    df_long = df_rmvp.melt(
        id_vars=["Município de notificação", "Codigo Município", "Ano"],
        value_vars=meses_pt,
        var_name="mes",
        value_name="casos",
    )
    mes_num = {nome: str(i + 1).zfill(2) for i, nome in enumerate(meses_pt)}
    df_long["mes_num"] = df_long["mes"].map(mes_num)
    df_long["data"] = pd.to_datetime(df_long["Ano"].astype(str) + "-" + df_long["mes_num"] + "-01")
    df = df_long.groupby("data", as_index=False)["casos"].sum()
    # Filtra para datas até o mês corrente
    hoje = pd.Timestamp.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    df = df[df["data"] <= hoje]
    train = df.iloc[:-12] if len(df) > 12 else df.iloc[:0]
    test = df.iloc[-12:] if len(df) > 12 else df.copy()
    return df, train, test


def _fit_all_models_with_future(df):
    MODEL_LINEAR = "Regressão Linear"
    MODEL_RF = "Random Forest"
    MODEL_PROPHET = "Prophet"
    MODEL_XGB = "XGBoost"
    MODEL_SARIMA = "SARIMA"
    results = {}
    future_results = {}
    # Gera datas futuras para o ano de 2025 (jan-dez)
    future_dates = pd.date_range(start="2025-01-01", end="2025-12-01", freq="MS")

    # XGBoost
    xgb_df = df.copy()
    xgb_df["ano"] = xgb_df["data"].dt.year
    xgb_df["mes"] = xgb_df["data"].dt.month
    xgb_df["mes_sin"] = np.sin(2 * np.pi * xgb_df["mes"] / 12)
    xgb_df["mes_cos"] = np.cos(2 * np.pi * xgb_df["mes"] / 12)
    xgb_df["is_2024"] = (xgb_df["ano"] == 2024).astype(int)
    xgb_df["is_2025"] = (xgb_df["ano"] == 2025).astype(int)
    x_xgb = xgb_df[["mes_sin", "mes_cos", "is_2024", "is_2025"]]
    y_xgb = xgb_df["casos"]
    x_train_xgb, x_test_xgb = x_xgb.iloc[:-12], x_xgb.iloc[-12:]
    y_train_xgb = y_xgb.iloc[:-12]
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(x_train_xgb, y_train_xgb)
    xgb_pred = xgb.predict(x_test_xgb)
    results[MODEL_XGB] = xgb_pred
    # Previsão futura XGBoost
    future_xgb = pd.DataFrame(
        {
            "mes_sin": np.sin(2 * np.pi * future_dates.month / 12),
            "mes_cos": np.cos(2 * np.pi * future_dates.month / 12),
            "is_2024": (future_dates.year == 2024).astype(int),
            "is_2025": (future_dates.year == 2025).astype(int),
        }
    )
    future_results[MODEL_XGB] = xgb.predict(future_xgb)

    # Regressão Linear
    lr_df = df.copy()
    lr_df["mes"] = lr_df["data"].dt.month
    lr_df["ano"] = lr_df["data"].dt.year
    lr_df["mes_seq"] = (lr_df["ano"] - lr_df["ano"].min()) * 12 + (lr_df["mes"] - 1)
    x_lr = lr_df[["mes_seq"]]
    y_lr = lr_df["casos"]
    x_train_lr, x_test_lr = x_lr.iloc[:-12], x_lr.iloc[-12:]
    y_train_lr = y_lr.iloc[:-12]
    lr = LinearRegression()
    lr.fit(x_train_lr, y_train_lr)
    lr_pred = lr.predict(x_test_lr)
    results[MODEL_LINEAR] = lr_pred
    # Previsão futura Linear
    # Calcular mes_seq para 2025
    future_lr = pd.DataFrame(
        {"mes_seq": [((2025 - lr_df["ano"].min()) * 12 + i) for i in range(12)]}
    )
    future_results[MODEL_LINEAR] = lr.predict(future_lr)

    # Random Forest
    rf_df = df.copy()
    rf_df["ano"] = rf_df["data"].dt.year
    rf_df["mes"] = rf_df["data"].dt.month
    rf_df["mes_sin"] = np.sin(2 * np.pi * rf_df["mes"] / 12)
    rf_df["mes_cos"] = np.cos(2 * np.pi * rf_df["mes"] / 12)
    rf_df["is_2024"] = (rf_df["ano"] == 2024).astype(int)
    rf_df["is_2025"] = (rf_df["ano"] == 2025).astype(int)
    x_rf = rf_df[["ano", "mes", "mes_sin", "mes_cos", "is_2024", "is_2025"]]
    y_rf = rf_df["casos"]
    x_train_rf, x_test_rf = x_rf.iloc[:-12], x_rf.iloc[-12:]
    y_train_rf = y_rf.iloc[:-12]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train_rf, y_train_rf)
    rf_pred = rf.predict(x_test_rf)
    results[MODEL_RF] = rf_pred
    # Previsão futura RF
    future_rf = pd.DataFrame(
        {
            "ano": future_dates.year,
            "mes": future_dates.month,
            "mes_sin": np.sin(2 * np.pi * future_dates.month / 12),
            "mes_cos": np.cos(2 * np.pi * future_dates.month / 12),
            "is_2024": (future_dates.year == 2024).astype(int),
            "is_2025": (future_dates.year == 2025).astype(int),
        }
    )
    future_results[MODEL_RF] = rf.predict(future_rf)

    # SARIMA
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    train = df.iloc[:-12]
    test = df.iloc[-12:]
    sarima = SARIMAX(train["casos"], order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima.fit(disp=False)
    sarima_pred = sarima_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    results[MODEL_SARIMA] = sarima_pred
    # Previsão futura SARIMA (2025)
    # Calcular steps até 2025
    last_date = df["data"].max()
    months_to_2025 = (2025 - last_date.year) * 12 + (1 - last_date.month)
    steps = months_to_2025 + 12
    future_sarima = sarima_fit.get_forecast(steps=steps)
    # Pegar apenas os últimos 12 meses (2025)
    future_results[MODEL_SARIMA] = future_sarima.predicted_mean.values[-12:]

    # Prophet
    df_prophet = df.rename(columns={"data": "ds", "casos": "y"})
    train_p = df_prophet.iloc[:-12]
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet.fit(train_p)
    # Previsão futura Prophet (2025)
    future = prophet.make_future_dataframe(
        periods=(12 + (2025 - df_prophet["ds"].dt.year.max() - 1) * 12), freq="MS"
    )
    forecast = prophet.predict(future)
    # Pegar apenas 2025
    forecast_2025 = forecast[forecast["ds"].dt.year == 2025]["yhat"].values
    results[MODEL_PROPHET] = forecast.iloc[-12:]["yhat"].values
    future_results[MODEL_PROPHET] = forecast_2025

    # MLP (Rede Neural)
    mlp_df = df.copy()
    mlp_df["ano"] = mlp_df["data"].dt.year
    mlp_df["mes"] = mlp_df["data"].dt.month
    mlp_df["mes_sin"] = np.sin(2 * np.pi * mlp_df["mes"] / 12)
    mlp_df["mes_cos"] = np.cos(2 * np.pi * mlp_df["mes"] / 12)
    mlp_df["is_2024"] = (mlp_df["ano"] == 2024).astype(int)
    mlp_df["is_2025"] = (mlp_df["ano"] == 2025).astype(int)
    x_mlp = mlp_df[["ano", "mes", "mes_sin", "mes_cos", "is_2024", "is_2025"]]
    y_mlp = mlp_df["casos"]
    x_train_mlp, x_test_mlp = x_mlp.iloc[:-12], x_mlp.iloc[-12:]
    y_train_mlp = y_mlp.iloc[:-12]
    scaler = StandardScaler()
    x_train_mlp_scaled = scaler.fit_transform(x_train_mlp)
    x_test_mlp_scaled = scaler.transform(x_test_mlp)
    mlp = Sequential(
        [
            Dense(32, activation="relu", input_shape=(x_train_mlp_scaled.shape[1],)),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    mlp.compile(optimizer="adam", loss="mse")
    mlp.fit(x_train_mlp_scaled, y_train_mlp, epochs=200, verbose=0)
    y_pred_mlp = mlp.predict(x_test_mlp_scaled).flatten()
    results["MLP"] = y_pred_mlp
    # Previsão futura MLP
    future_mlp = pd.DataFrame(
        {
            "ano": future_dates.year,
            "mes": future_dates.month,
            "mes_sin": np.sin(2 * np.pi * future_dates.month / 12),
            "mes_cos": np.cos(2 * np.pi * future_dates.month / 12),
            "is_2024": (future_dates.year == 2024).astype(int),
            "is_2025": (future_dates.year == 2025).astype(int),
        }
    )
    future_mlp_scaled = scaler.transform(future_mlp)
    future_results["MLP"] = mlp.predict(future_mlp_scaled).flatten()

    # LSTM (Rede Neural Recorrente)
    from keras.layers import LSTM, Input
    from keras.models import Model

    n_lags = 12  # igual ao notebook
    series = df["casos"].values
    x_lstm, y_lstm = [], []
    for i in range(n_lags, len(series)):
        x_lstm.append(series[i - n_lags : i])
        y_lstm.append(series[i])
    x_lstm, y_lstm = np.array(x_lstm), np.array(y_lstm)
    # Separar treino/teste
    x_train_lstm, x_test_lstm = x_lstm[:-12], x_lstm[-12:]
    y_train_lstm = y_lstm[:-12]
    # Redimensionar para (amostras, timesteps, features)
    x_train_lstm = x_train_lstm.reshape((x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
    x_test_lstm = x_test_lstm.reshape((x_test_lstm.shape[0], x_test_lstm.shape[1], 1))
    # Definir modelo LSTM
    input_lstm = Input(shape=(n_lags, 1))
    x = LSTM(32, activation="tanh")(input_lstm)
    x = Dense(16, activation="relu")(x)
    output = Dense(1)(x)
    lstm_model = Model(inputs=input_lstm, outputs=output)
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(x_train_lstm, y_train_lstm, epochs=200, verbose=0)
    y_pred_lstm = lstm_model.predict(x_test_lstm).flatten()
    # Para alinhar datas do teste
    test_dates = df["data"].iloc[-12:].values
    results["LSTM"] = (test_dates, y_pred_lstm)
    # Previsão futura LSTM (recursiva para 2025)
    # Calcular steps até 2025
    last_date = df["data"].max()
    months_to_2025 = (2025 - last_date.year) * 12 + (1 - last_date.month)
    input_lstm_arr = np.concatenate([series, np.zeros(months_to_2025)])[-(n_lags + 12) : -12].copy()
    lstm_preds_2025 = []
    for i in range(12):
        pred = lstm_model.predict(input_lstm_arr.reshape(1, n_lags, 1)).flatten()[0]
        lstm_preds_2025.append(pred)
        input_lstm_arr = np.roll(input_lstm_arr, -1)
        input_lstm_arr[-1] = pred
    future_results["LSTM"] = lstm_preds_2025

    return results, future_results, future_dates


def _plot_future_chart(future_results, future_dates):
    MODE_LINE_MARKERS = "lines+markers"
    # Filtrar datas e valores para mostrar apenas 2025
    mask_2025 = future_dates.year == 2025
    future_dates_2025 = future_dates[mask_2025]
    fig = go.Figure()
    # Previsões futuras (apenas 2025)
    for model, y_pred in future_results.items():
        y_pred_2025 = (
            y_pred[-len(future_dates_2025) :] if len(y_pred) >= len(future_dates_2025) else y_pred
        )
        fig.add_trace(
            go.Scatter(x=future_dates_2025, y=y_pred_2025, mode=MODE_LINE_MARKERS, name=f"{model}")
        )
    fig.update_layout(
        title="Previsão dos Modelos para 2025",
        xaxis_title="Mês",
        yaxis_title="Casos de Dengue",
        legend_title="Modelo",
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(
            tickformat="%b",
            dtick="M1",
            tickmode="array",
            tickvals=future_dates_2025,
            ticktext=[d.strftime("%b") for d in future_dates_2025],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_comparative_chart(test, results):
    MODE_LINE_MARKERS = "lines+markers"
    MODEL_LINEAR = "Regressão Linear"
    MODEL_RF = "Random Forest"
    MODEL_SARIMA = "SARIMA"
    MODEL_PROPHET = "Prophet"
    MODEL_XGB = "XGBoost"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["data"], y=test["casos"], mode=MODE_LINE_MARKERS, name="Real"))
    if MODEL_LINEAR in results:
        fig.add_trace(
            go.Scatter(
                x=test["data"], y=results[MODEL_LINEAR], mode=MODE_LINE_MARKERS, name=MODEL_LINEAR
            )
        )
    if MODEL_RF in results:
        fig.add_trace(
            go.Scatter(x=test["data"], y=results[MODEL_RF], mode=MODE_LINE_MARKERS, name=MODEL_RF)
        )
    if MODEL_SARIMA in results:
        fig.add_trace(
            go.Scatter(
                x=test["data"], y=results[MODEL_SARIMA], mode=MODE_LINE_MARKERS, name=MODEL_SARIMA
            )
        )
    if MODEL_PROPHET in results:
        fig.add_trace(
            go.Scatter(
                x=test["data"], y=results[MODEL_PROPHET], mode=MODE_LINE_MARKERS, name=MODEL_PROPHET
            )
        )
    if MODEL_XGB in results:
        fig.add_trace(
            go.Scatter(x=test["data"], y=results[MODEL_XGB], mode=MODE_LINE_MARKERS, name=MODEL_XGB)
        )
    if "MLP" in results:
        fig.add_trace(
            go.Scatter(
                x=test["data"], y=results["MLP"], mode=MODE_LINE_MARKERS, name="MLP - Neural Net"
            )
        )
    if "LSTM" in results:
        lstm_dates, lstm_pred = results["LSTM"]
        fig.add_trace(
            go.Scatter(x=lstm_dates, y=lstm_pred, mode=MODE_LINE_MARKERS, name="LSTM - Neural Net")
        )
    fig.update_layout(
        title="Comparação dos Modelos no Período de Teste",
        xaxis_title="Data",
        yaxis_title="Casos de Dengue",
    )
    st.plotly_chart(fig, use_container_width=True)
