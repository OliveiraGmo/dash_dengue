import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ===== Optional library imports and flags =====
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

# ===== Model evaluation helpers (added) =====

MONTH_MAP = {
    "Jan": 1,
    "Fev": 2,
    "Mar": 3,
    "Abr": 4,
    "Mai": 5,
    "Jun": 6,
    "Jul": 7,
    "Ago": 8,
    "Set": 9,
    "Out": 10,
    "Nov": 11,
    "Dez": 12,
}


def _build_monthly_series_from_df_rmvp(df_rmvp: pd.DataFrame) -> pd.DataFrame:
    """Constrói série mensal agregada (data, casos) a partir do df_rmvp já carregado na página principal."""
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    id_cols = [c for c in df_rmvp.columns if c not in months]
    df_long = df_rmvp.melt(id_vars=id_cols, value_vars=months, var_name="mes", value_name="casos")
    df_long["mes_num"] = df_long["mes"].map(MONTH_MAP).astype(int)
    df_long["Ano"] = df_long["Ano"].astype(int)
    df_long["data"] = pd.to_datetime(
        df_long["Ano"].astype(str) + "-" + df_long["mes_num"].astype(str) + "-01"
    )
    df = df_long.groupby("data", as_index=False)["casos"].sum().sort_values("data")
    df = df.dropna(subset=["casos"])
    return df[["data", "casos"]].reset_index(drop=True)


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(y_true - y_pred) / denom) * 100.0)


def _diebold_mariano(e1, e2, h=1, power=2):
    from scipy import stats

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if power == 1:
        d_t = np.abs(e1) - np.abs(e2)
    else:
        d_t = (e1**2) - (e2**2)
    d_mean = np.mean(d_t)
    T = d_t.shape[0]
    lag = max(h - 1, 0)
    gamma0 = np.var(d_t, ddof=1)
    cov = 0.0
    for j in range(1, lag + 1):
        cov_j = np.cov(d_t[j:], d_t[:-j], ddof=1)[0, 1]
        cov += 2 * (1 - j / (lag + 1)) * cov_j
    S = gamma0 + cov
    if S <= 0:
        return float("inf"), 0.0
    dm_stat = d_mean / np.sqrt(S / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mes"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    first = df["data"].min()
    df["mes_seq"] = (
        (df["data"].dt.year - first.year) * 12 + (df["data"].dt.month - first.month)
    ).astype(int)
    return df


def _make_lag_matrix(series: np.ndarray, lags: int):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def _fit_predict_sarimax(train_y, steps):
    model = SARIMAX(
        train_y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(steps=steps), dtype=float)


def _fit_predict_prophet(train_df, periods):
    df_p = train_df.rename(columns={"data": "ds", "casos": "y"})[["ds", "y"]]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=periods, freq="MS", include_history=False)
    forecast = m.predict(future)
    return forecast["yhat"].to_numpy(dtype=float)


def _fit_predict_lr(train_df, test_df):
    tr = _add_calendar_features(train_df)
    te = _add_calendar_features(test_df)
    feats = ["mes", "ano", "mes_seq"]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[feats])
    Xte = scaler.transform(te[feats])
    ytr = tr["casos"].to_numpy(dtype=float)
    lr = LinearRegression()
    lr.fit(Xtr, ytr)
    return lr.predict(Xte).astype(float)


def _fit_predict_rf(train_df, test_df):
    tr = _add_calendar_features(train_df)
    te = _add_calendar_features(test_df)
    feats = ["mes", "ano", "mes_seq"]
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(tr[feats], tr["casos"].to_numpy(dtype=float))
    return rf.predict(te[feats]).astype(float)


def _fit_predict_xgb(train_df, test_df):
    tr = _add_calendar_features(train_df)
    te = _add_calendar_features(test_df)
    feats = ["mes", "ano", "mes_seq"]
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(tr[feats], tr["casos"].to_numpy(dtype=float))
    return xgb.predict(te[feats]).astype(float)


def _fit_predict_mlp(train_df, test_df, lags=12, epochs=120):
    y = train_df["casos"].to_numpy(dtype=float)
    Xtr, ytr = _make_lag_matrix(y, lags)
    if len(Xtr) == 0:
        raise RuntimeError("Dados insuficientes para MLP com os lags solicitados.")
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(lags,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr, ytr, epochs=epochs, batch_size=16, verbose=0)

    history = list(y)
    preds = []
    for _ in range(len(test_df)):
        if len(history) < lags:
            preds.append(np.nan)
            history.append(history[-1])
            continue
        x = np.array(history[-lags:], dtype=float).reshape(1, -1)
        pred = float(model.predict(x, verbose=0).ravel()[0])
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds, dtype=float)


def _fit_predict_lstm(train_df, test_df, lags=12, epochs=60):
    from sklearn.preprocessing import StandardScaler

    y = train_df["casos"].to_numpy(dtype=float)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()
    Xtr, ytr = _make_lag_matrix(y_scaled, lags)
    if len(Xtr) == 0:
        raise RuntimeError("Dados insuficientes para LSTM com os lags solicitados.")
    Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))

    model = Sequential()
    model.add(KerasLSTM(64, input_shape=(lags, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr, ytr, epochs=epochs, batch_size=16, verbose=0)

    history = list(y_scaled)
    preds_scaled = []
    for _ in range(len(test_df)):
        if len(history) < lags:
            preds_scaled.append(np.nan)
            history.append(history[-1])
            continue
        x = np.array(history[-lags:], dtype=float).reshape(1, lags, 1)
        pred = float(model.predict(x, verbose=0).ravel()[0])
        preds_scaled.append(pred)
        history.append(pred)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
    return preds.astype(float)


# ===== Original function (kept signature) + new section =====


def show_evolution(df_rmvp, municipality_col, municipalities):
    # --- Parte original: evolução por município ---
    st.header("Evolução temporal dos casos por município e região")
    df_mun_ano = df_rmvp.groupby([municipality_col, "Ano"])["Total"].sum().reset_index()
    municipio_evol = st.selectbox("Selecione o município para evolução temporal", municipalities)
    df_evol = df_mun_ano[df_mun_ano[municipality_col] == municipio_evol]
    fig_evol = px.line(
        df_evol, x="Ano", y="Total", markers=True, title=f"Evolução dos casos em {municipio_evol}"
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    # --- Nova seção: Avaliação de Modelos (hold-out) ---
    st.markdown("---")
    st.subheader("📊 Avaliação de Modelos (hold-out)")

    # Parâmetros
    col1, col2 = st.columns(2)
    with col1:
        test_months = st.number_input(
            "Meses no conjunto de teste", min_value=6, max_value=60, value=12, step=1
        )
    with col2:
        incluir_opcionais = st.multiselect(
            "Incluir modelos opcionais (depende das libs instaladas)",
            options=["Prophet", "XGBoost", "MLP", "LSTM"],
            default=["Prophet", "XGBoost"],
        )

    with st.spinner("Treinando e comparando modelos..."):
        # Constrói série mensal a partir do df_rmvp já carregado
        df_series = (
            _build_monthly_series_from_df_rmvp(df_rmvp).sort_values("data").reset_index(drop=True)
        )
        if test_months <= 0 or test_months >= len(df_series):
            st.error(
                "Meses de teste inválido — ajuste o valor (deve ser > 0 e menor que o total de meses)."
            )
            return

        train = df_series.iloc[: -int(test_months)].copy()
        test = df_series.iloc[-int(test_months) :].copy()

        results = {}
        # Sempre tentamos SARIMAX + LR + RF
        try:
            results["SARIMAX"] = _fit_predict_sarimax(train["casos"], steps=len(test))
        except Exception as e:
            st.warning(f"SARIMAX falhou: {e}")
        try:
            results["LinearRegression"] = _fit_predict_lr(train, test)
        except Exception as e:
            st.warning(f"LinearRegression falhou: {e}")
        try:
            results["RandomForest"] = _fit_predict_rf(train, test)
        except Exception as e:
            st.warning(f"RandomForest falhou: {e}")

        # Opcionais
        if "Prophet" in incluir_opcionais:
            try:
                results["Prophet"] = _fit_predict_prophet(train, periods=len(test))
            except Exception as e:
                st.warning(f"Prophet falhou: {e}")
        if "XGBoost" in incluir_opcionais:
            try:
                results["XGBoost"] = _fit_predict_xgb(train, test)
            except Exception as e:
                st.warning(f"XGBoost falhou: {e}")
        if "MLP" in incluir_opcionais:
            try:
                results["MLP"] = _fit_predict_mlp(train, test, lags=12, epochs=120)
            except Exception as e:
                st.warning(f"MLP falhou: {e}")
        if "LSTM" in incluir_opcionais:
            try:
                results["LSTM"] = _fit_predict_lstm(train, test, lags=12, epochs=60)
            except Exception as e:
                st.warning(f"LSTM falhou: {e}")

        # Métricas
        rows = []
        preds_dict = {}
        for name, yhat in results.items():
            y_true = test["casos"].to_numpy(dtype=float)
            y_pred = np.asarray(yhat, dtype=float)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape_v = _mape(y_true, y_pred)
            smape_v = _smape(y_true, y_pred)
            rows.append(
                {
                    "model": name,
                    "n_test": len(test),
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "MAPE_%": mape_v,
                    "sMAPE_%": smape_v,
                }
            )
            preds_dict[name] = pd.DataFrame(
                {"data": test["data"], "y_true": y_true, "y_pred": y_pred}
            )

        if rows:
            metrics_df = pd.DataFrame(rows).set_index("model").sort_values("RMSE")

            # === BLOCO PRINCIPAL DE RESULTADOS (tabelas lado a lado, largura total) ===
            st.markdown(
                """
                <div style='padding-top: 40px;'>
                    <h3 style='color:#ffffff;'>🔍 Avaliação comparativa de desempenho</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Cria layout com duas colunas que ocupam toda a largura, como os gráficos acima
            col1, col2 = st.columns(2, gap="large")

            # === Tabela da esquerda: Métricas por modelo ===
            with col1:
                st.markdown("### 📋 Métricas por modelo (ordenado por RMSE)")
                st.dataframe(metrics_df.reset_index(), use_container_width=True, height=280)

            # === Tabela da direita: Teste Diebold–Mariano ===
            with col2:
                st.markdown("### 📊 Teste Diebold–Mariano (pares)")
                dm_rows = []
                names = list(results.keys())

                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = names[i], names[j]
                        y_true = preds_dict[a]["y_true"].to_numpy(dtype=float)
                        e1 = y_true - preds_dict[a]["y_pred"].to_numpy(dtype=float)
                        e2 = y_true - preds_dict[b]["y_pred"].to_numpy(dtype=float)
                        dm_stat, dm_p = _diebold_mariano(e1, e2, h=1, power=2)
                        dm_rows.append(
                            {"Modelo A": a, "Modelo B": b, "DM Stat": dm_stat, "p-valor": dm_p}
                        )

                if dm_rows:
                    dm_df = pd.DataFrame(dm_rows).sort_values("p-valor")
                    st.dataframe(dm_df, use_container_width=True, height=280)
                else:
                    st.info("Sem resultados de DM (talvez apenas um modelo disponível).")

            # === GRÁFICO ABAIXO DAS TABELAS ===
            st.markdown("---")
            st.markdown("### 📈 Comportamento Real vs Previsões (período de teste)")

            import plotly.graph_objects as go

            fig_compare = go.Figure()

            fig_compare.add_trace(
                go.Scatter(
                    x=test["data"],
                    y=test["casos"],
                    mode="lines+markers",
                    name="Verdade",
                    line=dict(width=3, color="#00FFAA"),
                    marker=dict(size=6),
                )
            )

            colors = [
                "#FF4B4B",
                "#FFD166",
                "#06D6A0",
                "#118AB2",
                "#8ECAE6",
                "#C77DFF",
                "#F72585",
                "#43AA8B",
                "#FFA600",
                "#4CC9F0",
            ]

            for idx, (name, dfm) in enumerate(preds_dict.items()):
                fig_compare.add_trace(
                    go.Scatter(
                        x=dfm["data"],
                        y=dfm["y_pred"],
                        mode="lines+markers",
                        name=name,
                        line=dict(width=2, color=colors[idx % len(colors)]),
                        marker=dict(size=5),
                    )
                )

            fig_compare.update_layout(
                xaxis_title="Data",
                yaxis_title="Casos de Dengue",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFFFFF"),
                height=480,
                margin=dict(t=30, b=20, l=20, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12),
                ),
            )

            st.plotly_chart(fig_compare, use_container_width=True)
