import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_incidence_rmvp():
    st.header("Incidência de Dengue no RMVP de 2007 à 2025")
    dados = pd.read_csv("src/data/dados_incidencia_mensal.csv")
    meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    municipios = sorted(dados["Município de notificação"].unique())
    municipios_selecionados = st.multiselect(
        "Selecione os municípios para visualizar no gráfico:",
        municipios,
        default=municipios[:8],
        help="Selecione um ou mais municípios para análise comparativa.",
    )
    if not municipios_selecionados:
        st.warning("Selecione pelo menos um município para visualizar o gráfico.")
        return
    dados_filtrados = dados[dados["Município de notificação"].isin(municipios_selecionados)]

    fig = px.line(
        dados_filtrados,
        x="Ano",
        y=meses,
        color="Município de notificação",
        title="Incidência de Dengue no RMVP de 2007 à 2025",
    )
    fig.update_layout(xaxis_title="Ano", yaxis_title="Incidência")
    fig.add_trace(
        go.Scatter(
            x=[dados["Ano"].min(), dados["Ano"].max()],
            y=[100, 100],
            mode="lines",
            line=dict(color="green", width=2, dash="dash"),
            name="Média Incidência",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[dados["Ano"].min(), dados["Ano"].max()],
            y=[300, 300],
            mode="lines",
            line=dict(color="orange", width=2, dash="dash"),
            name="Alta Incidência",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[dados["Ano"].min(), dados["Ano"].max()],
            y=[1000, 1000],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Altíssima Incidência",
            showlegend=True,
        )
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
