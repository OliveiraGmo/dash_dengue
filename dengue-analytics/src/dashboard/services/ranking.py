import plotly.express as px
import streamlit as st


def show_ranking(df_rmvp, municipality_col, years):
    st.header("Top 10 municípios com mais casos")
    ano_ranking = st.selectbox("Selecione o ano para ranking", years)
    df_ranking = df_rmvp[df_rmvp["Ano"] == ano_ranking]
    df_ranking = df_ranking[
        (df_ranking[municipality_col].notnull())
        & (df_ranking[municipality_col] != "0")
        & (df_ranking[municipality_col] != "")
    ]
    top_municipios = df_ranking.groupby(municipality_col)["Total"].sum().reset_index()
    top_municipios = top_municipios.sort_values(by="Total", ascending=False).head(10)
    fig_ranking = px.bar(
        top_municipios,
        x=municipality_col,
        y="Total",
        labels={municipality_col: "Município", "Total": "Casos"},
        title=f"Top 10 municípios com mais casos em {ano_ranking}",
    )
    st.plotly_chart(fig_ranking, use_container_width=True)
