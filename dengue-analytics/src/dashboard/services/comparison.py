import plotly.express as px
import streamlit as st


def show_comparison(df_rmvp, municipality_col, municipalities):
    st.header("Comparativo entre anos para municípios")
    municipalities_comp = st.multiselect(
        "Selecione municípios para comparar", municipalities, default=municipalities[:3]
    )
    df_comp = df_rmvp[df_rmvp[municipality_col].isin(municipalities_comp)]
    fig_comp = px.line(
        df_comp,
        x="Ano",
        y="Total",
        color=municipality_col,
        markers=True,
        title="Comparativo de casos anuais entre municípios",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
