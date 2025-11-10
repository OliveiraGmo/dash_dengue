import plotly.express as px
import streamlit as st


def show_rmvp_analysis(df_rmvp, municipality_col, months):
    st.header("Análise RMVP (Região Metropolitana do Vale do Paraíba)")
    municipios_rmvp = sorted(df_rmvp[municipality_col].unique())
    col1, col2 = st.columns(2)
    with col1:
        ano_rmvp = st.selectbox("Ano (RMVP)", sorted(df_rmvp["Ano"].unique()))
    with col2:
        municipio_rmvp = st.selectbox("Município (RMVP)", municipios_rmvp)
    df_rmvp_filtrado = df_rmvp[
        (df_rmvp[municipality_col] == municipio_rmvp) & (df_rmvp["Ano"] == ano_rmvp)
    ]
    st.subheader(f"Casos mensais em {municipio_rmvp} ({ano_rmvp}) - RMVP")
    if not df_rmvp_filtrado.empty:
        casos_mensais_rmvp = df_rmvp_filtrado[months].sum().reset_index()
        casos_mensais_rmvp.columns = ["Mês", "Casos"]
        fig_rmvp = px.bar(
            casos_mensais_rmvp, x="Mês", y="Casos", title="Casos Mensais de Dengue - RMVP"
        )

        # Mostrar gráfico mensal e resumo do município lado-a-lado para melhor alinhamento
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(fig_rmvp, use_container_width=True)
        with col_table:
            st.markdown("**Resumo do município e ano selecionados (RMVP)**")
            st.dataframe(df_rmvp_filtrado[[municipality_col, "Ano", "Total"]])
    else:
        st.info("Não há dados para o município e ano selecionados na RMVP.")

    st.subheader("Evolução anual dos casos na RMVP")
    df_rmvp_ano = df_rmvp.groupby("Ano")["Total"].sum().reset_index()
    fig_rmvp_ano = px.line(
        df_rmvp_ano, x="Ano", y="Total", markers=True, title="Casos Totais por Ano - RMVP"
    )
    # ocupar largura total, mas com altura controlada
    st.plotly_chart(fig_rmvp_ano.update_layout(height=380), use_container_width=True)
