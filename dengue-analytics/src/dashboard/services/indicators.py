import streamlit as st


def show_indicators(df_rmvp, municipality_col):
    df_validos = df_rmvp[
        (df_rmvp[municipality_col].notnull())
        & (df_rmvp[municipality_col] != "0")
        & (df_rmvp[municipality_col] != "")
    ]

    def format_value(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} milhões"
        elif value >= 1_000:
            return f"{value / 1_000:.0f} mil"
        else:
            return f"{value}"

    total_casos = int(df_validos["Total"].sum())
    media_anual = int(df_validos.groupby("Ano")["Total"].sum().mean())
    pico_ano = df_validos.groupby("Ano")["Total"].sum().idxmax()
    pico_valor = int(df_validos.groupby("Ano")["Total"].sum().max())
    muni_pico = df_validos.groupby(municipality_col)["Total"].sum().idxmax()
    muni_pico_valor = int(df_validos.groupby(municipality_col)["Total"].sum().max())
    st.markdown(
        """
        <style>
        .dashboard-bar {
            background: #2874A6;
            color: #fff;
            padding: 18px 0 10px 0;
            text-align: center;
            font-size: 2.2rem;
            font-weight: bold;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }
        .dashboard-date {
            background: #FDF6E3;
            color: #D32F2F;
            font-size: 1.1rem;
            font-weight: bold;
            text-align: right;
            padding: 4px 18px 4px 0;
            margin-bottom: 18px;
        }
        .cards-container {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            justify-content: center;
            align-items: flex-start;
            gap: 24px;
            margin-top: 18px;
            width: 100%;
            max-width: 100vw;
        }
        .card-indicador {
            background: #2874A6;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            border: 1px solid #1B4F72;
            padding: 22px 10px 18px 10px;
            min-width: 180px;
            width: auto;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 0;
        }
        .card-indicador {
            background: #2874A6;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            border: 1px solid #1B4F72;
            padding: 22px 10px 18px 10px;
            min-width: 220px;
            max-width: 270px;
            width: 100%;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .card-indicador .icon-indicador {
            font-size: 2.2rem;
            margin-bottom: 6px;
        }
        .card-indicador .titulo-indicador {
            font-size: 1.1rem;
            color: #fff;
            margin-bottom: 6px;
            font-weight: 500;
        }
        .card-indicador .valor-indicador {
            font-size: 2rem;
            font-weight: bold;
            color: #fff;
            line-height: 1.2;
            margin-bottom: 0;
        }
        @media (max-width: 900px) {
            .cards-container {
                flex-direction: column;
                align-items: center;
            }
            .card-indicador {
                max-width: 95vw;
                margin: 12px auto;
            }
        }
        @media (max-width: 600px) {
            .card-indicador {
                font-size: 1rem;
                padding: 12px 4px;
                max-width: 98vw;
            }
            .card-indicador .valor-indicador {
                font-size: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    card_style = """
        <div style="
            background:rgba(40,116,166,0.0);  /* transparente */
            border-radius:16px;
            box-shadow:0 2px 8px rgba(0,0,0,0.15);
            border:1px solid #1B4F72;
            padding:20px;
            text-align:center;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            width:100%;
            height:200px; /* Altura fixa para uniformidade */
            color:#fff;
        ">
            <span style='font-size:2.2rem;margin-bottom:6px;'>{icon}</span>
            <span style='font-size:1.1rem;font-weight:500;margin-bottom:6px;'>{title}</span>
            <span style='font-size:2rem;font-weight:bold;line-height:1.2;margin-bottom:0;'>{value}</span>
        </div>
    """

    col1.markdown(
        card_style.format(icon="🧑‍⚕️", title="Total de casos", value=format_value(total_casos)),
        unsafe_allow_html=True,
    )
    col2.markdown(
        card_style.format(icon="📅", title="Média anual de casos", value=format_value(media_anual)),
        unsafe_allow_html=True,
    )
    col3.markdown(
        card_style.format(
            icon="🏆", title="Ano com mais casos", value=f"{pico_ano} ({format_value(pico_valor)})"
        ),
        unsafe_allow_html=True,
    )
    col4.markdown(
        card_style.format(
            icon="🏙️",
            title="Município com mais casos (total acumulado)",
            value=f"{muni_pico} ({format_value(muni_pico_valor)})",
        ),
        unsafe_allow_html=True,
    )
