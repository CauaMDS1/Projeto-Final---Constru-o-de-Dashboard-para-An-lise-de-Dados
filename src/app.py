from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

from src.config import PCA_COMPONENTS_CSV, PIPELINE_REPORT_JSON, REVIEW_TRENDS_CSV, STEAM_ANALYTICS_CSV


COLORWAY = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
PRICE_BAND_ORDER = ["Free", "Low", "Medium", "High", "Premium"]
GRAPH_CONFIG = {"displayModeBar": True, "responsive": True}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not STEAM_ANALYTICS_CSV.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    frame = pd.read_csv(STEAM_ANALYTICS_CSV, low_memory=False)
    trends = (
        pd.read_csv(REVIEW_TRENDS_CSV)
        if REVIEW_TRENDS_CSV.exists()
        else pd.DataFrame(columns=["year", "review_count", "recommendation_rate"])
    )
    pca = pd.read_csv(PCA_COMPONENTS_CSV) if PCA_COMPONENTS_CSV.exists() else pd.DataFrame()
    return frame, trends, pca


df, trends_df, pca_df = load_data()

app = Dash(__name__, title="Steam Insights")
server = app.server


def empty_state() -> html.Div:
    return html.Div(
        className="empty-state",
        children=[
            html.H2("Dados processados nao encontrados"),
            html.P("Execute primeiro: python -m src.prepare_data"),
        ],
    )


def kpi_card(label: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(detail, className="kpi-detail"),
        ],
    )


def chart_panel(graph: dcc.Graph, detail: str = "") -> html.Div:
    return html.Div(
        className="chart-panel",
        children=[
            graph,
            html.Div(detail, className="chart-detail") if detail else None,
        ],
    )


def format_int(value: float | int) -> str:
    if pd.isna(value):
        return "0"
    return f"{int(value):,}".replace(",", ".")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "0,0%"
    return f"{value * 100:.1f}%".replace(".", ",")


def format_money(value: float) -> str:
    if pd.isna(value):
        return "US$ 0.00"
    return f"US$ {float(value):,.2f}"


def apply_chart_style(
    fig,
    *,
    height: int = 430,
    percent_y: bool = False,
    percent_x: bool = False,
    money_x: bool = False,
    count_x: bool = False,
    count_y: bool = False,
):
    fig.update_layout(
        template="plotly_white",
        height=height,
        colorway=COLORWAY,
        hovermode="closest",
        margin=dict(l=24, r=24, t=64, b=46),
        title=dict(font=dict(size=17), x=0.01, xanchor="left"),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color="#0f172a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, title_standoff=12)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, title_standoff=12)
    if percent_y:
        fig.update_yaxes(tickformat=".0%")
    if percent_x:
        fig.update_xaxes(tickformat=".0%")
    if money_x:
        fig.update_xaxes(tickprefix="US$ ", tickformat=",.0f")
    if count_x:
        fig.update_xaxes(tickformat=",d")
    if count_y:
        fig.update_yaxes(tickformat=",d")
    return fig


def genre_options() -> list[dict]:
    if df.empty:
        return []
    counts = df["primary_genre"].fillna("Unknown").value_counts().head(30)
    return [{"label": genre, "value": genre} for genre in counts.index]


def platform_options() -> list[dict]:
    return [
        {"label": "Windows", "value": "windows"},
        {"label": "Mac", "value": "mac"},
        {"label": "Linux", "value": "linux"},
    ]


def overview_layout() -> html.Div:
    if df.empty:
        return empty_state()

    total_games = len(df)
    total_reviews = df["recommendation_count"].sum()
    global_rate = df["recommended_count"].sum() / total_reviews if total_reviews else 0
    covered = (df["recommendation_count"] > 0).sum()
    avg_price = df["price_imputed"].mean()

    release_by_year = (
        df.dropna(subset=["release_year"])
        .groupby("release_year", as_index=False)
        .size()
        .rename(columns={"size": "games"})
    )
    release_by_year = release_by_year[release_by_year["release_year"] >= 2000]

    top_genres = (
        df.groupby("primary_genre", as_index=False)
        .agg(games=("app_id", "count"), recommendation_rate=("external_recommendation_rate", "mean"))
        .sort_values("games", ascending=False)
        .head(12)
    )

    price_summary = (
        df.groupby("price_band", as_index=False)
        .agg(
            games=("app_id", "count"),
            recommendation_rate=("external_recommendation_rate", "mean"),
        )
        .sort_values("games", ascending=False)
    )

    platform_summary = pd.DataFrame(
        {
            "platform": ["Windows", "Mac", "Linux"],
            "games": [df["windows"].sum(), df["mac"].sum(), df["linux"].sum()],
        }
    )

    fig_release = px.line(
        release_by_year,
        x="release_year",
        y="games",
        markers=True,
        title="Lancamentos por ano",
        labels={"release_year": "Ano de lancamento", "games": "Quantidade de jogos"},
        color_discrete_sequence=[COLORWAY[0]],
    )
    fig_release.update_traces(
        hovertemplate="Ano: %{x}<br>Jogos lancados: %{y:,}<extra></extra>"
    )
    fig_genres = px.bar(
        top_genres.sort_values("games"),
        x="games",
        y="primary_genre",
        orientation="h",
        title="Generos com mais jogos",
        labels={"games": "Quantidade de jogos", "primary_genre": "Genero"},
        color_discrete_sequence=[COLORWAY[1]],
    )
    fig_genres.update_traces(
        hovertemplate="Genero: %{y}<br>Jogos: %{x:,}<extra></extra>"
    )
    price_summary["price_band"] = pd.Categorical(
        price_summary["price_band"], categories=PRICE_BAND_ORDER, ordered=True
    )
    price_summary = price_summary.sort_values("price_band")
    fig_price = px.bar(
        price_summary,
        x="price_band",
        y="recommendation_rate",
        title="Taxa media de recomendacao por faixa de preco",
        color="games",
        color_continuous_scale="Blues",
        labels={
            "price_band": "Faixa de preco",
            "recommendation_rate": "Taxa media de recomendacao",
            "games": "Jogos",
        },
    )
    fig_price.update_traces(
        hovertemplate="Faixa: %{x}<br>Taxa media: %{y:.1%}<br>Jogos: %{marker.color:,}<extra></extra>"
    )
    fig_platform = px.bar(
        platform_summary,
        x="platform",
        y="games",
        title="Disponibilidade por plataforma",
        labels={"platform": "Plataforma", "games": "Quantidade de jogos"},
        color_discrete_sequence=[COLORWAY[2]],
    )
    fig_platform.update_traces(
        hovertemplate="Plataforma: %{x}<br>Jogos: %{y:,}<extra></extra>"
    )

    apply_chart_style(fig_release, count_y=True)
    apply_chart_style(fig_genres, count_x=True)
    apply_chart_style(fig_price, percent_y=True)
    apply_chart_style(fig_platform, count_y=True)

    return html.Div(
        children=[
            html.Div(
                className="kpi-grid",
                children=[
                    kpi_card("Jogos analisados", format_int(total_games), "catalogo Steam"),
                    kpi_card("Reviews integradas", format_int(total_reviews), "arquivo recommendations.csv"),
                    kpi_card("Taxa positiva", format_pct(global_rate), "recomendacoes dos usuarios"),
                    kpi_card("Cobertura", format_pct(covered / total_games), "jogos com reviews integradas"),
                    kpi_card("Preco medio", f"US$ {avg_price:.2f}", "apos imputacao"),
                ],
            ),
            html.Div(
                className="chart-grid",
                children=[
                    chart_panel(
                        dcc.Graph(figure=fig_release, config=GRAPH_CONFIG),
                        "Tendencia temporal do volume de jogos publicados na base.",
                    ),
                    chart_panel(
                        dcc.Graph(figure=fig_genres, config=GRAPH_CONFIG),
                        "Ranking por genero principal extraido do catalogo de jogos.",
                    ),
                    chart_panel(
                        dcc.Graph(figure=fig_price, config=GRAPH_CONFIG),
                        "Compara satisfacao media entre jogos gratuitos e faixas pagas.",
                    ),
                    chart_panel(
                        dcc.Graph(figure=fig_platform, config=GRAPH_CONFIG),
                        "Mostra a cobertura de Windows, Mac e Linux no catalogo.",
                    ),
                ],
            ),
        ]
    )


def filter_frame(
    selected_genres: list[str] | None,
    selected_platforms: list[str] | None,
    price_range: list[float] | None,
    year_range: list[int] | None,
    min_reviews: int,
) -> pd.DataFrame:
    filtered = df.copy()
    if selected_genres:
        filtered = filtered[filtered["primary_genre"].isin(selected_genres)]
    if selected_platforms:
        mask = np.zeros(len(filtered), dtype=bool)
        for platform in selected_platforms:
            mask = mask | (filtered[platform] == 1).to_numpy()
        filtered = filtered[mask]
    if price_range:
        filtered = filtered[
            (filtered["price_imputed"] >= price_range[0])
            & (filtered["price_imputed"] <= price_range[1])
        ]
    if year_range:
        filtered = filtered[
            (filtered["release_year"] >= year_range[0])
            & (filtered["release_year"] <= year_range[1])
        ]
    filtered = filtered[filtered["recommendation_count"] >= min_reviews]
    return filtered


def exploration_layout() -> html.Div:
    if df.empty:
        return empty_state()

    min_year = int(df["release_year"].dropna().min())
    max_year = int(df["release_year"].dropna().max())
    max_price = float(min(80, df["price_imputed"].quantile(0.99)))

    return html.Div(
        children=[
            html.Div(
                className="filters",
                children=[
                    html.Div(
                        children=[
                            html.Label("Genero"),
                            dcc.Dropdown(
                                id="genre-filter",
                                options=genre_options(),
                                multi=True,
                                placeholder="Todos",
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Plataforma"),
                            dcc.Checklist(
                                id="platform-filter",
                                options=platform_options(),
                                value=["windows", "mac", "linux"],
                                inline=True,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Preco"),
                            dcc.RangeSlider(
                                id="price-filter",
                                min=0,
                                max=max_price,
                                step=1,
                                value=[0, max_price],
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Ano de lancamento"),
                            dcc.RangeSlider(
                                id="year-filter",
                                min=min_year,
                                max=max_year,
                                step=1,
                                value=[max(2000, min_year), max_year],
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Reviews minimas"),
                            dcc.Slider(
                                id="min-reviews-filter",
                                min=0,
                                max=1000,
                                step=50,
                                value=50,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(id="filtered-kpis", className="kpi-grid compact"),
            html.Div(
                className="chart-grid",
                children=[
                    chart_panel(
                        dcc.Graph(id="scatter-price-rate", config=GRAPH_CONFIG),
                        "Cada ponto representa um jogo; o tamanho indica volume de reviews.",
                    ),
                    chart_panel(
                        dcc.Graph(id="top-games-bar", config=GRAPH_CONFIG),
                        "Ranking dos jogos com maior quantidade de reviews integradas.",
                    ),
                    chart_panel(
                        dcc.Graph(id="price-band-box", config=GRAPH_CONFIG),
                        "Compara mediana e dispersao da recomendacao entre faixas de preco.",
                    ),
                    chart_panel(
                        dcc.Graph(id="release-year-trend", config=GRAPH_CONFIG),
                        "Mostra como a recomendacao media varia pelo ano de lancamento.",
                    ),
                    chart_panel(
                        dcc.Graph(id="hours-histogram", config=GRAPH_CONFIG),
                        "Distribuicao limitada ao percentil 99 para reduzir efeito de outliers.",
                    ),
                    chart_panel(
                        dcc.Graph(id="genre-rate-bar", config=GRAPH_CONFIG),
                        "Generos com maior volume no filtro atual, ordenados pela taxa media.",
                    ),
                    chart_panel(
                        dcc.Graph(id="pca-scatter", config=GRAPH_CONFIG),
                        "PCA usando features normalizadas, padronizadas e codificadas.",
                    ),
                ],
            ),
            html.Div(
                className="table-wrap",
                children=[
                    html.H3("Jogos em destaque"),
                    dash_table.DataTable(
                        id="games-table",
                        page_size=10,
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "Arial",
                            "fontSize": 13,
                            "padding": "8px",
                            "textAlign": "left",
                        },
                        style_header={"fontWeight": "700", "backgroundColor": "#f1f5f9"},
                    ),
                ],
            ),
        ]
    )


@app.callback(
    Output("filtered-kpis", "children"),
    Output("scatter-price-rate", "figure"),
    Output("top-games-bar", "figure"),
    Output("price-band-box", "figure"),
    Output("release-year-trend", "figure"),
    Output("hours-histogram", "figure"),
    Output("genre-rate-bar", "figure"),
    Output("pca-scatter", "figure"),
    Output("games-table", "data"),
    Output("games-table", "columns"),
    Input("genre-filter", "value"),
    Input("platform-filter", "value"),
    Input("price-filter", "value"),
    Input("year-filter", "value"),
    Input("min-reviews-filter", "value"),
)
def update_exploration(selected_genres, selected_platforms, price_range, year_range, min_reviews):
    filtered = filter_frame(
        selected_genres,
        selected_platforms,
        price_range,
        year_range,
        min_reviews or 0,
    )

    if filtered.empty:
        blank = px.scatter(title="Sem dados para os filtros selecionados")
        blank.update_layout(template="plotly_white")
        return [kpi_card("Jogos filtrados", "0", "")], blank, blank, blank, blank, blank, blank, blank, [], []

    total_reviews = filtered["recommendation_count"].sum()
    global_rate = (
        filtered["recommended_count"].sum() / total_reviews if total_reviews else 0
    )
    kpis = [
        kpi_card("Jogos filtrados", format_int(len(filtered)), "apos filtros"),
        kpi_card("Reviews", format_int(total_reviews), "linhas agregadas"),
        kpi_card("Taxa positiva", format_pct(global_rate), "media ponderada"),
        kpi_card("Horas medias", f"{filtered['avg_review_hours'].mean():.1f}", "por review"),
    ]

    scatter_data = filtered.nlargest(5000, "recommendation_count")
    fig_scatter = px.scatter(
        scatter_data,
        x="price_imputed",
        y="external_recommendation_rate",
        size="recommendation_count",
        color="primary_genre",
        hover_name="name",
        title="Preco vs taxa de recomendacao",
        labels={
            "price_imputed": "Preco US$",
            "external_recommendation_rate": "Taxa de recomendacao",
            "recommendation_count": "Reviews",
            "primary_genre": "Genero",
        },
        hover_data={
            "primary_genre": True,
            "price_imputed": ":$.2f",
            "external_recommendation_rate": ":.1%",
            "recommendation_count": ":,",
            "avg_review_hours": ":.1f",
        },
    )
    fig_scatter.update_traces(
        marker=dict(opacity=0.72, line=dict(width=0.4, color="white")),
    )

    top_games = filtered.nlargest(15, "recommendation_count").sort_values(
        "recommendation_count"
    )
    fig_top = px.bar(
        top_games,
        x="recommendation_count",
        y="name",
        color="external_recommendation_rate",
        orientation="h",
        title="Jogos com mais reviews integradas",
        color_continuous_scale="Greens",
        labels={
            "recommendation_count": "Reviews integradas",
            "name": "Jogo",
            "external_recommendation_rate": "Taxa de recomendacao",
        },
        hover_data={
            "primary_genre": True,
            "price_imputed": ":$.2f",
            "external_recommendation_rate": ":.1%",
            "avg_review_hours": ":.1f",
        },
    )

    box_data = filtered[filtered["recommendation_count"] > 0]
    box_data = box_data.copy()
    box_data["price_band"] = pd.Categorical(
        box_data["price_band"], categories=PRICE_BAND_ORDER, ordered=True
    )
    fig_box = px.box(
        box_data,
        x="price_band",
        y="external_recommendation_rate",
        points=False,
        title="Distribuicao da recomendacao por faixa de preco",
        category_orders={"price_band": PRICE_BAND_ORDER},
        labels={
            "price_band": "Faixa de preco",
            "external_recommendation_rate": "Taxa de recomendacao",
        },
    )

    by_year = (
        filtered.dropna(subset=["release_year"])
        .groupby("release_year", as_index=False)
        .agg(
            games=("app_id", "count"),
            recommendation_rate=("external_recommendation_rate", "mean"),
        )
    )
    fig_year = px.line(
        by_year,
        x="release_year",
        y="recommendation_rate",
        markers=True,
        title="Recomendacao media por ano de lancamento",
        labels={
            "release_year": "Ano de lancamento",
            "recommendation_rate": "Taxa media de recomendacao",
        },
        hover_data={"games": ":,"},
    )

    hours_data = filtered[
        filtered["avg_review_hours"] <= filtered["avg_review_hours"].quantile(0.99)
    ]
    fig_hours = px.histogram(
        hours_data,
        x="avg_review_hours",
        nbins=40,
        title="Distribuicao de horas jogadas nas reviews",
        labels={"avg_review_hours": "Horas medias por review"},
        color_discrete_sequence=[COLORWAY[4]],
    )

    genre_rate = (
        filtered.groupby("primary_genre", as_index=False)
        .agg(
            games=("app_id", "count"),
            recommendation_rate=("external_recommendation_rate", "mean"),
        )
        .sort_values("games", ascending=False)
        .head(12)
        .sort_values("recommendation_rate")
    )
    fig_genre_rate = px.bar(
        genre_rate,
        x="recommendation_rate",
        y="primary_genre",
        orientation="h",
        color="games",
        color_continuous_scale="Blues",
        title="Recomendacao media por genero",
        labels={
            "recommendation_rate": "Taxa media de recomendacao",
            "primary_genre": "Genero",
            "games": "Jogos",
        },
    )

    if pca_df.empty:
        fig_pca = px.scatter(title="PCA indisponivel: execute python -m src.prepare_data")
    else:
        pca_plot = pca_df[pca_df["app_id"].isin(filtered["app_id"])]
        pca_plot = pca_plot.nlargest(min(6000, len(pca_plot)), "popularity_score")
        fig_pca = px.scatter(
            pca_plot,
            x="PC1",
            y="PC2",
            color="price_band",
            symbol="target_high_recommendation",
            hover_name="name",
            hover_data={
                "primary_genre": True,
                "recommendation_count": True,
                "external_recommendation_rate": ":.1%",
                "PC1": ":.2f",
                "PC2": ":.2f",
            },
            title="PCA: agrupamento dos jogos por perfil",
            labels={
                "PC1": "Componente principal 1",
                "PC2": "Componente principal 2",
                "price_band": "Faixa de preco",
                "target_high_recommendation": "Alta recomendacao",
            },
        )
        fig_pca.update_traces(marker=dict(opacity=0.72, line=dict(width=0.3, color="white")))

    apply_chart_style(fig_scatter, money_x=True, percent_y=True, height=470)
    apply_chart_style(fig_top, count_x=True, height=470)
    apply_chart_style(fig_box, percent_y=True, height=430)
    apply_chart_style(fig_year, percent_y=True, height=430)
    apply_chart_style(fig_hours, count_y=True, height=430)
    apply_chart_style(fig_genre_rate, percent_x=True, height=430)
    apply_chart_style(fig_pca, height=470)

    table = filtered.nlargest(50, "popularity_score")[
        [
            "name",
            "primary_genre",
            "release_year",
            "price_imputed",
            "recommendation_count",
            "external_recommendation_rate",
            "avg_review_hours",
        ]
    ].copy()
    table["price_display"] = table["price_imputed"].apply(format_money)
    table["reviews_display"] = table["recommendation_count"].apply(format_int)
    table["recommendation_pct"] = table["external_recommendation_rate"].apply(format_pct)
    table["hours_display"] = table["avg_review_hours"].map(lambda value: f"{value:.1f}")
    table["external_recommendation_rate"] = (
        table["external_recommendation_rate"] * 100
    ).round(1)
    table["avg_review_hours"] = table["avg_review_hours"].round(1)
    columns = [
        {"name": "Jogo", "id": "name"},
        {"name": "Genero", "id": "primary_genre"},
        {"name": "Ano", "id": "release_year"},
        {"name": "Preco", "id": "price_display"},
        {"name": "Reviews", "id": "reviews_display"},
        {"name": "Recomendacao", "id": "recommendation_pct"},
        {"name": "Horas medias", "id": "hours_display"},
    ]
    table_display = table[
        [
            "name",
            "primary_genre",
            "release_year",
            "price_display",
            "reviews_display",
            "recommendation_pct",
            "hours_display",
        ]
    ]
    return kpis, fig_scatter, fig_top, fig_box, fig_year, fig_hours, fig_genre_rate, fig_pca, table_display.to_dict("records"), columns


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    children=[
                        html.H1("Steam Insights"),
                        html.P("Analise de jogos, recomendacoes e comportamento de usuarios"),
                    ]
                ),
                html.Div(
                    className="source-note",
                    children="Dados: games.csv + recommendations.csv + coleta API opcional",
                ),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Dashboard 1 - Visao Geral", value="overview"),
                dcc.Tab(label="Dashboard 2 - Exploracao Interativa", value="exploration"),
            ],
        ),
        html.Div(id="tab-content"),
    ],
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab_value):
    if tab_value == "exploration":
        return exploration_layout()
    return overview_layout()


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: #f8fafc;
                color: #0f172a;
            }
            .app-shell {
                max-width: 1440px;
                margin: 0 auto;
                padding: 24px;
            }
            .topbar {
                display: flex;
                justify-content: space-between;
                gap: 24px;
                align-items: flex-end;
                margin-bottom: 18px;
            }
            h1 {
                margin: 0;
                font-size: 32px;
                letter-spacing: 0;
            }
            p {
                margin: 6px 0 0;
                color: #475569;
            }
            .source-note {
                color: #475569;
                font-size: 13px;
                text-align: right;
            }
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(5, minmax(150px, 1fr));
                gap: 12px;
                margin: 18px 0;
            }
            .kpi-grid.compact {
                grid-template-columns: repeat(4, minmax(150px, 1fr));
            }
            .kpi-card {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 14px;
                min-height: 76px;
            }
            .kpi-label {
                font-size: 12px;
                text-transform: uppercase;
                color: #64748b;
                font-weight: 700;
            }
            .kpi-value {
                font-size: 24px;
                font-weight: 800;
                margin-top: 6px;
            }
            .kpi-detail {
                font-size: 12px;
                color: #64748b;
                margin-top: 4px;
            }
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 22px;
                align-items: stretch;
                margin-top: 18px;
            }
            .chart-panel {
                background: white;
                border: 1px solid #dbe4ef;
                border-radius: 8px;
                padding: 12px 12px 14px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
                min-height: 500px;
            }
            .chart-panel .dash-graph {
                width: 100%;
            }
            .chart-detail {
                color: #475569;
                border-top: 1px solid #e2e8f0;
                font-size: 12px;
                line-height: 1.4;
                margin: 4px 6px 0;
                padding-top: 10px;
            }
            .filters {
                display: grid;
                grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr;
                gap: 14px;
                align-items: end;
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 14px;
                margin: 18px 0 22px;
            }
            .filters label {
                display: block;
                font-size: 12px;
                font-weight: 700;
                color: #334155;
                margin-bottom: 8px;
            }
            .table-wrap {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 18px;
                margin-top: 24px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            .table-wrap h3 {
                margin: 0 0 12px;
                font-size: 18px;
            }
            .empty-state {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 32px;
                margin-top: 18px;
            }
            @media (max-width: 1000px) {
                .topbar {
                    display: block;
                }
                .source-note {
                    text-align: left;
                    margin-top: 10px;
                }
                .kpi-grid,
                .kpi-grid.compact,
                .chart-grid,
                .filters {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050, use_reloader=False)
