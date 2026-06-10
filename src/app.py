from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

from src.config import PCA_COMPONENTS_CSV, PIPELINE_REPORT_JSON, REVIEW_TRENDS_CSV, STEAM_ANALYTICS_CSV


APP_BACKGROUND = "#080d19"
PANEL_BACKGROUND = "#111827"
PANEL_BACKGROUND_ALT = "#172033"
PLOT_BACKGROUND = "#0d1424"
BORDER_COLOR = "#243147"
TEXT_COLOR = "#e5eefb"
MUTED_TEXT_COLOR = "#a8b3c7"
GRID_COLOR = "#25334a"
COLORWAY = ["#22d3ee", "#a3e635", "#f97316", "#c084fc", "#60a5fa", "#fb7185", "#34d399"]
TEAL_SCALE = ["#0f172a", "#0e7490", "#22d3ee", "#a3e635"]
BLUE_SCALE = ["#111827", "#1d4ed8", "#22d3ee", "#a3e635"]
GREEN_SCALE = ["#111827", "#047857", "#22c55e", "#a3e635"]
HEATMAP_SCALE = ["#7f1d1d", "#f97316", "#22d3ee", "#a3e635"]
PRICE_BAND_ORDER = ["Free", "Low", "Medium", "High", "Premium"]
RECOMMENDATION_BAND_ORDER = ["Low", "Medium", "High", "Excellent"]
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


def weighted_recommendation_rate(frame: pd.DataFrame) -> float:
    total_reviews = frame["recommendation_count"].sum()
    if not total_reviews:
        return 0
    return frame["recommended_count"].sum() / total_reviews


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
        template="plotly_dark",
        height=height,
        colorway=COLORWAY,
        hovermode="closest",
        margin=dict(l=28, r=26, t=66, b=50),
        title=dict(font=dict(size=17, color=TEXT_COLOR), x=0.01, xanchor="left"),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=MUTED_TEXT_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(
            family="Segoe UI, Inter, Arial, Helvetica, sans-serif",
            size=12,
            color=TEXT_COLOR,
        ),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#22d3ee",
            font=dict(color=TEXT_COLOR, size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=BORDER_COLOR,
        zeroline=False,
        title_standoff=12,
        tickfont=dict(color=MUTED_TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=BORDER_COLOR,
        zeroline=False,
        title_standoff=12,
        tickfont=dict(color=MUTED_TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=MUTED_TEXT_COLOR),
            title=dict(font=dict(color=TEXT_COLOR)),
            outlinecolor=BORDER_COLOR,
        )
    )
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


def recommendation_band_options() -> list[dict]:
    return [{"label": value, "value": value} for value in RECOMMENDATION_BAND_ORDER]


def overview_layout() -> html.Div:
    if df.empty:
        return empty_state()

    covered_df = df[df["recommendation_count"].fillna(0) > 0].copy()
    total_games = len(df)
    total_reviews = covered_df["recommendation_count"].sum()
    global_rate = weighted_recommendation_rate(covered_df)
    covered = len(covered_df)
    median_price = df["price_imputed"].median()
    avg_hours = covered_df["avg_review_hours"].mean() if covered else 0

    release_by_year = (
        df.dropna(subset=["release_year"])
        .groupby("release_year", as_index=False)
        .size()
        .rename(columns={"size": "games"})
    )
    release_by_year = release_by_year[release_by_year["release_year"] >= 2000]

    review_trend = trends_df.copy()
    if not review_trend.empty:
        review_trend["year"] = pd.to_numeric(review_trend["year"], errors="coerce")
        review_trend = review_trend.dropna(subset=["year"]).sort_values("year")
        review_trend["year"] = review_trend["year"].astype(int)

    genre_catalog = (
        df.groupby("primary_genre", as_index=False)
        .agg(games=("app_id", "count"))
        .sort_values("games", ascending=False)
        .head(12)
    )

    price_summary = (
        covered_df.groupby("price_band", as_index=False)
        .agg(
            games=("app_id", "count"),
            reviews=("recommendation_count", "sum"),
            recommended=("recommended_count", "sum"),
            avg_hours=("avg_review_hours", "mean"),
        )
    )
    price_summary["recommendation_rate"] = (
        price_summary["recommended"] / price_summary["reviews"]
    ).fillna(0)
    price_summary["price_band"] = pd.Categorical(
        price_summary["price_band"], categories=PRICE_BAND_ORDER, ordered=True
    )
    price_summary = price_summary.sort_values("price_band")

    genre_rate = (
        covered_df.groupby("primary_genre", as_index=False)
        .agg(
            games=("app_id", "count"),
            reviews=("recommendation_count", "sum"),
            recommended=("recommended_count", "sum"),
        )
        .sort_values("games", ascending=False)
        .head(12)
    )
    genre_rate["recommendation_rate"] = (
        genre_rate["recommended"] / genre_rate["reviews"]
    ).fillna(0)
    genre_rate = genre_rate.sort_values("recommendation_rate")

    platform_rows = []
    for column, label in [("windows", "Windows"), ("mac", "Mac"), ("linux", "Linux")]:
        available = df[df[column] == 1]
        recommended = covered_df[covered_df[column] == 1]
        platform_rows.append(
            {
                "platform": label,
                "games": len(available),
                "reviews": recommended["recommendation_count"].sum(),
                "recommended": recommended["recommended_count"].sum(),
            }
        )
    platform_summary = pd.DataFrame(platform_rows)
    platform_summary["recommendation_rate"] = (
        platform_summary["recommended"] / platform_summary["reviews"]
    ).fillna(0)

    best_price = (
        price_summary.sort_values("recommendation_rate", ascending=False).iloc[0]
        if not price_summary.empty
        else None
    )
    best_platform = (
        platform_summary.sort_values("recommendation_rate", ascending=False).iloc[0]
        if not platform_summary.empty
        else None
    )
    review_span = (
        f"{int(review_trend['year'].min())}-{int(review_trend['year'].max())}"
        if not review_trend.empty
        else "sem recorte temporal"
    )

    fig_release = px.line(
        release_by_year,
        x="release_year",
        y="games",
        markers=True,
        title="Crescimento do catalogo por ano de lancamento",
        labels={"release_year": "Ano de lancamento", "games": "Quantidade de jogos"},
        color_discrete_sequence=[COLORWAY[0]],
    )
    fig_release.update_traces(
        hovertemplate="Ano: %{x}<br>Jogos lancados: %{y:,}<extra></extra>"
    )

    fig_review_trend = px.line(
        review_trend,
        x="year",
        y="recommendation_rate",
        markers=True,
        title="Taxa positiva por ano da review",
        labels={
            "year": "Ano da review",
            "recommendation_rate": "Taxa positiva",
            "review_count": "Reviews",
        },
        hover_data={"review_count": ":,"},
        color_discrete_sequence=[COLORWAY[3]],
    )

    fig_genres = px.bar(
        genre_catalog.sort_values("games"),
        x="games",
        y="primary_genre",
        orientation="h",
        title="Composicao do catalogo por genero",
        labels={"games": "Quantidade de jogos", "primary_genre": "Genero"},
        color_discrete_sequence=[COLORWAY[1]],
    )
    fig_genres.update_traces(
        hovertemplate="Genero: %{y}<br>Jogos: %{x:,}<extra></extra>"
    )

    fig_price = px.bar(
        price_summary,
        x="price_band",
        y="recommendation_rate",
        title="Valor percebido: recomendacao por faixa de preco",
        color="reviews",
        color_continuous_scale=BLUE_SCALE,
        labels={
            "price_band": "Faixa de preco",
            "recommendation_rate": "Taxa positiva",
            "reviews": "Reviews",
        },
    )
    fig_price.update_traces(
        hovertemplate="Faixa: %{x}<br>Taxa positiva: %{y:.1%}<br>Reviews: %{marker.color:,}<extra></extra>"
    )

    fig_platform = px.bar(
        platform_summary,
        x="platform",
        y="games",
        title="Plataformas: alcance e satisfacao",
        labels={
            "platform": "Plataforma",
            "games": "Quantidade de jogos",
            "recommendation_rate": "Taxa positiva",
        },
        color="recommendation_rate",
        color_continuous_scale=TEAL_SCALE,
        hover_data={"recommendation_rate": ":.1%", "reviews": ":,"},
    )
    fig_platform.update_traces(
        hovertemplate="Plataforma: %{x}<br>Jogos: %{y:,}<br>Taxa positiva: %{marker.color:.1%}<extra></extra>"
    )

    fig_genre_rate = px.bar(
        genre_rate,
        x="recommendation_rate",
        y="primary_genre",
        orientation="h",
        color="reviews",
        color_continuous_scale=GREEN_SCALE,
        title="Generos populares nem sempre sao os mais bem avaliados",
        labels={
            "recommendation_rate": "Taxa positiva",
            "primary_genre": "Genero",
            "reviews": "Reviews",
        },
    )

    apply_chart_style(fig_release, count_y=True)
    apply_chart_style(fig_review_trend, percent_y=True)
    apply_chart_style(fig_genres, count_x=True)
    apply_chart_style(fig_price, percent_y=True)
    apply_chart_style(fig_platform, count_y=True)
    apply_chart_style(fig_genre_rate, percent_x=True)

    return html.Div(
        children=[
            html.Div(
                className="kpi-grid",
                children=[
                    kpi_card("Jogos analisados", format_int(total_games), "catalogo Steam"),
                    kpi_card("Reviews integradas", format_int(total_reviews), "arquivo recommendations.csv"),
                    kpi_card("Taxa positiva", format_pct(global_rate), "recomendacoes dos usuarios"),
                    kpi_card("Cobertura", format_pct(covered / total_games), "jogos com reviews integradas"),
                    kpi_card("Preco mediano", f"US$ {median_price:.2f}", "apos imputacao"),
                    kpi_card("Horas medias", f"{avg_hours:.1f}", "por review agregada"),
                ],
            ),
            html.Div(
                className="story-grid",
                children=[
                    kpi_card(
                        "Valor percebido",
                        str(best_price["price_band"]) if best_price is not None else "-",
                        f"maior taxa positiva: {format_pct(best_price['recommendation_rate'])}"
                        if best_price is not None
                        else "",
                    ),
                    kpi_card(
                        "Plataforma destaque",
                        str(best_platform["platform"]) if best_platform is not None else "-",
                        f"taxa positiva: {format_pct(best_platform['recommendation_rate'])}"
                        if best_platform is not None
                        else "",
                    ),
                    kpi_card(
                        "Recorte temporal",
                        review_span,
                        "anos disponiveis em recommendations.csv",
                    ),
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
                        dcc.Graph(figure=fig_review_trend, config=GRAPH_CONFIG),
                        "Esta leitura usa a data da review, evitando confundir lancamento com early access ou atualizacao do catalogo.",
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
                        "Combina quantidade de jogos disponiveis com a taxa positiva das reviews integradas.",
                    ),
                    chart_panel(
                        dcc.Graph(figure=fig_genre_rate, config=GRAPH_CONFIG),
                        "Compara satisfacao dos generos mais presentes, ponderando pelo volume de reviews.",
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
    hours_range: list[float] | None,
    recommendation_bands: list[str] | None,
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
    if hours_range:
        filtered = filtered[
            (filtered["avg_review_hours"] >= hours_range[0])
            & (filtered["avg_review_hours"] <= hours_range[1])
        ]
    if recommendation_bands:
        filtered = filtered[filtered["recommendation_rate_band"].isin(recommendation_bands)]
    return filtered


def exploration_layout() -> html.Div:
    if df.empty:
        return empty_state()

    min_year = int(df["release_year"].dropna().min())
    max_year = int(df["release_year"].dropna().max())
    max_price = float(min(80, df["price_imputed"].quantile(0.99)))
    reviewed = df[df["recommendation_count"].fillna(0) > 0]
    max_hours = float(min(250, reviewed["avg_review_hours"].quantile(0.99))) if not reviewed.empty else 1
    max_hours = max(1, round(max_hours))

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
                    html.Div(
                        children=[
                            html.Label("Horas jogadas"),
                            dcc.RangeSlider(
                                id="hours-filter",
                                min=0,
                                max=max_hours,
                                step=1,
                                value=[0, max_hours],
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Classe de recomendacao"),
                            dcc.Dropdown(
                                id="recommendation-band-filter",
                                options=recommendation_band_options(),
                                multi=True,
                                placeholder="Todas",
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
                        dcc.Graph(id="scatter-hours-rate", config=GRAPH_CONFIG),
                        "Mostra se engajamento medido por horas acompanha a satisfacao declarada.",
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
                        dcc.Graph(id="genre-price-heatmap", config=GRAPH_CONFIG),
                        "Cruza genero e faixa de preco para localizar combinacoes com melhor percepcao de valor.",
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
                        style_as_list_view=True,
                        style_table={
                            "overflowX": "auto",
                            "border": f"1px solid {BORDER_COLOR}",
                            "borderRadius": "8px",
                        },
                        style_cell={
                            "fontFamily": "Segoe UI, Inter, Arial, Helvetica, sans-serif",
                            "fontSize": 13,
                            "padding": "10px 12px",
                            "textAlign": "left",
                            "backgroundColor": PANEL_BACKGROUND,
                            "color": TEXT_COLOR,
                            "border": f"1px solid {BORDER_COLOR}",
                            "minWidth": "110px",
                            "maxWidth": "320px",
                            "whiteSpace": "normal",
                        },
                        style_header={
                            "fontWeight": "700",
                            "backgroundColor": PANEL_BACKGROUND_ALT,
                            "color": "#f8fafc",
                            "border": f"1px solid {BORDER_COLOR}",
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "#0d1424"},
                            {
                                "if": {"state": "active"},
                                "backgroundColor": "#123345",
                                "border": "1px solid #22d3ee",
                            },
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "#172554",
                                "border": "1px solid #60a5fa",
                            },
                        ],
                        style_cell_conditional=[
                            {"if": {"column_id": "name"}, "minWidth": "210px", "fontWeight": "700"},
                            {
                                "if": {
                                    "column_id": [
                                        "release_year",
                                        "price_display",
                                        "reviews_display",
                                        "recommendation_pct",
                                        "hours_display",
                                    ]
                                },
                                "textAlign": "right",
                            },
                        ],
                    ),
                ],
            ),
        ]
    )


@app.callback(
    Output("filtered-kpis", "children"),
    Output("scatter-price-rate", "figure"),
    Output("scatter-hours-rate", "figure"),
    Output("top-games-bar", "figure"),
    Output("price-band-box", "figure"),
    Output("release-year-trend", "figure"),
    Output("hours-histogram", "figure"),
    Output("genre-rate-bar", "figure"),
    Output("genre-price-heatmap", "figure"),
    Output("pca-scatter", "figure"),
    Output("games-table", "data"),
    Output("games-table", "columns"),
    Input("genre-filter", "value"),
    Input("platform-filter", "value"),
    Input("price-filter", "value"),
    Input("year-filter", "value"),
    Input("min-reviews-filter", "value"),
    Input("hours-filter", "value"),
    Input("recommendation-band-filter", "value"),
)
def update_exploration(
    selected_genres,
    selected_platforms,
    price_range,
    year_range,
    min_reviews,
    hours_range,
    recommendation_bands,
):
    filtered = filter_frame(
        selected_genres,
        selected_platforms,
        price_range,
        year_range,
        min_reviews or 0,
        hours_range,
        recommendation_bands,
    )

    if filtered.empty:
        blank = px.scatter(title="Sem dados para os filtros selecionados")
        apply_chart_style(blank)
        return [kpi_card("Jogos filtrados", "0", "")], blank, blank, blank, blank, blank, blank, blank, blank, blank, [], []

    total_reviews = filtered["recommendation_count"].sum()
    global_rate = weighted_recommendation_rate(filtered)
    median_hours = filtered["avg_review_hours"].median()
    median_price = filtered["price_imputed"].median()
    free_share = filtered["is_free"].mean()
    kpis = [
        kpi_card("Jogos filtrados", format_int(len(filtered)), "apos filtros"),
        kpi_card("Reviews", format_int(total_reviews), "linhas agregadas"),
        kpi_card("Taxa positiva", format_pct(global_rate), "media ponderada"),
        kpi_card("Horas medianas", f"{median_hours:.1f}", "por review"),
        kpi_card("Preco mediano", f"US$ {median_price:.2f}", "jogos filtrados"),
        kpi_card("Jogos gratuitos", format_pct(free_share), "participacao no filtro"),
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
        marker=dict(opacity=0.78, line=dict(width=0.5, color=APP_BACKGROUND)),
    )

    hours_scatter_data = filtered[
        filtered["avg_review_hours"] <= filtered["avg_review_hours"].quantile(0.99)
    ].nlargest(5000, "recommendation_count")
    fig_hours_scatter = px.scatter(
        hours_scatter_data,
        x="avg_review_hours",
        y="external_recommendation_rate",
        size="recommendation_count",
        color="price_band",
        hover_name="name",
        title="Horas jogadas vs taxa de recomendacao",
        category_orders={"price_band": PRICE_BAND_ORDER},
        labels={
            "avg_review_hours": "Horas medias por review",
            "external_recommendation_rate": "Taxa de recomendacao",
            "recommendation_count": "Reviews",
            "price_band": "Faixa de preco",
        },
        hover_data={
            "primary_genre": True,
            "price_imputed": ":$.2f",
            "external_recommendation_rate": ":.1%",
            "recommendation_count": ":,",
        },
    )
    fig_hours_scatter.update_traces(
        marker=dict(opacity=0.78, line=dict(width=0.5, color=APP_BACKGROUND)),
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
        title="Popularidade vs satisfacao",
        color_continuous_scale=GREEN_SCALE,
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
            reviews=("recommendation_count", "sum"),
            recommended=("recommended_count", "sum"),
        )
    )
    by_year["recommendation_rate"] = (
        by_year["recommended"] / by_year["reviews"]
    ).fillna(0)
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
        hover_data={"games": ":,", "reviews": ":,"},
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
            reviews=("recommendation_count", "sum"),
            recommended=("recommended_count", "sum"),
        )
        .sort_values("games", ascending=False)
        .head(12)
    )
    genre_rate["recommendation_rate"] = (
        genre_rate["recommended"] / genre_rate["reviews"]
    ).fillna(0)
    genre_rate = genre_rate.sort_values("recommendation_rate")
    fig_genre_rate = px.bar(
        genre_rate,
        x="recommendation_rate",
        y="primary_genre",
        orientation="h",
        color="reviews",
        color_continuous_scale=BLUE_SCALE,
        title="Recomendacao media por genero",
        labels={
            "recommendation_rate": "Taxa media de recomendacao",
            "primary_genre": "Genero",
            "reviews": "Reviews",
        },
    )

    heatmap_base = filtered[filtered["recommendation_count"] > 0].copy()
    if heatmap_base.empty:
        fig_heatmap = px.imshow([[0]], title="Genero x faixa de preco indisponivel")
    else:
        top_heat_genres = (
            heatmap_base["primary_genre"].value_counts().head(10).index.tolist()
        )
        heatmap_base = heatmap_base[heatmap_base["primary_genre"].isin(top_heat_genres)]
        heatmap_base["price_band"] = pd.Categorical(
            heatmap_base["price_band"], categories=PRICE_BAND_ORDER, ordered=True
        )
        heatmap = (
            heatmap_base.groupby(["primary_genre", "price_band"], observed=False)
            .agg(
                reviews=("recommendation_count", "sum"),
                recommended=("recommended_count", "sum"),
            )
            .reset_index()
        )
        heatmap["recommendation_rate"] = (
            heatmap["recommended"] / heatmap["reviews"]
        ).fillna(np.nan)
        heatmap_matrix = (
            heatmap.pivot(
                index="primary_genre",
                columns="price_band",
                values="recommendation_rate",
            )
            .reindex(index=top_heat_genres, columns=PRICE_BAND_ORDER)
        )
        fig_heatmap = px.imshow(
            heatmap_matrix,
            aspect="auto",
            color_continuous_scale=HEATMAP_SCALE,
            zmin=0.5,
            zmax=1.0,
            text_auto=".0%",
            title="Genero x preco: onde a satisfacao aparece",
            labels={
                "x": "Faixa de preco",
                "y": "Genero",
                "color": "Taxa positiva",
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
        fig_pca.update_traces(marker=dict(opacity=0.78, line=dict(width=0.4, color=APP_BACKGROUND)))

    apply_chart_style(fig_scatter, money_x=True, percent_y=True, height=470)
    apply_chart_style(fig_hours_scatter, percent_y=True, height=470)
    apply_chart_style(fig_top, count_x=True, height=470)
    apply_chart_style(fig_box, percent_y=True, height=430)
    apply_chart_style(fig_year, percent_y=True, height=430)
    apply_chart_style(fig_hours, count_y=True, height=430)
    apply_chart_style(fig_genre_rate, percent_x=True, height=430)
    fig_heatmap.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
    apply_chart_style(fig_heatmap, height=470)
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
    return (
        kpis,
        fig_scatter,
        fig_hours_scatter,
        fig_top,
        fig_box,
        fig_year,
        fig_hours,
        fig_genre_rate,
        fig_heatmap,
        fig_pca,
        table_display.to_dict("records"),
        columns,
    )


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    className="title-block",
                    children=[
                        html.H1("Steam Insights"),
                        html.P("Analise de jogos, recomendacoes e comportamento de usuarios"),
                    ]
                ),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Visão Geral", value="overview"),
                dcc.Tab(label="Exploração Interativa", value="exploration"),
            ],
        ),
        html.Div(id="tab-content", className="tab-content"),
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
            :root {
                --bg: #080d19;
                --bg-soft: #0d1424;
                --panel: #111827;
                --panel-2: #172033;
                --panel-3: #0b1220;
                --border: #243147;
                --border-strong: #334155;
                --ink: #e5eefb;
                --ink-strong: #f8fafc;
                --muted: #a8b3c7;
                --muted-2: #7c8aa5;
                --cyan: #22d3ee;
                --lime: #a3e635;
                --violet: #c084fc;
                --orange: #f97316;
                --danger: #fb7185;
                --focus: #67e8f9;
            }
            * {
                box-sizing: border-box;
            }
            body {
                margin: 0;
                min-height: 100vh;
                font-family: "Segoe UI", Inter, Arial, Helvetica, sans-serif;
                background: linear-gradient(180deg, #10182b 0%, var(--bg) 42%, #060a13 100%);
                color: var(--ink);
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
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 18px 20px;
                margin-bottom: 16px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }
            .title-block {
                min-width: 0;
            }
            h1 {
                margin: 0;
                color: var(--ink-strong);
                font-size: 34px;
                line-height: 1.08;
                letter-spacing: 0;
            }
            p {
                margin: 6px 0 0;
                color: var(--muted);
                line-height: 1.5;
            }
            #tabs {
                margin-top: 16px;
            }
            .tab {
                background: var(--panel-3) !important;
                color: var(--muted) !important;
                border: 1px solid var(--border) !important;
                border-bottom: 1px solid var(--border-strong) !important;
                border-radius: 8px 8px 0 0 !important;
                padding: 12px 16px !important;
                font-weight: 800 !important;
                letter-spacing: 0 !important;
                transition: background-color 180ms ease-out, color 180ms ease-out, border-color 180ms ease-out;
            }
            .tab:hover {
                color: var(--ink-strong) !important;
                border-color: rgba(34, 211, 238, 0.5) !important;
            }
            .tab--selected {
                background: var(--panel-2) !important;
                color: var(--ink-strong) !important;
                border-color: var(--cyan) !important;
                box-shadow: inset 0 -2px 0 var(--cyan);
            }
            .tab:focus-visible,
            .dash-dropdown:focus-visible,
            .Select-control:focus-within,
            .rc-slider-handle:focus-visible {
                outline: 2px solid var(--focus);
                outline-offset: 2px;
            }
            .tab-content {
                padding-top: 2px;
            }
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin: 18px 0;
            }
            .kpi-grid.compact {
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            }
            .story-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin: 0 0 18px;
            }
            .story-grid .kpi-card {
                background: var(--panel-2);
                border-color: rgba(34, 211, 238, 0.34);
            }
            .kpi-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 15px;
                min-height: 92px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
                transition: border-color 180ms ease-out, background-color 180ms ease-out;
            }
            .kpi-card:hover {
                border-color: rgba(163, 230, 53, 0.46);
                background: #142035;
            }
            .kpi-label {
                font-size: 12px;
                text-transform: uppercase;
                color: #9fb1cf;
                font-weight: 800;
                letter-spacing: 0;
            }
            .kpi-value {
                color: var(--ink-strong);
                font-size: 25px;
                font-weight: 800;
                margin-top: 6px;
                line-height: 1.05;
            }
            .kpi-detail {
                font-size: 12px;
                color: var(--muted);
                line-height: 1.35;
                margin-top: 4px;
            }
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px;
                align-items: stretch;
                margin-top: 18px;
            }
            .chart-panel {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 12px 12px 14px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
                min-height: 500px;
                overflow: hidden;
            }
            .chart-panel .dash-graph {
                width: 100%;
            }
            .chart-detail {
                color: var(--muted);
                border-top: 1px solid var(--border);
                font-size: 12px;
                line-height: 1.4;
                margin: 4px 6px 0;
                padding-top: 10px;
            }
            .filters {
                display: grid;
                grid-template-columns: repeat(4, minmax(180px, 1fr));
                gap: 14px;
                align-items: start;
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 14px;
                margin: 18px 0 22px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
            }
            .filters label {
                display: block;
                font-size: 12px;
                font-weight: 800;
                color: #cbd5e1;
                letter-spacing: 0;
                margin-bottom: 8px;
            }
            .Select-control,
            .Select-menu-outer,
            .Select-placeholder,
            .Select-value,
            .Select-input,
            .Select-input > input {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
            }
            .Select-control {
                min-height: 40px;
                border: 1px solid var(--border-strong) !important;
                border-radius: 8px !important;
                box-shadow: none !important;
            }
            .Select-control:hover {
                border-color: var(--cyan) !important;
            }
            .Select-menu-outer {
                border: 1px solid var(--border-strong) !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.26) !important;
                overflow: hidden;
            }
            .Select-option {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
            }
            .Select-option.is-focused {
                background: #123345 !important;
            }
            .Select-option.is-selected {
                background: #164e63 !important;
                color: #ecfeff !important;
            }
            .Select--multi .Select-value {
                border: 1px solid rgba(34, 211, 238, 0.45) !important;
                border-radius: 999px !important;
                background: rgba(34, 211, 238, 0.13) !important;
                color: #cffafe !important;
            }
            .Select--multi .Select-value-label {
                color: #cffafe !important;
            }
            .Select-arrow {
                border-top-color: var(--muted) !important;
            }
            .filters input[type="checkbox"] {
                accent-color: var(--cyan);
                width: 16px;
                height: 16px;
            }
            .filters label input {
                margin-right: 6px;
            }
            .dash-dropdown {
                width: 100%;
                min-height: 40px;
                padding: 0 10px;
                color: var(--ink);
                text-align: left;
                background: var(--panel-3);
                border: 1px solid var(--border-strong);
                border-radius: 8px;
                transition: border-color 180ms ease-out, background-color 180ms ease-out;
            }
            .dash-dropdown:hover,
            .dash-dropdown[data-state="open"] {
                border-color: var(--cyan);
                background: #0f1b2f;
            }
            .dash-dropdown-grid-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                width: 100%;
            }
            .dash-dropdown-value {
                color: var(--ink);
                font-size: 14px;
            }
            .dash-dropdown-placeholder {
                color: #cbd5e1;
            }
            .dash-dropdown-trigger-icon {
                color: var(--muted);
                flex: 0 0 auto;
            }
            .dash-dropdown-focus-target,
            .dash-input-container {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
                border-color: var(--border-strong) !important;
            }
            .dash-input-container {
                border-radius: 6px !important;
                caret-color: var(--cyan);
            }
            .dash-dropdown-content {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
                border: 1px solid var(--border-strong) !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.26) !important;
                overflow: hidden;
            }
            .dash-dropdown-search-container {
                background: var(--panel-3) !important;
                border-bottom: 1px solid var(--border) !important;
                color: var(--ink) !important;
            }
            .dash-dropdown-search-container input,
            .dash-dropdown-content input[type="search"],
            .dash-dropdown-content input[type="text"] {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
                border: 1px solid var(--violet) !important;
                border-radius: 6px !important;
                caret-color: var(--cyan);
            }
            .dash-dropdown-search-container input::placeholder {
                color: var(--muted) !important;
            }
            .dash-dropdown-search-icon {
                color: var(--muted) !important;
            }
            .dash-dropdown-select-all,
            .dash-dropdown-deselect-all,
            .dash-dropdown-content button {
                background: var(--panel-3) !important;
                color: #cbd5e1 !important;
                border: 0 !important;
            }
            .dash-dropdown-select-all:hover,
            .dash-dropdown-deselect-all:hover,
            .dash-dropdown-content button:hover {
                color: var(--cyan) !important;
            }
            .dash-options-list-option-checkbox {
                accent-color: var(--cyan);
            }
            .dash-slider-container {
                display: grid;
                grid-template-columns: minmax(56px, max-content) minmax(0, 1fr) minmax(64px, max-content);
                align-items: center;
                gap: 10px;
            }
            .dash-slider-wrapper {
                grid-column: 2;
                min-width: 0;
            }
            .dash-range-slider-min-input {
                grid-column: 1;
            }
            .dash-range-slider-max-input {
                grid-column: 3;
            }
            .dash-slider-container:not(:has(.dash-range-slider-min-input)) {
                grid-template-columns: minmax(0, 1fr) minmax(64px, max-content);
            }
            .dash-slider-container:not(:has(.dash-range-slider-min-input)) .dash-slider-wrapper {
                grid-column: 1;
            }
            .dash-slider-container:not(:has(.dash-range-slider-min-input)) .dash-range-slider-max-input {
                grid-column: 2;
            }
            .dash-range-slider-input {
                width: 64px !important;
                min-width: 56px;
                min-height: 36px;
                padding: 6px 8px;
                text-align: center;
            }
            .dash-slider-root {
                width: 100%;
                min-height: 46px;
            }
            .dash-slider-track {
                background: #1f2937 !important;
                height: 6px !important;
            }
            .dash-slider-range {
                background: var(--violet) !important;
            }
            .dash-slider-thumb {
                width: 18px !important;
                height: 18px !important;
                background: var(--bg) !important;
                border: 2px solid var(--lime) !important;
                border-radius: 999px !important;
                box-shadow: 0 0 0 4px rgba(163, 230, 53, 0.12) !important;
            }
            .dash-slider-thumb:focus-visible,
            .dash-slider-thumb:hover {
                border-color: var(--cyan) !important;
                box-shadow: 0 0 0 5px rgba(34, 211, 238, 0.16) !important;
                outline: none;
            }
            .dash-slider-mark {
                color: var(--muted-2) !important;
                font-size: 11px;
            }
            .dash-slider-tooltip {
                color: var(--ink) !important;
                background: var(--panel-2) !important;
                border: 1px solid var(--border) !important;
                border-radius: 6px;
            }
            [role="listbox"] {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
                border: 1px solid var(--border-strong) !important;
                border-radius: 8px !important;
                overflow: hidden;
            }
            [role="option"] {
                background: var(--panel-3) !important;
                color: var(--ink) !important;
            }
            [role="option"]:hover,
            [role="option"][data-highlighted] {
                background: #123345 !important;
            }
            [role="option"][aria-selected="true"] {
                background: #164e63 !important;
                color: #ecfeff !important;
            }
            .rc-slider {
                padding: 8px 0 20px;
            }
            .rc-slider-rail {
                background-color: #1f2937;
                height: 6px;
            }
            .rc-slider-track {
                background-color: var(--cyan);
                height: 6px;
            }
            .rc-slider-handle {
                width: 18px;
                height: 18px;
                margin-top: -6px;
                background-color: var(--bg);
                border: 2px solid var(--lime);
                opacity: 1;
                box-shadow: 0 0 0 4px rgba(163, 230, 53, 0.12);
            }
            .rc-slider-handle:hover,
            .rc-slider-handle-dragging {
                border-color: var(--cyan) !important;
                box-shadow: 0 0 0 5px rgba(34, 211, 238, 0.16) !important;
            }
            .rc-slider-tooltip-inner {
                background: var(--panel-2);
                color: var(--ink);
                border: 1px solid var(--border);
                box-shadow: none;
            }
            .table-wrap {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 18px;
                margin-top: 24px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
            }
            .table-wrap h3 {
                margin: 0 0 12px;
                color: var(--ink-strong);
                font-size: 18px;
                letter-spacing: 0;
            }
            .dash-spreadsheet-container {
                color: var(--ink);
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                border-color: var(--border) !important;
            }
            .js-plotly-plot .modebar {
                background: rgba(13, 20, 36, 0.72) !important;
                border: 1px solid rgba(36, 49, 71, 0.72);
                border-radius: 8px;
            }
            .js-plotly-plot .modebar-btn path {
                fill: var(--muted) !important;
            }
            .js-plotly-plot .modebar-btn:hover path {
                fill: var(--cyan) !important;
            }
            .empty-state {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 32px;
                margin-top: 18px;
                color: var(--ink);
            }
            .empty-state h2 {
                margin: 0;
                color: var(--ink-strong);
                letter-spacing: 0;
            }
            @media (max-width: 1000px) {
                .topbar {
                    display: block;
                    padding: 16px;
                }
                .kpi-grid,
                .kpi-grid.compact,
                .story-grid,
                .chart-grid,
                .filters {
                    grid-template-columns: 1fr;
                }
            }
            @media (max-width: 640px) {
                .app-shell {
                    padding: 14px;
                }
                h1 {
                    font-size: 28px;
                }
                .tab {
                    padding: 10px 12px !important;
                    font-size: 13px !important;
                }
                .chart-panel {
                    padding: 8px;
                    min-height: 440px;
                }
            }
            @media (prefers-reduced-motion: reduce) {
                *,
                *::before,
                *::after {
                    transition-duration: 0.01ms !important;
                    scroll-behavior: auto !important;
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
