from pandas import read_csv, DataFrame, Series
from numpy import array
from matplotlib.pyplot import subplots, savefig, figure, suptitle
from matplotlib.figure import Figure
from dslabs_functions import ts_aggregation_by, set_chart_labels, HEIGHT, plot_multiline_chart

# Leitura dos dados
filename = "datasets/0_1_forecast_ny_arrests.csv"
file_tag = "1_Distribution_1_forecasting_ny_arrests"
subject = " [ny arrests]"
data: DataFrame = read_csv(
    filename,
    index_col="Date",
    sep=";",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

target = 'Manhattan'
series: Series = data[target]

# Agregações
ss_week: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)
ss_month: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
ss_quarter: Series = ts_aggregation_by(series, gran_level="Q", agg_func=sum)
ss_year: Series = ts_aggregation_by(series, gran_level="Y", agg_func=sum)

# Gráficos de Boxplots
fig: Figure
axs: array
fig, axs = subplots(2, 4, figsize=(4 * 5, 2 * 5))  # 2 linhas, 4 colunas

granularities_boxplot = [ss_week, ss_month, ss_quarter, ss_year]
gran_names_boxplot = ["WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY"]

# Plotar os boxplots
for i, (gran, name) in enumerate(zip(granularities_boxplot, gran_names_boxplot)):
    set_chart_labels(axs[0, i], title=name)
    axs[0, i].boxplot(gran.dropna().values)

# Estatísticas descritivas
for i, gran in enumerate(granularities_boxplot):
    axs[1, i].grid(False)
    axs[1, i].set_axis_off()
    axs[1, i].text(0, 0.5, str(gran.describe()), fontsize="small")

savefig(f"images/B/DP_DataDistribution/1_Distribution_boxplots_{target}.png", bbox_inches="tight")

# Gráficos de Histogramas
granularities_histogram = [ss_week, ss_month, ss_quarter, ss_year]
gran_names_histogram = ["Weekly", "Monthly", "Quarterly", "Yearly"]

fig, axs = subplots(1, len(granularities_histogram), figsize=(len(granularities_histogram) * 5, 5))

# Título principal do gráfico de histogramas
fig.suptitle(f"Histogram Analysis of NY Arrests: {target}", fontsize=12)

# Loop para plotar os histogramas
for gran, name, ax in zip(granularities_histogram, gran_names_histogram, axs):
    set_chart_labels(ax, title=name, xlabel=target, ylabel="Nr records")
    ax.hist(gran.dropna().values)

savefig(f"images/B/DP_DataDistribution/1_Distribution_histograms_{target}.png", bbox_inches="tight")



def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series


series: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
figure(figsize=(3 * HEIGHT, HEIGHT))
lags = get_lagged_series(series, 20, 10)

plot_multiline_chart(series.index.to_list(), lags, xlabel="Date", ylabel=target)
suptitle(f"Distribution of Lags for {target} in Months" + subject, fontsize=12)
savefig(f"images/B/DP_DataDistribution/1_Distribution_lags_{target}.png", bbox_inches="tight")



from matplotlib.pyplot import setp
from matplotlib.gridspec import GridSpec

def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    # ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return

series: Series = data[target]
autocorrelation_study(series, 10, 1)
suptitle(f"Autocorrelation for {target}" + subject, fontsize=12)
savefig(f"images/B/DP_DataDistribution/1_Distribution_autocorrelation_{target}.png", bbox_inches="tight")

