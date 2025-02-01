import pandas as pd
from matplotlib import pyplot as plt
from numpy import sum
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by

file_tag = "2_DataProfiling_2_GDP_Europe_Distribution"
target = "GDP"
index = "Year"
filename = "datasets/0_2_forecast_gdp_europe.csv"

def ts_aggregation_by_years(
    data: Series | DataFrame,
    num_years: int,
    agg_func: str = "mean",
) -> Series | DataFrame:
    """
    Aggregates time series data by a specified number of years.

    Parameters:
        data (Series | DataFrame): Input time series data with a DatetimeIndex.
        num_years (int): Number of years to group by.
        agg_func (str): Aggregation function to apply (e.g., 'mean', 'sum').

    Returns:
        Series | DataFrame: Aggregated data grouped by the specified number of years.
    """
    # Ensure the index is datetime
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("The index must be a DatetimeIndex or PeriodIndex.")

    # Convert the index to groups based on the number of years
    group_periods = (data.index.year // num_years) * num_years

    if isinstance(data, pd.Series):
        # Convert Series to DataFrame temporarily for grouping
        df = data.to_frame(name="value")
        df["group"] = group_periods
        grouped = df.groupby("group")["value"].agg(agg_func)
    else:
        # If it's already a DataFrame
        data["group"] = group_periods
        grouped = data.groupby("group").agg(agg_func)

    # Reset index to datetime for group start
    grouped.index = pd.to_datetime(grouped.index, format="%Y")

    return grouped

data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]
ss_5years: Series = ts_aggregation_by_years(series, num_years=5, agg_func=sum)
ss_10years: Series = ts_aggregation_by_years(series, num_years=10, agg_func=sum)

figure(figsize=(14, 6))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel="years",
    ylabel=target,
    title=f"Forecasting Europe yearly {target}",
)
# savefig(f"images/B/DP_DataDistribution/{file_tag}.png", bbox_inches="tight")

# Five Number Summary
from numpy import array
from matplotlib.pyplot import show, subplots
from matplotlib.figure import Figure
from dslabs_functions import set_chart_labels

# Criando a figura e os eixos
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Três gráficos lado a lado
fig.suptitle(f"Boxplot Analysis of Europe Forecasting: {target}", fontsize=14)

# Gráfico 1: Dados anuais
axs[0].boxplot(series)
axs[0].set_title("Yearly")
axs[0].set_ylabel("Values")
axs[0].grid(True)
axs[0].text(
    0.5, -0.2, str(series.describe()), fontsize=10, ha='center', va='top', transform=axs[0].transAxes
)

# Gráfico 2: Dados agregados a cada 5 anos
axs[1].boxplot(ss_5years)
axs[1].set_title("5-Year Aggregation")
axs[1].grid(True)
axs[1].text(
    0.5, -0.2, str(ss_5years.describe()), fontsize=10, ha='center', va='top', transform=axs[1].transAxes
)

# Gráfico 3: Dados agregados a cada 10 anos
axs[2].boxplot(ss_10years)
axs[2].set_title("10-Year Aggregation")
axs[2].grid(True)
axs[2].text(
    0.5, -0.2, str(ss_10years.describe()), fontsize=10, ha='center', va='top', transform=axs[2].transAxes
)

# Ajustando espaçamento
plt.subplots_adjust(bottom=0.3)  # Dá espaço para os textos na parte inferior

feature="5Number"
savefig(f"images/B/DP_DataDistribution/{file_tag}_{feature}_study.png", bbox_inches="tight")

grans: list[Series] = [series, ss_5years, ss_10years]
gran_names: list[str] = ["Yearly", "5Years", "10Years"]
fig: Figure
axs: array
fig, axs = subplots(1, len(grans), figsize=(14, 6))
fig.suptitle(f"Forecasting Europe {target}")
for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values)
feature="VarDistribution"
# savefig(f"images/B/DP_DataDistribution/{file_tag}_{feature}_study.png", bbox_inches="tight")

# AutoCorrelation
from dslabs_functions import plot_multiline_chart


def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series


plt.figure(figsize=(14, 6))
plt.suptitle(f"Distribution of lags for Europe Forecasting: {target}", fontsize=12)
lags = get_lagged_series(series, 20, 10)
plot_multiline_chart(series.index.to_list(), lags, xlabel=index, ylabel=target)
feature="AutoCorrelation"
# savefig(f"images/B/DP_DataDistribution/{file_tag}_{feature}.png", bbox_inches="tight")

# "AutoCorrelation Study"
from matplotlib.pyplot import setp
from matplotlib.gridspec import GridSpec


def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(14, 6), constrained_layout=True)
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
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    fig.suptitle(f"Autocorrelation for Europe {target}", fontsize=14)
    return

autocorrelation_study(series, 10, 1)
# savefig(f"images/B/DP_DataDistribution/{file_tag}_{feature}_study.png", bbox_inches="tight")