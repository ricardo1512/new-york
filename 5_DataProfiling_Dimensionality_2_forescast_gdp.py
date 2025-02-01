from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT

file_tag = "2_1_DataProfiling_2_GDP_Europe_Dimensionality"
target = "GDP"
data: DataFrame = read_csv(
    "datasets/0_2_forecast_gdp_europe.csv",
    index_col="Year",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

figure(figsize=(14, 6))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} yearly {target}",
)
savefig(f"images/B/DP_DataDimensionalityGranularity/{file_tag}_study.png", bbox_inches="tight")

from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from matplotlib.figure import Figure

# Multivariate TimeSeries
def plot_ts_multivariate_chart(data: DataFrame, title: str) -> list[Axes]:
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(data.shape[1], 1, figsize=(14, 6))
    fig.suptitle(title)

    for i in range(data.shape[1]):
        col: str = data.columns[i]
        plot_line_chart(
            data[col].index.to_list(),
            data[col].to_list(),
            ax=axs[i],
            xlabel=data.index.name,
            ylabel=col,
        )
    return axs

target = "GDP"
data: DataFrame = read_csv(
    "datasets/0_2_forecast_gdp_europe.csv",
    index_col="Year",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
print("Nr. Records = ", data.shape)
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])

feature="MultiVariate"
plot_ts_multivariate_chart(data, title=f"Forecasting GDP Europe: Data Dimensionality")
savefig(f"images/B/DP_DataDimensionalityGranularity/{file_tag}_{feature}_study.png", bbox_inches="tight")