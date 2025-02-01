from pandas import read_csv, DataFrame, Series

from dslabs_functions import plot_line_chart, HEIGHT

filename = "datasets/0_1_forecast_ny_arrests.csv"
file_tag = "1_Dimen_Granular_1_forecasting_ny_arrests"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename,
    index_col="Date",
    sep=";",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,)

vars = data.columns.tolist()
target = 'Manhattan'
# ['Date;Bronx;Brooklyn;Queens;StatenIsland;Manhattan']

print("Nr. Records = ", data.shape)
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])


from matplotlib.axes import Axes
from matplotlib.pyplot import subplots, savefig, figure
from matplotlib.figure import Figure


def plot_ts_multivariate_chart(data: DataFrame, title: str) -> list[Axes]:
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(data.shape[1], 1, figsize=(3 * HEIGHT, HEIGHT / 2 * data.shape[1]))
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

plot_ts_multivariate_chart(data, title=f"Forecasting New York Arrests: Data Dimensionality")
savefig(f"images/B/DP_DataDimensionalityGranularity/1_Dimensionality_forec_ny_arrests.png", bbox_inches="tight")

from pandas import Index, Period


def ts_aggregation_by(
    data: Series | DataFrame,
    gran_level: str = "D",
    agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df

grans: list[str] = ["Q"] # ["W", "M", "Q", "Y"]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
fig.suptitle(f"NY arrests [{target}]: aggregation study")
series: Series = data[target]

# Garantir que axs seja uma lista, mesmo com um único subgráfico
if len(grans) == 1:
    axs = [axs]  # Transformar em lista, pois axs é um único objeto

for i in range(len(grans)):
    ss: Series = ts_aggregation_by(series, grans[i])
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        ax=axs[i],
        xlabel=f"{ss.index.name} ({grans[i]})",
        ylabel=target,
        title=f"granularity={grans[i]}",
    )
fig.tight_layout()
savefig(f"images/B/DP_DataDimensionalityGranularity/1_Granularity_{target}_{grans[0]}.png", bbox_inches="tight")