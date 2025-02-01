from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT

file_tag = "2_2_DataProfiling_2_GDP_Europe_Granularity"
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

figure(figsize=(14, 6))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} yearly {target}",
)
savefig(f"images/B/DP_DataDimensionalityGranularity/{file_tag}.png", bbox_inches="tight")

# Aggregation
import pandas as pd

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


ss_years: Series = ts_aggregation_by_years(series, num_years=5,agg_func="mean")
figure(figsize=(14, 6))
plot_line_chart(
    ss_years.index.to_list(),
    ss_years.to_list(),
    xlabel="5 years",
    ylabel=target,
    title=f"Forecasting GDP Europe: Data Granularity, 5 years mean",
)
feature="Aggregation"
savefig(f"images/B/DP_DataDimensionalityGranularity/{file_tag}_{feature}_5.png", bbox_inches="tight")

ss_years: Series = ts_aggregation_by_years(series, num_years=10,agg_func="mean")
figure(figsize=(14, 6))
plot_line_chart(
    ss_years.index.to_list(),
    ss_years.to_list(),
    xlabel="5 years",
    ylabel=target,
    title=f"Forecasting GDP Europe: Data Granularity, 10 years mean",
)
feature="Aggregation"
savefig(f"images/B/DP_DataDimensionalityGranularity/{file_tag}_{feature}_10.png", bbox_inches="tight")

# Aggregation Study
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure

grans: list[str] = [10]# [1, 5, 10]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(grans), 1, figsize=(14, 6))
fig.suptitle(f"{file_tag} {target} aggregation study")
fig.suptitle(f"Forecasting GDP Europe: aggregation study")

# Garantir que axs seja uma lista, mesmo com um único subgráfico
if len(grans) == 1:
    axs = [axs]  # Transformar em lista, pois axs é um único objeto

for i in range(len(grans)):
    ss: Series = ts_aggregation_by_years(series, grans[i], agg_func="mean")
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        ax=axs[i],
        xlabel=f"{ss.index.name} ({grans[i]})",
        ylabel=target,
        title=f"granularity Years={grans[i]}",
    )
    savefig(f"images/B/DP_DataDimensionalityGranularity/{file_tag}_{feature}_study_10.png", bbox_inches="tight")