from pandas import read_csv, DataFrame, Series

from dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by

filename = "datasets/0_1_forecast_ny_arrests.csv"
file_tag = "1_Distribution_1_forecasting_ny_arrests"
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
series: Series = data[target]

from pandas import Series
from matplotlib.pyplot import subplots, show, gca, savefig
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from dslabs_functions import HEIGHT, set_chart_labels
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show
from dslabs_functions import plot_line_chart

def plot_components(
    series: Series,
    title: str = "",
    x_label: str = "time",
    y_label: str = "",
) -> list[Axes]:
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return axs



plot_components(
    series,
    title=f"Components: daily, {target}" + subject,
    x_label=series.index.name,
    y_label=target,
)

# savefig(f"images/B/DP_DataStationarity/1_Stationarity_components_{target}.png", bbox_inches="tight")


from matplotlib.pyplot import plot, legend

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"Stationary study, {target}" + subject,
    name="original",
)
n: int = len(series)
plot(series.index, [series.mean()] * n, "r-", label="mean")
legend()
# savefig(f"images/B/DP_DataStationarity/1_Stationarity_stationarity_study_{target}_a.png", bbox_inches="tight")


from matplotlib.pyplot import plot, legend
n: int = len(series)
BINS = 10
mean_line: list[float] = []

for i in range(BINS):
    segment: Series = series[i * n // BINS: (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"Stationary study, {target}" + subject,
    name="original",
    show_stdev=True,
)
n: int = len(series)
plot(series.index, mean_line, "r-", label="mean")
legend()
# savefig(f"images/B/DP_DataStationarity/1_Stationarity_stationarity_study_{target}_b.png", bbox_inches="tight")



from statsmodels.tsa.stattools import adfuller

def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05

print(f"The series of {target} {('is' if eval_stationarity(series) else 'is not')} stationary")