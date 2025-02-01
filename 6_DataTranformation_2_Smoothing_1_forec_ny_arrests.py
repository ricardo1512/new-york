import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, figure, subplots
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart, ts_aggregation_by

filename = "datasets/2_1_aggregation_forecast_ny_arrests.csv"
file_output= "datasets/2_2_smoothing_forecast_ny_arrests.csv"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename,
    index_col="Date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,)

vars = data.columns.tolist()
target = 'Manhattan'
# ['Date;Bronx;Brooklyn;Queens;StatenIsland;Manhattan']
file_tag = "Forecasting:" + subject
series: Series = data[target]

sizes: list[int] = [2, 5, 10, 15]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
fig.suptitle(f"{file_tag} after smoothing")

for i in range(len(sizes)):
    ss_smooth: Series = series.rolling(window=sizes[i]).mean()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        ax=axs[i],
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"size={sizes[i]}",
    )

savefig(f"images/B/DT_Smoothing/1_Smoothing_{target}.png", bbox_inches="tight")


# Aplicar suavização com janela de tamanho 5 em todas as colunas (incluindo 'target') e substituir as variáveis
size = 2

for var in data.columns:
    data[var] = data[var].rolling(window=size).mean()

plot_ts_multivariate_chart(data, title=f"{file_tag} after smothing: size=2")

savefig(f"images/B/DT_Smoothing/1_Smoothing_{target}_multivariate_size_2.png", bbox_inches="tight")

data = data.dropna(axis=0)

# Exportar o DataFrame suavizado para o arquivo CSV
data.to_csv(file_output, index_label="Date")