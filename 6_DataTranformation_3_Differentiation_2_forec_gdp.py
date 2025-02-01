import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, figure, subplots
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart, ts_aggregation_by

filename = "datasets/2_2_smoothing_forecast_gdp_europe.csv"
file_output= "datasets/2_3_differentiation_forecast_gdp_europe.csv"
subject = " [GDP Europe]"

data: DataFrame = read_csv(filename,
    index_col="Year")
data.index = pd.to_datetime(data.index)

vars = data.columns.tolist()
target = 'GDP'
file_tag = "Forecasting:" + subject

diff_df: DataFrame = data
"""
diff_df: DataFrame = data.diff()

series: Series = diff_df[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} first differentiation",
)
savefig(f"images/B/DT_Differentiation/1_Differentiation_{target}_first.png", bbox_inches="tight")

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} after first differentiation")
savefig(f"images/B/DT_Differentiation/1_Differentiation_{target}_first_multivariate.png", bbox_inches="tight")


diff_df: DataFrame = diff_df.diff()

series: Series = diff_df[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} second differentiation",
)
savefig(f"images/B/DT_Differentiation/1_Differentiation_{target}_second.png", bbox_inches="tight")

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} after second differentiation")
savefig(f"images/B/DT_Differentiation/1_Differentiation_{target}_second_multivariate.png", bbox_inches="tight")
"""
diff_df = diff_df.dropna(axis=0)

# Exportar o DataFrame suavizado para o arquivo CSV
diff_df.to_csv(file_output, index_label="Year")
