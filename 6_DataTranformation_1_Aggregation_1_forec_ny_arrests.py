import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig, figure
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart, ts_aggregation_by

filename = "datasets/0_1_forecast_ny_arrests.csv"
file_output= "datasets/2_1_aggregation_forecast_ny_arrests.csv"
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
file_tag = "Forecasting:" + subject


agg_df: DataFrame = ts_aggregation_by(data, gran_level="M", agg_func="sum")

series: Series = agg_df[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} monthly",
)

savefig(f"images/B/DT_Aggregation/1_Aggregation_{target}_target_monthly_sum.png", bbox_inches="tight")

plot_ts_multivariate_chart(agg_df, title=f"{file_tag} after monthly aggregation")

savefig(f"images/B/DT_Aggregation/1_Aggregation_{target}_multivariate_monthly_sum.png", bbox_inches="tight")

agg_df.to_csv(file_output, index_label="Date")