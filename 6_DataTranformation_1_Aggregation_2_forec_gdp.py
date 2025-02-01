import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig, figure
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart, ts_aggregation_by

filename = "datasets/0_2b_forecast_gdp_europe.csv"
file_output= "datasets/2_1_aggregation_forecast_gdp_europe.csv"
subject = " [gdp europe]"
data: DataFrame = read_csv(filename,
    index_col="Year",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,)

vars = data.columns.tolist()
target = 'GDP'
# ['Date;Bronx;Brooklyn;Queens;StatenIsland;Manhattan']
file_tag = "Forecasting:" + subject

agg_df = data.resample('1Y').agg('mean').reset_index()

agg_df['Year'] = agg_df['Year'].dt.year

# Define o 'Year' como índice para garantir que seja usado no eixo X
agg_df.set_index('Year', inplace=True)

series: Series = agg_df[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} 1-year",
)

savefig(f"images/B/DT_Aggregation/1_Aggregation_{target}_target_1-year_mean.png", bbox_inches="tight")

plot_ts_multivariate_chart(agg_df, title=f"{file_tag} after a 1-year aggregation")

savefig(f"images/B/DT_Aggregation/1_Aggregation_{target}_1years_mean.png", bbox_inches="tight")

agg_df.to_csv(file_output, index_label="Year")