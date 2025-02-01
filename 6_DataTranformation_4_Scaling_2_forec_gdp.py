import pandas as pd
from matplotlib.pyplot import figure, savefig
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_ts_multivariate_chart, HEIGHT, plot_line_chart

filename = "datasets/2_3_differentiation_forecast_gdp_europe.csv"
file_output= "datasets/2_4_scaling_forecast_gdp_europe.csv"
subject = " [GDP Europe]"

# Ler o arquivo CSV
data: DataFrame = read_csv(filename, index_col="Year")
data.index = pd.to_datetime(data.index)

# Variáveis e alvo
vars = data.columns.tolist()
target = 'GDP'
file_tag = "Forecasting:" + subject

# Criar o objeto MinMaxScaler
scaler = MinMaxScaler()

# Aplicar Min-Max Scaling
df_minmax = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

series: Series = df_minmax[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} scaling",
)
savefig(f"images/B/DT_Scaling/2_Scaling_{target}_uni.png", bbox_inches="tight")

# Plotar o gráfico com os dados normalizados
plot_ts_multivariate_chart(df_minmax, title=f"{file_tag} after MinMax Scaling")
plt.savefig(f"images/B/DT_Scaling/2_Scaling_{target}_minmax_multivariate.png", bbox_inches="tight")

# Exportar o DataFrame escalado para o arquivo CSV
df_minmax.to_csv(file_output, index_label="Year")