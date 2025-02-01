import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import subplots, show, savefig, figure

from dslabs_functions import run_NB, run_KNN, CLASS_EVAL_METRICS, plot_multibar_chart

# Carregar os dados
filename = "datasets/1_1_2_missing_values_imputation_class_ny_arrests.csv"
subject = " [ny arrests]"
file_tag = "2_4_Scaling_original_zscore_minmax_1_class_ny_arrests"
file_output = "datasets/1_1_4_scaling_class_ny_arrests.csv"
data: pd.DataFrame = pd.read_csv(filename)
target = 'CLASS'

numeric = data.select_dtypes(include=['number']).columns
numeric = list(numeric)
print(numeric)
vars_to_scale = [col for col in numeric if col != target]


# Obter uma amostra de 10% dos dados
data_sample = data.sample(frac=1, random_state=42)
# Z-Score Normalization
scaler = StandardScaler(with_mean=True, with_std=True)
data_sample[vars_to_scale] = scaler.fit_transform(data_sample[vars_to_scale])
# Garantir valores não negativos (shift positivo)
data_sample[vars_to_scale] = data_sample[vars_to_scale] - data_sample[vars_to_scale].min()

# MinMax Normalization
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_sample[vars_to_scale] = scaler.fit_transform(data_sample[vars_to_scale])

"""
# Criar uma figura com 3 subplots empilhados verticalmente
fig, axs = subplots(3, 1, figsize=(10, 15), squeeze=False)

# Plotar os dados em cada subplot
axs[0, 0].set_title("Original Data")
data_sample.boxplot(ax=axs[0, 0])

axs[1, 0].set_title("Z-Score Normalization")
data_zscore.boxplot(ax=axs[1, 0])

axs[2, 0].set_title("MinMax Normalization")
data_minmax.boxplot(ax=axs[2, 0])

# Ajustar o espaçamento entre os subplots
fig.subplots_adjust(hspace=0.5)  # Controla o espaço vertical entre os gráficos

# Ajustar os rótulos do eixo x para todos os subplots
for ax in axs[:, 0]:
    ax.tick_params(axis='x', rotation=90, labelsize=7)

# Salvar a figura
savefig(f"images/PreparationScaling/{file_tag}.png", bbox_inches="tight")

data[vars_to_scale] = data_minmax.copy()
data.to_csv(file_output, index=False)
"""
"""
# Resetar índices para alinhamento correto
data_sample.reset_index(drop=True, inplace=True)

# data_concat = pd.concat([data_sample['CLASS'], data_zscore[vars_to_scale]], axis=1)
# print(data['CLASS'].shape)
# print(data_zscore[vars_to_scale].shape)
train, test = train_test_split(data_sample[numeric], test_size=0.3, random_state=42)

def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: np.ndarray = train.values
    tstY = test.pop(target).values
    tstX: np.ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval


imput = "min_max"
figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Scaling eval. with " + imput + subject, percentage=True
)
savefig(f"images/PreparationScaling/{file_tag}_eval_" + imput + ".png", bbox_inches="tight")
"""

data_sample.to_csv(file_output, index=False)
