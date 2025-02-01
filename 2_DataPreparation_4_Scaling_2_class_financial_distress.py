import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import subplots, show, savefig, figure
import numpy as np

from dslabs_functions import run_NB, run_KNN, CLASS_EVAL_METRICS, plot_multibar_chart

# Carregar os dados
filename = "datasets/1_2_3_outliers_treatment_class_financial_distress.csv"
subject = " [financial_distress]"
file_tag = "2_4_Scaling_original_zscore_minmax_2_financial_distress"
file_output = "datasets/1_2_4_scaling_class_financial_distress.csv"
data: pd.DataFrame = pd.read_csv(filename)

vars_not_to_scale: list[str] = ['CLASS']

vars_to_scale = [var for var in data.columns if var not in vars_not_to_scale]

# Dados originais
original = data[vars_to_scale]

# Z-Score Normalization
scaler = StandardScaler(with_mean=True, with_std=True)
data_zscore = pd.DataFrame(scaler.fit_transform(data[vars_to_scale]), columns=vars_to_scale)
# Garantir valores não negativos (shift positivo)
data_zscore = data_zscore - np.min(data_zscore)

# MinMax Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data_minmax = pd.DataFrame(scaler.fit_transform(data[vars_to_scale]), columns=vars_to_scale)

"""
# Criar uma figura com 3 subplots empilhados verticalmente
fig, axs = subplots(3, 1, figsize=(10, 15), squeeze=False)

# Plotar os dados em cada subplot
axs[0, 0].set_title("Original Data")
original.boxplot(ax=axs[0, 0])

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

"""

data.reset_index(drop=True, inplace=True)
data_minmax.reset_index(drop=True, inplace=True)
# Realizar o train-test split (70% treino e 30% teste)
data = pd.concat([data['CLASS'], data_minmax[vars_to_scale]], axis=1)

train, test = train_test_split(data, test_size=0.3, random_state=42)

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

"""
target = "CLASS"
imput = "z_score"
figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Scaling eval. with " + imput + subject, percentage=True
)
savefig(f"images/PreparationScaling/{file_tag}_eval_" + imput + ".png", bbox_inches="tight")
"""

# data[vars_to_scale] = data_minmax.copy()
data.to_csv(file_output, index=False)