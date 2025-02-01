import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN, NR_STDEV

# Carregar os dados
filename = "datasets/0_2_class_financial_distress.csv"
subject = " [financial distress]"
file_tag = "2_3_MissingValuesImputation_2_financial_distress"
file_output = "datasets/1_2_3_outliers_treatment_class_financial_distress.csv"
data: pd.DataFrame = pd.read_csv(filename)

target = "CLASS"
columns = data.columns
col_non_outliers = ['Company', 'Time', 'CLASS']

col_outliers  = [col for col in columns if col not in col_non_outliers]

# MinMax Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data = pd.DataFrame(scaler.fit_transform(data), columns=columns)

NR_STDEV = 2.5

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom

data_prep = data.copy()
data_prep.head()

summary5: DataFrame = data_prep.describe()

for var in col_outliers:
    top, bottom = determine_outlier_thresholds_for_var(summary5[var])
    # median: float = data_prep[var].median()
    # data_prep[var] = data_prep[var].apply(lambda x: median if x > top or x < bottom else x)
    # Winsorization:
    # data_prep[var] = data_prep[var].apply(lambda x: top if x > top else (bottom if x < bottom else x))
    # Remover outliers fora do intervalo [bottom, top]
    data_prep = data_prep[(data_prep[var] >= bottom) & (data_prep[var] <= top)]


train, test = train_test_split(data_prep, test_size=0.3, random_state=42)

def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval

# "winsorization", "mean", "median", "o_removal"
imput = "o_removal"
plt.figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"MV Imputation eval. with " + imput + subject, percentage=True
)
plt.savefig(f"images/PreparationOutliersTreatment/{file_tag}_eval_" + imput + ".png", bbox_inches="tight")


data_prep.to_csv(file_output, index=False)

"""
# Imprimir as estatísticas descritivas em grupos de 10 colunas
for i in range(0, len(col_outliers), 5):
    group = col_outliers[i:i+5]  # Seleciona um grupo de 10 colunas
    print(f"\nColumns {i+1} to {min(i+5, len(col_outliers))}:")
    print("Data before replacing outliers:")
    print(data[group].describe())
    print("Data after replacing outliers:")
    print(data_prep[group].describe())
"""

import matplotlib.pyplot as plt
import pandas as pd


import matplotlib.pyplot as plt
import pandas as pd

# Função para gerar gráficos de barras (5x5 por bloco)
def plot_std_comparison(std_table, block_size=5):
    num_vars = len(std_table)

    # Definir o número de blocos (5x5 gráficos por bloco)
    num_blocks = (num_vars // (block_size * block_size)) + (1 if num_vars % (block_size * block_size) != 0 else 0)

    for block in range(num_blocks):
        # Selecionar as variáveis do bloco atual
        start = block * (block_size * block_size)
        end = min(start + (block_size * block_size), num_vars)
        block_data = std_table.iloc[start:end]

        # Calcular o número de linhas e colunas para o gráfico
        nrows = block_size
        ncols = block_size

        # Criar a figura para o bloco de gráficos (5x5)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        fig.suptitle(f"Standard Deviation Before and After Outlier Treatment ({block + 1})", fontsize=14, color='#4080bf')

        # Plotar o gráfico de barras para cada variável no bloco 5x5
        for i, row in enumerate(block_data.itertuples()):
            row_idx = i // ncols  # Índice da linha
            col_idx = i % ncols   # Índice da coluna
            ax = axes[row_idx, col_idx]  # Acessa a célula do gráfico

            ax.bar(["Before", "After"], [row._2, row._3], color=['#4080bf', '#F79522'])

            # Títulos e labels personalizados
            ax.set_title(row.Variable, color='#4080bf', fontsize=10)  # Título azul
            ax.set_ylim([0, max(row._2, row._3) * 1.2])  # Ajustar o limite do eixo Y

            # Ajustar a cor dos números do eixo Y para cinza
            for label in ax.get_yticklabels():
                label.set_color('gray')

            # Ajustar a cor das linhas do gráfico (bordas e eixos) para cinza
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')

            # Ajustar a cor dos valores do eixo X para azul
            for label in ax.get_xticklabels():
                label.set_color('#4080bf')  # Cor azul para os valores do eixo X

        # Remover os gráficos não utilizados no último bloco, se houver
        for j in range(len(block_data), nrows * ncols):
            fig.delaxes(axes.flatten()[j])

        plt.savefig(f"images/PreparationOutliersTreatment/outliers_comparison_block_{block + 1}.png", bbox_inches="tight")
"""
# Criar um dicionário para armazenar os resultados
std_comparison = {
    "Variable": [],
    "Std Before": [],
    "Std After": []
}

# Iterar por cada variável em col_outliers
for var in col_outliers:
    std_comparison["Variable"].append(var)
    std_comparison["Std Before"].append(data[var].std())
    std_comparison["Std After"].append(data_prep[var].std())

# Converter o dicionário em um DataFrame
std_table = pd.DataFrame(std_comparison)

# Exportar para um arquivo CSV
std_table.to_csv("datasets/datapreparation_outliers_std_comparison.csv", index=False)

print("Tabela exportada com sucesso para 'std_comparison.csv'")

# Gerar os gráficos
plot_std_comparison(std_table)
"""