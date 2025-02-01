from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig, show, subplots, figure
from numpy import ndarray
from matplotlib.figure import Figure
from dslabs_functions import plot_multi_scatters_chart, plot_bar_chart, HEIGHT
import matplotlib.pyplot as plt
import pandas as pd

# Carregar os dados
filename = "datasets/0_1b_class_ny_arrests.csv"
file_tag = "1_Distribution_1_class_ny_arrests"
subject = " [ny arrests]"
data: pd.DataFrame = pd.read_csv(filename, index_col="ARREST_KEY", na_values="")
data = data.dropna()  # Remover as linhas com valores ausentes
target = "CLASS"

# Filtrar apenas colunas numéricas
numeric_vars = data.select_dtypes(include=["number"]).columns.to_list()
# Converter a coluna LAW_CAT_CD em binário (M = 1, F = 0)
data["PERP_SEX"] = data["PERP_SEX"].replace({"M": 1, "F": 0})

# Converter a coluna LAW_CAT_CD em binário (M = 1, F = 0)
data["CLASS"] = data["CLASS"].replace({"nonNY": 1, "NY": 0})

# Combinar variáveis numéricas e binárias
vars = numeric_vars + ["CLASS"]

# Exibir as variáveis selecionadas
print(f"Variáveis selecionadas: {vars}")



group_size = 5  # Tamanho do grupo de variáveis
num_groups = (len(vars) + group_size - 1) // group_size  # Número de grupos

# Loop para dividir e plotar os grupos
for group_index in range(num_groups):
    print(group_index + 1, "grupo")
    # Determina o grupo de variáveis
    target = "CLASS"
    start_index = group_index * group_size
    end_index = min((group_index + 1) * group_size, len(vars))
    group_vars = vars[start_index:end_index] + [target]

    # Criação da figura e subgráficos para o grupo atual
    n: int = len(group_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    
    variables_text = ", ".join(group_vars)  # Juntar os nomes das variáveis com vírgulas
    fig.suptitle(f"Scatter Plot Matrix for Variables: {variables_text}" + subject, fontsize=14)
    
    # Geração dos gráficos para o grupo
    for i in range(len(group_vars)):
        print(i)
        var1: str = group_vars[i]
        for j in range(i + 1, len(group_vars)):
            print(j)
            var2: str = group_vars[j]
            plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
    print(f"Salvando figura...")
    # Salva a figura do grupo
    savefig(f"images/ProfilingSparcity/{file_tag}_sparsity_study_group_{group_index + 1}.png", bbox_inches="tight")
    plt.close(fig)


from seaborn import heatmap

corr_mtx: DataFrame = data[vars].corr().abs()

fig = figure(figsize=(10, 8))

heatmap(
    abs(corr_mtx),
    xticklabels=vars,
    yticklabels=vars,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
# Adicionar título
fig.suptitle("Correlation Matrix Heatmap" + subject, fontsize=14)

savefig(f"images/ProfilingSparcity/{file_tag}_correlation_analysis.png", bbox_inches="tight")
# show()