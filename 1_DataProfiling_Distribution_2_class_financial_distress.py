from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show, xticks
from dslabs_functions import plot_bar_chart, get_variable_types

filename = "datasets/0_2_class_financial_distress.csv"
file_tag = "1_Distribution_2_class_financial_distress"
subject = " [financial distress]"
data: DataFrame = read_csv(filename, na_values="")


variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

"""
if numeric:  # Verifica se há variáveis numéricas
    ax = data[numeric].boxplot(rot=45)
    ax.set_title("Global Boxplot for Numeric Variables" + subject)  # Define o título
    savefig(f"images/ProfilingDistribution/{file_tag}_global_boxplot.png", bbox_inches='tight')
    show()
else:
    print("There are no numeric variables.")

"""

"""
from numpy import ndarray
from matplotlib.pyplot import savefig, subplots
from math import ceil
from dslabs_functions import HEIGHT, get_variable_types

if [] != numeric:
    cols = 3  # Fixar número de colunas como 3
    total_groups = ceil(len(numeric) / cols)  # Número total de grupos (arquivos)

    for group in range(total_groups):
        # Definir os índices das variáveis no grupo atual
        start_idx = group * cols
        end_idx = min(start_idx + cols, len(numeric))
        current_variables = numeric[start_idx:end_idx]

        # Criar figura e subplots para o grupo atual
        rows = 1
        fig, axs = subplots(
            rows,
            cols,
            figsize=(cols * HEIGHT, rows * HEIGHT),
            squeeze=False,
        )

        # Plotar gráficos para as variáveis do grupo atual
        for i, variable in enumerate(current_variables):
            axs[0, i].set_title(f"Boxplot for {variable}" + subject)
            axs[0, i].boxplot(data[variable].dropna().values)

        # Remover subplots vazios se o número de variáveis for menor que 3
        for i in range(len(current_variables), cols):
            fig.delaxes(axs[0, i])

        # Salvar o arquivo PNG para o grupo atual
        savefig(f"images/ProfilingDistribution/{file_tag}_boxplots_group_{group + 1}.png", bbox_inches="tight")
        print(f"Saved: {file_tag}_boxplots_group_{group + 1}.png")

else:
    print("There are no numeric variables.")
"""


from pandas import Series
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_multibar_chart, HEIGHT

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

NR_STDEV: int = 3
IQR_FACTOR: float = 3

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


def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: int = NR_STDEV,
    iqrfactor: float = IQR_FACTOR,
) -> dict:
    outliers_iqr: list = []
    outliers_stdev: list = []
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        top: float
        bottom: float
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}


if [] != numeric:
    outliers: dict[str, int] = count_outliers(data, numeric)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of standard outliers per variable" + subject,
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    # Rotaciona os rótulos do eixo X em 45 graus
    xticks(rotation=90)

    savefig(f"images/ProfilingDistribution/{file_tag}_outliers_standard.png", bbox_inches="tight")
    show()
else:
    print("There are no numeric variables.")


from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, subplots
from dslabs_functions import set_chart_labels, HEIGHT
from scipy.stats import norm, expon, lognorm
from pandas import Series
from math import log

"""
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions["LogNor(%.1f,%.2f)" % (log(scale), sigma)] = lognorm.pdf(
        x_values, sigma, loc, scale
    )
    return distributions


def histogram_with_distributions(ax, series: Series, var: str):
    values: list = series.sort_values().to_list()
    ax.hist(values, bins="sturges", density=True)  # 30 bins como solicitado
    distributions: dict = compute_known_distributions(values)
    
    # Plotar as distribuições ajustadas
    for label, dist in distributions.items():
        ax.plot(values, dist, label=label)
    
    ax.legend(loc="best", fontsize=6)
    ax.set_title(f"Best fit for {var}" + subject)
    ax.set_xlabel(var)
    ax.set_ylabel("Density")


# Obtendo as variáveis numéricas
variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

if numeric:  # Verifica se a lista não está vazia
    group_size = 3  # Número de gráficos por imagem
    rows, cols = 1, 3  # Configuração: 1 linha e 3 colunas
    
    # Loop para dividir os gráficos em grupos
    for group_start in range(0, len(numeric), group_size):
        print(f"Processando grupo {group_start // group_size + 1} com índices de {group_start} a {group_start + group_size}")
        
        # Criar nova figura e eixos
        fig: Figure
        axs: ndarray
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        
        # Criar os gráficos do grupo atual
        for idx, col_name in enumerate(numeric[group_start:group_start + group_size]):
            i, j = divmod(idx, cols)  # Localizar posição (linha, coluna)
            set_chart_labels(
                axs[i, j],
                title=f"Histogram for {col_name}" + subject,
                xlabel=col_name,
                ylabel="nr records",
            )
            histogram_with_distributions(axs[i, j], data[col_name].dropna(), col_name)
        
        # Remover eixos vazios no último grupo (se necessário)
        for idx in range(len(numeric[group_start:group_start + group_size]), group_size):
            if axs[0, idx]:  # Verifica se o eixo existe
                fig.delaxes(axs[0, idx])
        
        # Salvar a imagem para o grupo atual
        group_number = group_start // group_size + 1
        savefig(f"images/ProfilingDistribution/{file_tag}_histograms_group_{group_number}.png", bbox_inches="tight")
        print(f"{group_number}.º grupo criado!")
else:
    print("There are no numeric variables.")

"""
from matplotlib.pyplot import subplots, savefig, show
from pandas import Series
from dslabs_functions import plot_bar_chart

"""
# Obtendo as variáveis simbólicas
variables_types: dict[str, list] = get_variable_types(data)
symbolic: list[str] = variables_types["binary"]

if symbolic:  # Verifica se a lista não está vazia
    # Definir o número de colunas fixo
    cols = 3
    
    # Calcular o número de linhas necessárias
    rows = (len(symbolic) // cols) + (1 if len(symbolic) % cols != 0 else 0)
    
    # Criar a figura e os eixos
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    
    # Loop para criar os gráficos, sem necessidade de índices manuais
    for n, var in enumerate(symbolic):
        counts: Series = data[var].value_counts()
        row, col = divmod(n, cols)  # Dividir o índice para determinar a linha e a coluna
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[row, col],
            title=f"Histogram for {var}" + subject,
            xlabel=var,
            ylabel="nr records",
            percentage=False,
        )
    
    # Remover eixos extras caso haja
    for idx in range(len(symbolic), rows * cols):
        fig.delaxes(axs.flatten()[idx])
    
    # Salvar a imagem com os subgráficos
    savefig(f"images/ProfilingDistribution/{file_tag}_histograms_bynary.png")
    show()
else:
    print("There are no symbolic variables.")
"""

"""
target = "CLASS"

values: Series = data[target].value_counts()
# print(values)

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})" + subject,
)
savefig(f"images/ProfilingDistribution/{file_tag}_class_distribution.png", bbox_inches="tight")
show()
"""

"""
variables = numeric

# Configuração para agrupamento de 3 gráficos por imagem
group_size = 3
rows, cols = 1, 3  # 1 linha e 3 colunas por imagem

# Verifica se há variáveis a serem plotadas
if variables:
    for group_start in range(0, len(variables), group_size):
        # Criar nova figura para o grupo
        fig, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

        # Loop para plotar os gráficos do grupo
        for idx, var in enumerate(variables[group_start:group_start + group_size]):
            i, j = divmod(idx, cols)  # Calcula a posição da célula (linha, coluna)
            if var in numeric:  # Para variáveis numéricas, cria um histograma
                axs[i, j].hist(data[var].dropna().values, bins=20)
                axs[i, j].set_title(f"Histogram for {var}" + subject)
                axs[i, j].set_xlabel(var)
                axs[i, j].set_ylabel("nr records")
            
            elif var in binary or var in symbolic:  # Para variáveis binárias ou simbólicas, cria um gráfico de barras
                counts: Series = data[var].value_counts()
                plot_bar_chart(
                    counts.index.to_list(),
                    counts.to_list(),
                    ax=axs[i, j],
                    title=f"Histogram for {var}" + subject,
                    xlabel=var,
                    ylabel="nr records",
                    percentage=False,
                )
            
        # Remover eixos extras no último grupo (se necessário)
        for idx in range(len(variables[group_start:group_start + group_size]), group_size):
            fig.delaxes(axs[0, idx])

        # Salvar a imagem para o grupo atual
        group_number = group_start // group_size + 1
        savefig(f"images/ProfilingDistribution/DistrNumeric/{file_tag}_histograms_group_{group_number}.png", bbox_inches="tight")
        print(f"{group_number}.º grupo criado!")
    show()
else:
    print("There are no variables to plot.")

"""