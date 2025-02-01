from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig, show, subplots, figure
from dslabs_functions import get_variable_types, plot_bar_chart
from pandas import DataFrame
from numpy import ndarray
from matplotlib.figure import Figure
from dslabs_functions import get_variable_types, plot_bar_chart, HEIGHT
import matplotlib.pyplot as plt


import pandas as pd

def derive_coordinates_dms(df: DataFrame, coord_vars: list[str], round_to: int = 2) -> DataFrame:
    for coord in coord_vars:
        print("calculando: ", coord)
        # Calculate degrees, minutes, and seconds for each coordinate value
        df[coord + "_degrees"] = df[coord].apply(lambda x: int(x))  # Get degrees (integer part)
        
        # Calculate minutes (after removing the degrees part)
        df[coord + "_minutes"] = df[coord].apply(lambda x: int((x - int(x)) * 60))  # Get minutes
        
        # Calculate seconds (remaining part after degrees and minutes)
        df[coord + "_seconds"] = df[coord].apply(lambda x: round(((x - int(x)) * 60 - int((x - int(x)) * 60)) * 60, round_to))  # Get seconds
        
        # Round the components to the specified precision
        df[coord + "_degrees"] = df[coord + "_degrees"].astype(int)
        df[coord + "_minutes"] = df[coord + "_minutes"].astype(int)
        df[coord + "_seconds"] = df[coord + "_seconds"].round(round_to).astype(float)

    return df


filename = "datasets/0_1b_class_ny_arrests.csv"
file_tag = "1_Distribution_1_class_ny_arrests"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename, index_col="ARREST_KEY")
coord_vars  = ['Latitude', 'Longitude']

data = data.dropna(subset=coord_vars)
#data = data.sample(n=100000)
data_dms = derive_coordinates_dms(data, coord_vars)



def plot_coordinates_dms(df: pd.DataFrame, coord: str):
    print("open plot: ", coord)
    
    # Crie os três componentes para as coordenadas
    degrees = df[coord + "_degrees"]
    minutes = df[coord + "_minutes"]
    seconds = df[coord + "_seconds"]

    # Contar a quantidade de itens por valor único
    degree_counts = degrees.value_counts().sort_index()
    minute_counts = minutes.value_counts().sort_index()
    second_counts = seconds.value_counts().sort_index()

    # Criar os subgráficos
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(f"Granularity study for {coord}" + subject, fontsize=14)

    # Graus
    axs[0].bar(degree_counts.index, degree_counts.values)
    axs[0].set_title("Degrees")
    axs[0].set_ylabel("Count")
    axs[0].set_xlabel("Degrees")

    # Minutos
    axs[1].bar(minute_counts.index, minute_counts.values)
    axs[1].set_title("Minutes")
    axs[1].set_ylabel("Count")
    axs[1].set_xlabel("Minutes")

    # Segundos
    axs[2].bar(second_counts.index, second_counts.values)
    axs[2].set_title("Seconds")
    axs[2].set_ylabel("Count")
    axs[2].set_xlabel("Seconds")

    # Ajuste os rótulos do eixo x para cada gráfico
    for ax in axs:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"{int(tick)}" for tick in ax.get_xticks()], rotation=45)

    # Salvar o gráfico
    plt.tight_layout()
    savefig(f"images/ProfilingGranularity/{file_tag}_granularity_{coord}.png", bbox_inches="tight")
    
    # Mostrar o gráfico
    plt.show()

# Plotando para latitude e longitude
plot_coordinates_dms(data_dms, "Latitude")
plot_coordinates_dms(data_dms, "Longitude")

"""
def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    for date in date_vars:
        df[date + "_year"] = df[date].dt.year
        df[date + "_quarter"] = df[date].dt.quarter
        df[date + "_month"] = df[date].dt.month
        df[date + "_day"] = df[date].dt.day
    return df

def analyse_date_granularity(data: DataFrame, var: str, levels: list[str], subject: str) -> ndarray:
    cols: int = len(levels)
    fig: Figure
    axs: ndarray
    rows, cols = 2, 2  # Layout ajustado para 4 gráficos
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {var}" + subject)

    for idx, level in enumerate(levels):
        i, j = divmod(idx, cols)  # Determina a linha e a coluna com base no índice
        counts: Series[int] = data[f"{var}_{level}"].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title=level,
            xlabel=level,
            ylabel="nr records",
            percentage=False,
        )
        axs[i, j].set_xticklabels(axs[i, j].get_xticklabels(), rotation=45)
    return axs

filename = "datasets/0_1_class_ny_arrests.csv"
file_tag = "1_Distribution_1_class_ny_arrests"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename, index_col="ARREST_KEY", parse_dates=True, dayfirst=True)

variables_types: dict[str, list] = get_variable_types(data)
data_ext: DataFrame = derive_date_variables(data, variables_types["date"])

for v_date in variables_types["date"]:
    analyse_date_granularity(data, v_date, ["year", "quarter", "month", "day"], subject)
    savefig(f"images/ProfilingGranularity/{file_tag}_granularity_{v_date}.png", bbox_inches="tight")
    show()
"""
