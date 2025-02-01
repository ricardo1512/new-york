import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from sklearn.model_selection import train_test_split
from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import savefig, show, figure
from sklearn.preprocessing import MinMaxScaler

from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN
from dslabs_functions import plot_bar_chart

# Carregar os dados
filename = "datasets/1_1_1_variables_encoding_class_ny_arrests.csv"
subject = " [ny arrests]"
file_tag = "2_2_MissingValuesImputation_1_class_ny_arrests"
file_output = "datasets/1_1_2_missing_values_imputation_class_ny_arrests.csv"
data: pd.DataFrame = pd.read_csv(filename, index_col="ARREST_KEY", na_values=[""])

# Obter uma amostra de 10% dos dados
data = data.sample(frac=1, random_state=42)

data = data.drop(['PD_DESC', 'OFNS_DESC', 'Latitude', 'Longitude'], axis=1)
target = "CLASS"
"""
numeric = data.select_dtypes(include=['number']).columns
binary = [col for col in numeric if data[col].nunique() == 2]
numeric = [col for col in numeric if col not in binary] + [target]
symbolic = data.select_dtypes(include=['object', 'string']).columns
"""
# data = data[numeric]

# MinMax Normalization
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = pd.DataFrame(scaler.fit_transform(data), columns=numeric)

import pandas as pd
from sklearn.impute import SimpleImputer


# mean, median, most_frequent, drop
# imput = 'mean'

# Criar um SimpleImputer que usa a mediana
# imputer = SimpleImputer(strategy=imput)

# Aplicar o imputer nas colunas numéricas
# data_imputed  = imputer.fit_transform(data)
# data_imputed = pd.DataFrame(data_imputed, columns=data.columns)



# Combinar todas as categorias em uma lista
# columns_to_check = list(numeric) + list(symbolic)
imput = "mv_remove"
data_imputed = data.dropna(axis=0)
data_imputed.to_csv(file_output, index=False)

"""
# Contar o número de linhas com pelo menos um NaN
num_missing_rows = data[columns_to_check].isna().any(axis=1).sum()

# Calcular o total de linhas
total_rows = data.shape[0]

# Calcular a percentagem de linhas com pelo menos um NaN
missing_percentage = (num_missing_rows / total_rows) * 100

print(f"Number of rows with at least one missing value: {num_missing_rows}")
print(f"Percentage of rows with at least one missing value: {missing_percentage:.2f}%")
"""
"""
# Imprimir o shape do DataFrame antes de eliminar os missing values
print("Shape before cleaning:", data.shape)
print()
# Imprimir o número de itens com missing values por coluna antes da limpeza
missing_before = data[columns_to_check].isna().sum()
print("Missing values before cleaning:\n", missing_before)

# Calcular o número total de itens com missing values (número de linhas com ao menos um NaN)
total_missing_before = data.isna().sum(axis=1).gt(0).sum()
print(f"Total number of rows with missing values: {total_missing_before}")

# Eliminar as linhas com pelo menos um NaN nas colunas especificadas
data_cleaned = data.dropna(subset=columns_to_check)

# Imprimir o shape do DataFrame após eliminar os missing values
print("Shape after cleaning:", data_cleaned.shape)
print()
# Calcular o número de linhas removidas
rows_removed = data.shape[0] - data_cleaned.shape[0]
print(f"Total number of rows removed: {rows_removed}")

# Imprimir o número de itens com missing values por coluna após a limpeza
missing_after = data_cleaned[columns_to_check].isna().sum()
print("Missing values after cleaning:\n", missing_after)

# Salvar o novo dataset sem valores ausentes
data_cleaned.to_csv(file_output, index=False)
"""
"""
# Contagem de valores ausentes em cada coluna
mv: dict[str, int] = {}
for var in columns_to_check:
    # Somar os valores NaN na coluna 'var'
    nr = data_cleaned[var].isna().sum()
    if nr > 0:
        mv[var] = nr


figure()
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable" + subject,
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"images/PreparationMissingValueImputation/{file_tag}_mv.png", bbox_inches='tight')
show()
"""

"""

# Realizar o train-test split (70% treino e 30% teste)
train, test = train_test_split(data_imputed, test_size=0.3, random_state=42)

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



figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"MV Imputation eval. with " + imput + subject, percentage=True
)
savefig(f"images/PreparationMissingValueImputation/{file_tag}_eval_" + imput + ".png", bbox_inches="tight")
"""