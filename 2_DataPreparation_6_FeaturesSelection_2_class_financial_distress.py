from pandas import Series, DataFrame, read_csv, Index
import pandas as pd

# Carregar os dados
filename_TRAIN = "datasets/1_2_5_balancing_TRAIN_class_financial_distress.csv"
filename_TEST = "datasets/1_2_5_balancing_TEST_class_financial_distress.csv"
subject = " [financial_distress]"
file_tag = "2_6_feature_selection_2_financial_distress"
file_output_TRAIN = "datasets/1_2_6_feature_selection_TRAIN_class_financial_distress.csv"
file_output_TEST = "datasets/1_2_6_feature_selection_TEST_class_financial_distress.csv"

train: DataFrame = read_csv(filename_TRAIN)
test: DataFrame = read_csv(filename_TEST)

target = "CLASS"

# Data leakage
train = train.drop(columns=["Financial Distress"])
test = test.drop(columns=["Financial Distress"])

# Lista de variáveis numéricas
num_vars = train.select_dtypes(include=['number']).columns
print("Número de colunas original: ", len(num_vars))

# Calcular o desvio padrão e ordenar de forma decrescente
print(train.select_dtypes(include=['number']).std().sort_values(ascending=False))

from math import ceil
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import HEIGHT, evaluate_approach, plot_multiline_chart

"""
def study_redundancy_for_feature_selection(
    train: DataFrame,
    test: DataFrame,
    target: str,
    min_threshold: float = 0.90,
    lag: float = 0.05,
    metric: str = "accuracy",
    file_tag: str = "",
) -> dict:
    options: list[float] = [
        round(min_threshold + i * lag, 3)
        for i in range(ceil((1 - min_threshold) / lag) + 1)
    ]

    df: DataFrame = train.drop(target, axis=1, inplace=False)
    corr_matrix: DataFrame = abs(df.corr())
    variables: Index[str] = corr_matrix.columns
    results: dict[str, list] = {"NB": [], "KNN": []}
    for thresh in options:
        vars2drop: list = []
        for v1 in variables:
            vars_corr: Series = (corr_matrix[v1]).loc[corr_matrix[v1] >= thresh]
            vars_corr.drop(v1, inplace=True)
            if len(vars_corr) > 1:
                lst_corr = list(vars_corr.index)
                for v2 in lst_corr:
                    if v2 not in vars2drop:
                        vars2drop.append(v2)

        print(f"Threshold: {thresh}, Variables dropped: {list(vars2drop)}")

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        eval: dict | None = evaluate_approach(
            train_copy, test_copy, target=target, metric=metric
        )
        if eval is not None:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])

    plot_multiline_chart(
        options,
        results,
        title=f"Redundancy study ({metric})" + subject,
        xlabel="correlation threshold",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"images/PreparationFeatureSelection/{file_tag}_study_redundant_variables_{metric}.png", bbox_inches="tight")
    return results


eval_metric = "recall"

figure(figsize=(2 * HEIGHT, HEIGHT))
study_redundancy_for_feature_selection(
    train,
    test,
    target=target,
    min_threshold=0.25,
    lag=0.05,
    metric=eval_metric,
    file_tag=file_tag,
)
"""


from dslabs_functions import (
    select_low_variance_variables,
    study_variance_for_feature_selection,
    apply_feature_selection,
    select_redundant_variables,
    study_redundancy_for_feature_selection, evaluate_approach,
)
vars2drop: list[str] = select_redundant_variables(
    train, min_threshold=0.85, target=target
)
train, test = apply_feature_selection(
    train, test, vars2drop, filename=f"{file_tag}", tag="redundant"
)

# StandardScaler, MinMaxScaler
print(f"Original data: \ntrain={train.shape}, test={test.shape}")
print()
print(f"After redundant FS min_threshold=0.85: \ntrain_cp={train.shape}, test_cp={test.shape}")




from math import ceil
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import HEIGHT, evaluate_approach, plot_multiline_chart, select_low_variance_variables
"""

def study_variance_for_feature_selection(
    train: DataFrame,
    test: DataFrame,
    target: str = "class",
    max_threshold: float = 1,
    lag: float = 0.05,
    metric: str = "accuracy",
    file_tag: str = "",
) -> dict:
    options: list[float] = [
        round(i * lag, 3) for i in range(1, ceil(max_threshold / lag + lag))
    ]
    results: dict[str, list] = {"NB": [], "KNN": []}
    summary5: DataFrame = train.describe()
    for thresh in options:
        print(thresh)
        vars2drop: Index[str] = summary5.columns[
            summary5.loc["std"] * summary5.loc["std"] < thresh
        ]
        vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        if train_copy.shape[1] == 0:
            print("Erro: O conjunto de treino não tem features restantes.")
            continue  # ou ajuste o código para tratar o caso sem features
        eval: dict[str, list] | None = evaluate_approach(
            train_copy, test_copy, target=target, metric=metric
        )
        if eval is not None:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])

    plot_multiline_chart(
        options,
        results,
        title=f"Variance study ({metric})" + subject,
        xlabel="variance threshold",
        ylabel=metric,
        percentage=True
    )
    savefig(f"images/PreparationFeatureSelection/{file_tag}_study_variance_{metric}.png", bbox_inches="tight")
    return results


eval_metric = "recall"

figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    train,
    test,
    target=target,
    max_threshold=0.12,
    lag=0.005,
    metric=eval_metric,
    file_tag=file_tag
)
"""

def select_low_variance_variables(
    data: DataFrame, max_threshold: float, target: str = "class"
) -> list:
    summary5: DataFrame = data.describe()
    vars2drop: Index[str] = summary5.columns[
        summary5.loc["std"] * summary5.loc["std"] < max_threshold
    ]
    vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop
    return list(vars2drop.values)

vars2drop: list[str] = select_low_variance_variables(train, 0.075, target=target)
print("Variables to drop", len(vars2drop), vars2drop)

def apply_feature_selection(
    data: DataFrame,
    vars2drop: list
):
    data_copy: DataFrame = data.drop(vars2drop, axis=1, inplace=False)
    return data_copy

train = apply_feature_selection(
    train, vars2drop
)
test = apply_feature_selection(
    test, vars2drop
)
print(f"Original data: data={train.shape}")
print(f"After low variance FS: data={train.shape}")


# Salvar os DataFrames resultantes, caso seja necessário
train.to_csv(file_output_TRAIN, index=False)
test.to_csv(file_output_TEST, index=False)