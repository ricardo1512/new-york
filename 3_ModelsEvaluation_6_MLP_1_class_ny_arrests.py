from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar os dados
filename_TRAIN = "datasets/1_1_5_balancing_TRAIN_class_ny_arrests.csv"
filename_TEST = "datasets/1_1_5_balancing_TEST_class_ny_arrests.csv"
subject = " [ny arrests]"
file_tag = "3_5_models_evaluation_1_ny_arrests"

target = "CLASS"
TRAIN: DataFrame = read_csv(filename_TRAIN)
TEST: DataFrame = read_csv(filename_TEST)

frac = 0.005
TRAIN = TRAIN.sample(frac=frac, random_state=42)
TEST = TEST.sample(frac=frac, random_state=42)

vars = TRAIN.drop(columns=target).columns.tolist()
labels: list = list(TRAIN[target].unique())
labels.sort()

# Separar features (X) e target (y)
trnX = TRAIN.drop(columns=[target])
trnY = TRAIN[target]
tstX = TEST.drop(columns=[target])
tstY = TEST[target]


from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.neural_network import MLPClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart

LAG: int = 500
NR_MAX_ITER: int = 5000

def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
         "constant",
         "invscaling",
         "adaptive",
     ]  # only used if optimizer='sgd'
    #lr_types: list[Literal["constant"]] = [
    # "constant"
    #]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]# [0.05]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    for i in range(len(lr_types)):
        print(f"lr_type={lr_types[i]}")
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            print(f"\tlr={lr}")
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                print(f"\t\tn={nr_iterations[j]}")
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}" + subject,
            xlabel="nr iterations",
            ylabel=metric,
            percentage=False,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params


eval_metric = "accuracy"

figure()
best_model, params = mlp_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=NR_MAX_ITER,
    lag=LAG,
    metric=eval_metric,
)
savefig(f"images/ModelsEvaluation-MLP/{file_tag}_mlp_{eval_metric}_study.png", bbox_inches="tight")

prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/ModelsEvaluation-MLP/{file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")



lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
lr: float = params["params"][1]

nr_iterations: list[int] = [LAG] + [i for i in range(2 * LAG, NR_MAX_ITER + 1, LAG)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "accuracy"

warm_start: bool = False
for n in nr_iterations:
    clf = MLPClassifier(
        warm_start=warm_start,
        learning_rate=lr_type,
        learning_rate_init=lr,
        max_iter=n,
        activation="logistic",
        solver="sgd",
        verbose=False,
    )
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
    warm_start = True

figure()
plot_multiline_chart(
    nr_iterations,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
    xlabel="nr_iterations",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/ModelsEvaluation-MLP/{file_tag}_mlp_{eval_metric}_overfitting.png", bbox_inches="tight")


from numpy import arange
from dslabs_functions import plot_line_chart


figure()
plot_line_chart(
    arange(len(best_model.loss_curve_)),
    best_model.loss_curve_,
    title="Loss curve for MLP best model training",
    xlabel="iterations",
    ylabel="loss",
    percentage=False,
)
savefig(f"images/ModelsEvaluation-MLP/{file_tag}_mlp_{eval_metric}_loss_curve.png", bbox_inches="tight")
