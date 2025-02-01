import numpy as np
from pandas import DataFrame, read_csv
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

# Carregar os dados
filename_TRAIN = "data/1_2_6_feature_selection_TRAIN_class_financial_distress.csv"
filename_TEST = "data/1_2_6_feature_selection_TEST_class_financial_distress.csv"
subject = " [financial distress]"
file_tag = "3_3_models_evaluation_1_financial_distress"

target = "CLASS"
train: DataFrame = read_csv(filename_TRAIN)
test: DataFrame = read_csv(filename_TEST)

labels: list = list(train[target].unique())
labels.sort()
vars = [col for col in train.columns if col != target]

# Separar features (X) e target (y)
trnX = train.drop(columns=[target])
trnY = train[target]
tstX = test.drop(columns=[target])
tstY = test[target]

# Configurar StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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
    kf,
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
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                scores = []  # Store fold scores
                
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                )
                
                # Stratified KFold cross-validation
                for train_idx, val_idx in kf.split(trnX, trnY):
                    trnX_fold, valX_fold = trnX.iloc[train_idx], trnX.iloc[val_idx]
                    trnY_fold, valY_fold = trnY.iloc[train_idx], trnY.iloc[val_idx]

                    # Train the model on the current fold
                    clf.fit(trnX_fold, trnY_fold)
                    prdY_fold: array = clf.predict(valX_fold)
                
                    # Evaluate the model on the fold
                    fold_eval: float = CLASS_EVAL_METRICS[metric](valY_fold, prdY_fold)
                    scores.append(fold_eval)

                # Evaluate the model on the final test set
                prdY: array = clf.predict(tstX)
                final_eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                avg_score = np.mean(scores)  # Average score across folds

                # Store the result for plotting            
                y_tst_values.append(final_eval)

                warm_start = True
                if final_eval - best_performance > DELTA_IMPROVE:
                    best_performance = final_eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params


eval_metric = "accuracy"

figure()
best_model, params = mlp_study(
    kf,
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=NR_MAX_ITER,
    lag=LAG,
    metric=eval_metric,
)
savefig(f"images/ModelsEvaluation-RedesNeuronais/{file_tag}_mlp_{eval_metric}_study.png", bbox_inches="tight")


# Best Model Performance
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/ModelsEvaluation-RedesNeuronais/{file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")


# Overfitting Study
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
savefig(f"images/ModelsEvaluation-RedesNeuronais/{file_tag}_mlp_{eval_metric}_overfitting.png", bbox_inches="tight")

# Loss Curve
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
savefig(f"images/ModelsEvaluation-RedesNeuronais/{file_tag}_mlp_{eval_metric}_loss_curve.png", bbox_inches="tight")
