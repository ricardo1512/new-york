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

from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import RandomForestClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


def random_forests_study(
    kf,
    trnX: pd.DataFrame,
    trnY: pd.Series,
    tstX: pd.DataFrame,
    tstY: pd.Series,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[RandomForestClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    best_model: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for f in max_features:
            y_tst_values: list[float] = []
            for n in n_estimators:
                scores = []  # Store fold scores
                clf = RandomForestClassifier(
                    n_estimators=n, max_depth=d, max_features=f
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
                
                # Update the best model if improvement is found
                if final_eval - best_performance > DELTA_IMPROVE:
                    best_performance = final_eval
                    best_params["params"] = (d, f, n)
                    best_model = clf
                print(f'RF d={d} f={f} n={n}')
            values[f] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Random Forests with max_depth={d}" + subject,
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
    )
    return best_model, best_params

eval_metric = "accuracy"

figure()
best_model, params = random_forests_study(
    kf,
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=1000,
    lag=250,
    metric=eval_metric,
)
savefig(f"images/ModelsEvaluation-RandomForest/{file_tag}_rf_{eval_metric}_study.png", bbox_inches="tight")
show()

# Best Accuracy
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/ModelsEvaluation-RandomForest/{file_tag}_rf_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")
show()

# Variable Importance
from numpy import std, argsort
from dslabs_functions import plot_horizontal_bar_chart

stdevs: list[float] = list(
    std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
)
importances = best_model.feature_importances_
indices: list[int] = argsort(importances)[::-1]
elems: list[str] = []
imp_values: list[float] = []
for f in range(len(vars)):
    elems += [vars[indices[f]]]
    imp_values.append(importances[indices[f]])
    print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

figure()
plot_horizontal_bar_chart(
    elems,
    imp_values,
    error=stdevs,
    title="RF variables importance" + subject,
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
savefig(f"images/ModelsEvaluation-RandomForest/{file_tag}_rf_{eval_metric}_vars_ranking.png", bbox_inches="tight")

# Overfitting Study
d_max: int = params["params"][0]
feat: float = params["params"][1]
nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric: str = "accuracy"

for n in nr_estimators:
    clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    nr_estimators,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"RF overfitting study for d={d_max} and f={feat}" + subject,
    xlabel="nr_estimators",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/ModelsEvaluation-RandomForest/{file_tag}_rf_{eval_metric}_overfitting.png", bbox_inches="tight")