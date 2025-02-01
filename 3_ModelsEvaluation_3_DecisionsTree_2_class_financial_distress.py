import numpy as np
from pandas import DataFrame, read_csv
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

# Carregar os dados
filename_TRAIN = "datasets/1_2_6_feature_selection_TRAIN_class_financial_distress.csv"
filename_TEST = "datasets/1_2_6_feature_selection_TEST_class_financial_distress.csv"
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
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files
from dslabs_functions import plot_evaluation_results, plot_multiline_chart

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Literal
import pandas as pd


def trees_study(
        kf, trnX: pd.DataFrame, trnY: pd.Series, tstX: pd.DataFrame, tstY: pd.Series, d_max: int = 10, lag: int = 2,
        metric: str = 'accuracy'
) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max + 1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}

    for c in criteria:
        print('Criterion: ', c)
        y_tst_values: list[float] = []
        for d in depths:
            print('Max Depth: ', d)
            scores = []  # Store fold scores

            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)

            # Stratified KFold cross-validation
            for train_idx, val_idx in kf.split(trnX, trnY):
                trnX_fold, valX_fold = trnX.iloc[train_idx], trnX.iloc[val_idx]
                trnY_fold, valY_fold = trnY.iloc[train_idx], trnY.iloc[val_idx]

                # Train the model on the current fold
                clf.fit(trnX_fold, trnY_fold)
                prdY_fold = clf.predict(valX_fold)

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
                best_params['params'] = (c, d)
                best_model = clf

            print(f'DT {c} max_depth={d} | Cross-Validation Mean: {avg_score:.4f} | Test Evaluation: {final_eval:.4f}')

        values[c] = y_tst_values

    print(f'Best DT model with criterion={best_params["params"][0]} and max_depth={best_params["params"][1]}')
    plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='Max Depth (d)', ylabel=metric,
                         percentage=True)

    return best_model, best_params


eval_metric = 'accuracy'
best_model, params = trees_study(kf, trnX, trnY, tstX, tstY, d_max=25, metric=eval_metric)
savefig(f'images/ModelsEvaluation-DecisionTrees/{file_tag}_dt_{eval_metric}_study.png', bbox_inches="tight")

prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
# figure()
# plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
# savefig(f'images/ModelsEvaluation-DecisionTrees/{file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval_HM.png', bbox_inches="tight")

"""
crit: Literal["entropy", "gini"] = params["params"][0]
d_max = 25
depths: list[int] = [i for i in range(2, d_max + 1, 1)]
y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "accuracy"
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    depths,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"DT overfitting study for {crit}",
    xlabel="max_depth",
    ylabel=str(eval_metric),
    percentage=True,
)
# savefig(f"images/ModelsEvaluation-DecisionTrees/{file_tag}_dt_{eval_metric}_overfitting.png", bbox_inches="tight")

"""
"""
from sklearn.tree import export_graphviz
from matplotlib.pyplot import imread, imshow, axis
from subprocess import call

tree_filename: str = f"images/ModelsEvaluation-DecisionTrees/{file_tag}_dt_{eval_metric}_best_tree"
max_depth2show = 4
st_labels: list[str] = [str(value) for value in labels]

import os
from subprocess import call

# Caminho do executável do Graphviz
# os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
vars = list(trnX.columns)

dot_data: str = export_graphviz(
    best_model,
    out_file=tree_filename + ".dot",
    max_depth=max_depth2show,
    feature_names=vars,
    class_names=st_labels,
    filled=True,
    rounded=True,
    impurity=False,
    special_characters=True,
    precision=2,
)
# Convert to png
call(
    [r"C:\Program Files\Graphviz\bin\dot.exe", "-Tpng", tree_filename + ".dot", "-o", tree_filename + ".png", "-Gdpi=600"]
)

figure(figsize=(14, 6))
imshow(imread(tree_filename + ".png"))
axis("off")
"""


from numpy import argsort
from dslabs_functions import plot_horizontal_bar_chart

importances = best_model.feature_importances_
indices: list[int] = argsort(importances)[::-1]
elems: list[str] = []
imp_values: list[float] = []
for f in range(len(vars)):
    elems += [vars[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

figure()
plot_horizontal_bar_chart(
    elems,
    imp_values,
    title="Decision Tree variables importance" + subject,
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
savefig(f"images/ModelsEvaluation-DecisionTrees/{file_tag}_dt_{eval_metric}_vars_ranking.png", bbox_inches="tight")

