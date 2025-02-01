from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar os dados
filename_TRAIN = "datasets/1_1_5_balancing_TRAIN_class_ny_arrests.csv"
filename_TEST = "datasets/1_1_5_balancing_TEST_class_ny_arrests.csv"
subject = " [ny arrests]"
file_tag = "3_2_models_evaluation_1_ny_arrests"

target = "CLASS"
TRAIN: DataFrame = read_csv(filename_TRAIN)
TEST: DataFrame = read_csv(filename_TEST)

frac = 0.02
TRAIN = TRAIN.sample(frac=frac, random_state=42)
TEST = TEST.sample(frac=frac, random_state=42)

labels: list = list(TRAIN[target].unique())
labels.sort()

# Separar features (X) e target (y)
trnX = TRAIN.drop(columns=[target])
trnY = TRAIN[target]
tstX = TEST.drop(columns=[target])
tstY = TEST[target]


from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from dslabs_functions import read_train_test_from_files, plot_evaluation_results

def knn_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='accuracy'
        ) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']
    dist: list[Literal['manhattan']] = ['manhattan']

    kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params['params'] = (k, d)
                best_model = clf
            print(f'KNN {d} k={k}')
        values[d] = y_tst_values
    print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

    return best_model, best_params

eval_metric = 'accuracy'

figure()
best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=9, metric=eval_metric)
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
# savefig(f'images/ModelsEvaluation-KNN/{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")




figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/ModelsEvaluation-KNN/{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")


"""
from matplotlib.pyplot import figure, savefig
# distance = "manhattan"
distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
K_MAX = 25
kvalues: list[int] = [i for i in range(1, K_MAX + 1, 2)]
y_tst_values: list = []
y_trn_values: list = []
acc_metric: str = "accuracy"
for k in kvalues:
    print(k)
    clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    kvalues,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"KNN overfitting study for {distance}" + subject,
    xlabel="K",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/ModelsEvaluation-KNN/{file_tag}_knn_overfitting.png", bbox_inches="tight")
"""