from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar os dados
filename_TRAIN = "datasets/1_1_5_balancing_TRAIN_class_ny_arrests.csv"
filename_TEST = "datasets/1_1_5_balancing_TEST_class_ny_arrests.csv"
subject = " [ny arrests]"
file_tag = "3_1_models_evaluation_1_ny_arrests"

target = "CLASS"
TRAIN: DataFrame = read_csv(filename_TRAIN)
TEST: DataFrame = read_csv(filename_TEST)

frac = 1
TRAIN = TRAIN.sample(frac=frac, random_state=42)
TEST = TEST.sample(frac=frac, random_state=42)

labels: list = list(TRAIN[target].unique())
labels.sort()

# Separar features (X) e target (y)
trnX = TRAIN.drop(columns=[target])
trnY = TRAIN[target]
tstX = TEST.drop(columns=[target])
tstY = TEST[target]

from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart

def naive_Bayes_study(
    trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)
        # print(f'NB {clf}')
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})" + subject,
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params

eval_metric = "recall"
figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
# savefig(f"images/ModelsEvaluation-NaiveBayes/{file_tag}_nb_{eval_metric}_study.png", bbox_inches="tight")


from dslabs_functions import plot_evaluation_results

prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/ModelsEvaluation-NaiveBayes/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")
