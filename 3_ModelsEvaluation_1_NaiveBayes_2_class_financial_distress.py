import pandas as pd
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig
from pandas import DataFrame, read_csv
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score
from dslabs_functions import plot_bar_chart, plot_evaluation_results, CLASS_EVAL_METRICS, DELTA_IMPROVE

# Carregar os dados
filename_TRAIN = "datasets/1_2_6_feature_selection_TRAIN_class_financial_distress.csv"
filename_TEST = "datasets/1_2_6_feature_selection_TEST_class_financial_distress.csv"
subject = " [financial_distress]"
file_tag = "3_1_models_evaluation_2_financial_distress"

target = "CLASS"
TRAIN: DataFrame = read_csv(filename_TRAIN)
TEST: DataFrame = read_csv(filename_TEST)

labels: list = list(TRAIN[target].unique())
labels.sort()

# Separar features (X) e target (y)
trnX = TRAIN.drop(columns=[target])
trnY = TRAIN[target]
tstX = TEST.drop(columns=[target])
tstY = TEST[target]

# Configurar StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np

def naive_Bayes_study(
    kf, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
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

    for clf_name, clf in estimators.items():
        scores = []
        for train_idx, val_idx in kf.split(trnX, trnY):
            trnX_fold, valX_fold = trnX.iloc[train_idx], trnX.iloc[val_idx]
            trnY_fold, valY_fold = trnY.iloc[train_idx], trnY.iloc[val_idx]

            # Treinar o modelo na dobra atual
            clf.fit(trnX_fold, trnY_fold)
            prdY_fold = clf.predict(valX_fold)

            # Avaliar o modelo na dobra
            fold_eval: float = CLASS_EVAL_METRICS[metric](valY_fold, prdY_fold)
            scores.append(fold_eval)

        # Avaliar no conjunto de teste final (tstX, tstY)
        prdY: array = clf.predict(tstX)
        final_eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        avg_score = np.mean(scores)  # Média das métricas nas dobras

        xvalues.append(clf_name)
        yvalues.append(final_eval)

        if final_eval - best_performance > DELTA_IMPROVE:
            best_performance = final_eval
            best_params["name"] = clf_name
            best_params[metric] = final_eval
            best_model = clf

        print(f'NB {clf_name} | Cross-Validation Mean: {avg_score:.4f} | Test Evaluation: {final_eval:.4f}')

    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})" + subject,
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


# Avaliar modelos
eval_metric = "recall"

best_model, params = naive_Bayes_study(kf, trnX, trnY, tstX, tstY, eval_metric)
# savefig(f"images/ModelsEvaluation-NaiveBayes/{file_tag}_nb_{eval_metric}_study.png", bbox_inches="tight")

# Plotar resultados finais
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/ModelsEvaluation-NaiveBayes/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches="tight")

