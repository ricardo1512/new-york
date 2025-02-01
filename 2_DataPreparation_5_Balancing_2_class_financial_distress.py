import numpy as np
from pandas import read_csv, concat, DataFrame, Series
from matplotlib.pyplot import figure, subplots, show, savefig
from sklearn.model_selection import train_test_split

from dslabs_functions import plot_bar_chart, run_NB, run_KNN, CLASS_EVAL_METRICS, plot_multibar_chart

# Carregar os dados
filename = "datasets/1_2_4_scaling_class_financial_distress.csv"
subject = " [financial distress]"
file_tag = "2_4_balancing_2_financial_distress"
file_output_TRAIN = "datasets/1_2_5_balancing_TRAIN_class_financial_distress.csv"
file_output_TEST = "datasets/1_2_5_balancing_TEST_class_financial_distress.csv"

target = "CLASS"
original: DataFrame = read_csv(filename)

train_original, test_original = train_test_split(original, test_size=0.3, random_state=42)

original = train_original.copy()

target_count: Series = original[target].value_counts()

positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)

df_positives: Series = original[original[target] == positive_class]
df_negatives: Series = original[original[target] == negative_class]


# OVERSAMPLING

df_pos_sample: DataFrame = DataFrame(
    df_positives.sample(len(df_negatives), replace=True)
)
df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)

print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(df_negatives))
print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")

"""

# UNDERSAMPLING
df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")

"""

"""
from numpy import ndarray
from pandas import Series
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = original.pop(target).values
X: ndarray = original.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(original.columns) + [target]


smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)


train, test = train_test_split(df_smote, test_size=0.3, random_state=42)

def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: np.ndarray = train.values
    tstY = test.pop(target).values
    tstX: np.ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval

target = "CLASS"

imput = "oversampling"


figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Scaling eval. with " + imput + subject, percentage=True
)

# Salvar a figura
savefig(f"images/PreparationBalancing/{file_tag}_{imput}.png", bbox_inches="tight")
"""

df_over.to_csv(file_output_TRAIN, index=False)
test_original.to_csv(file_output_TEST, index=False)