import pandas as pd
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_forecasting_eval, plot_forecasting_series

filename = "datasets/2_4_scaling_forecast_ny_arrests.csv"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename,
    index_col="Date")
data.index = pd.to_datetime(data.index)
target = "Manhattan"
series: Series = data[target]

from numpy import mean
from pandas import Series
from sklearn.base import RegressorMixin


class RollingMeanRegressor(RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win
        self.memory: list = []

    def fit(self, X: Series):
        self.memory = X.iloc[-self.win_size :]
        # print(self.memory)
        return

    def predict(self, X: Series):
        estimations = self.memory.tolist()
        for i in range(len(X)):
            new_value = mean(estimations[len(estimations) - self.win_size - i :])
            estimations.append(new_value)

        prd_series: Series = Series(estimations[self.win_size :])
        prd_series.index = X.index
        return prd_series

from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart


def rolling_mean_study(train: Series, test: Series, measure: str = "R2"):
    # win_size = (3, 5, 10, 15, 20, 25, 30, 40, 50)
    win_size = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for w in win_size:
        pred = RollingMeanRegressor(win=w)
        pred.fit(train)
        prd_tst = pred.predict(test)

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (w,)
            best_model = pred
        yvalues.append(eval)

    print(f"Rolling Mean best with win={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        win_size, yvalues, title=f"Rolling Mean ({measure})" + subject, xlabel="window size", ylabel=measure, percentage=flag
    )

    return best_model, best_params

from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig
from dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series, HEIGHT

timecol: str = "Date"
measure: str = "R2"


def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)

fig = figure(figsize=(HEIGHT, HEIGHT))
best_model, best_params = rolling_mean_study(train, test)

savefig(f"images/B/ME_RollingMean/1_rollingmean_{measure}_study.png", bbox_inches="tight")

params = best_params["params"]
prd_trn: Series = best_model.predict(train)
prd_tst: Series = best_model.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Rolling Mean ({measure}, win={params[0]})" + subject)

savefig(f"images/B/ME_RollingMean/1_rollingmean_{measure}_win{params[0]}_eval.png", bbox_inches="tight")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Rolling Mean (win={params[0]})" + subject,
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/B/ME_RollingMean/1_rollingmean_{measure}_forecast.png", bbox_inches="tight")