import pandas as pd
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig
from dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

filename = "datasets/2_2_smoothing_forecast_ny_arrests.csv"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename,
    index_col="Date")
data.index = pd.to_datetime(data.index)
target = "Manhattan"
series: Series = data[target]

from pandas import Series
from sklearn.base import RegressorMixin


class PersistenceOptimistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last: float = 0.0
        return

    def fit(self, X: Series):
        self.last = X.iloc[-1]
        # print(self.last)
        return

    def predict(self, X: Series):
        prd: list = X.shift().values.ravel()
        prd[0] = self.last
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series

def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)

fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{target} - Persistence Optimist" + subject)

savefig(f"images/B/ME_Persistence/1_persistence_optim_eval.png", bbox_inches="tight")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{target} - Persistence Optimist" + subject,
    xlabel="Date",
    ylabel=target,
)

savefig(f"images/B/ME_Persistence/1_persistence_optim_forecast.png", bbox_inches="tight")

class PersistenceRealistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0
        self.estimations = [0]
        self.obs_len = 0

    def fit(self, X: Series):
        for i in range(1, len(X)):
            self.estimations.append(X.iloc[i - 1])
        self.obs_len = len(self.estimations)
        self.last = X.iloc[len(X) - 1]
        prd_series: Series = Series(self.estimations)
        prd_series.index = X.index
        return prd_series

    def predict(self, X: Series):
        prd: list = len(X) * [self.last]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series


fr_mod = PersistenceRealistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{target} - Persistence Realist" + subject)
savefig(f"images/B/ME_Persistence/1_persistence_realist_eval.png", bbox_inches="tight")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{target} - Persistence Realist" + subject,
    xlabel="Date",
    ylabel=target,
)
savefig(f"images/B/ME_Persistence/1_persistence_realist_forecast.png", bbox_inches="tight")