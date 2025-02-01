import pandas as pd
from matplotlib.pyplot import savefig
from pandas import read_csv, DataFrame, Series
from dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

filename = "datasets/2_4_scaling_forecast_ny_arrests.csv"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename,
    index_col="Date")
data.index = pd.to_datetime(data.index)
target = "Manhattan"

from sklearn.base import RegressorMixin


class SimpleAvgRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean: float = 0.0
        return

    def fit(self, X: Series):
        self.mean = X.mean()
        return

    def predict(self, X: Series) -> Series:
        prd: list = len(X) * [self.mean]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series

series: Series = data[target]

def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{target} - Simple Average"+subject)

savefig(f"images/B/ME_SimpleAverage/1_simpleAvg_eval.png", bbox_inches="tight")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{target} - Simple Average" + subject,
    xlabel="Date",
    ylabel=target,
)
savefig(f"images/B/ME_SimpleAverage/1_simpleAvg_forecast.png", bbox_inches="tight")
