import pandas as pd
from matplotlib.pyplot import savefig
from numpy import arange
from pandas import read_csv, DataFrame, Series
from sklearn.linear_model import LinearRegression

from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart, series_train_test_split, \
    plot_forecasting_eval

filename = "datasets/2_4_scaling_forecast_ny_arrests.csv"
subject = " [ny arrests]"
data: DataFrame = read_csv(filename,
    index_col="Date")
data.index = pd.to_datetime(data.index)
target = "Manhattan"
series: Series = data[target]
timecol: str = "Date"
measure: str = "R2"

def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(data)).reshape(-1, 1)
tstY = test.to_numpy()

model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Linear Regression" + subject)

savefig(f"images/B/ME_LinearRegression/1_linear_regression_eval.png", bbox_inches="tight")

from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Linear Regression" + subject,
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/B/ME_LinearRegression/1_linear_regression_forecast.png", bbox_inches="tight")