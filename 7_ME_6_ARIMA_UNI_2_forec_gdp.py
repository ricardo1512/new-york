import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from numpy import arange
from pandas import read_csv, DataFrame, Series
from sklearn.linear_model import LinearRegression

from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart, series_train_test_split, \
    plot_forecasting_eval

filename = "datasets/2_3_differentiation_forecast_gdp_europe.csv"
subject = " [GDP Europe]"
data: DataFrame = read_csv(filename,
    index_col="Year")
data.index = pd.to_datetime(data.index)
target = "GDP"
series: Series = data[target]
timecol: str = "Year"
measure: str = "MAPE"

def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)

from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA
from dslabs_functions import series_train_test_split, HEIGHT

predictor = ARIMA(train, order=(3, 1, 2))
model = predictor.fit()
# print(model.summary())
model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
savefig(f"images/B/ME_ARIMA/2_ARIMA_plot_diagnostics_UNI.png", bbox_inches="tight")

from matplotlib.pyplot import figure, savefig, subplots
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart


def arima_study(train: Series, test: Series, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "MAPE" # measure == "R2" or
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    plt.suptitle(f"ARIMA ({measure})" + subject)

    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                # print(f"ARIMA ({p}, {d}, {q})", eval)
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )

    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params

from matplotlib.pyplot import savefig

best_model, best_params = arima_study(train, test, measure=measure)
savefig(f"images/B/ME_ARIMA/2_ARIMA_{measure}_study_UNI.png", bbox_inches="tight")

from dslabs_functions import plot_forecasting_eval

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"ARIMA (p={params[0]}, d={params[1]}, q={params[2]})" + subject,
)
savefig(f"images/B/ME_ARIMA/2_ARIMA_{measure}_eval_UNI.png")

from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"ARIMA " + subject,
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/B/ME_ARIMA/2_ARIMA_{measure}_forecast_UNI.png")