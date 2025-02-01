import pandas as pd
from matplotlib.pyplot import savefig
from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart

filename = "datasets/2_3_differentiation_forecast_gdp_europe.csv"
subject = " [GDP Europe]"
data: DataFrame = read_csv(filename,
    index_col="Year")
data.index = pd.to_datetime(data.index)
target = "GDP"
series: Series = data[target]
timecol: str = "Year"
measure: str = "R2"

def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)


def exponential_smoothing_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for alpha in alpha_values:
        tool = SimpleExpSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (alpha,)
            best_model = model
        yvalues.append(eval)

    print(f"Exponential Smoothing best with alpha={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing ({measure})" + subject,
        xlabel="alpha",
        ylabel=measure,
        percentage=flag,
    )

    return best_model, best_params


best_model, best_params = exponential_smoothing_study(train, test, measure=measure)

savefig(f"images/B/ME_ExponentialSmoothing/2_exponential_smoothing_{measure}_study.png", bbox_inches="tight")

from dslabs_functions import plot_forecasting_eval

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Exponential Smoothing alpha={params[0]}" + subject)
savefig(f"images/B/ME_ExponentialSmoothing/2_exponential_smoothing_{measure}_eval.png", bbox_inches="tight")

from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Exponential Smoothing ({measure}) Forecasting" + subject,
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/B/ME_ExponentialSmoothing/2_exponential_smoothing_{measure}_forecast.png", bbox_inches="tight")