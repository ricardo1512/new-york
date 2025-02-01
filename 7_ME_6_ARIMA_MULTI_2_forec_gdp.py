from pandas import read_csv, DataFrame, Series
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig, subplots
from statsmodels.tsa.arima.model import ARIMA
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart, plot_forecasting_eval, \
    plot_forecasting_series, HEIGHT

# Configurações iniciais
filename = "datasets/2_3_differentiation_forecast_gdp_europe.csv"
subject = " [GDP Europe]"
data: DataFrame = read_csv(filename,
    index_col="Year")
data.index = pd.to_datetime(data.index)
target = "GDP"

exog_columns = [col for col in data.columns if col != target]

series: Series = data[target]
exog_data: DataFrame = data[exog_columns]
timecol: str = "Year"
measure: str = "MAPE"

# Divisão em treino e teste
def series_train_test_split(series: Series, exog: DataFrame, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train, test = series[:trn_size], series[trn_size:]
    exog_train, exog_test = exog[:trn_size], exog[trn_size:]
    return train, test, exog_train, exog_test

train, test, exog_train, exog_test = series_train_test_split(series, exog_data, trn_pct=0.90)

# Ajustando o modelo ARIMA com exógenas
predictor = ARIMA(train, order=(3, 1, 2), exog=exog_train)
model = predictor.fit()
# print(model.summary())
# Diagnóstico do modelo
model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
savefig(f"images/B/ME_ARIMA/2_ARIMA_plot_diagnostics_MULTI.png", bbox_inches="tight")

# Estudo de ARIMA com variáveis exógenas
def arima_study(train: Series, test: Series, exog_train: DataFrame, exog_test: DataFrame, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "MAPE" # measure == "R2" or
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * 6, 6))
    plt.suptitle(f"ARIMA ({measure})" + subject)

    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                print("i:", i, "p=", p, "d=", d, "q=", q)
                try:
                    arima = ARIMA(train, exog=exog_train, order=(p, d, q))
                    model = arima.fit()
                    prd_tst = model.forecast(steps=len(test), exog=exog_test)
                    eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                    if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                        best_performance: float = eval
                        best_params["params"] = (p, d, q)
                        best_model = model
                    yvalues.append(eval)
                except Exception as e:
                    print(f"Error with ARIMA({p},{d},{q}): {e}")
                    yvalues.append(None)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )

    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params

# Executando o estudo
best_model, best_params = arima_study(train, test, exog_train, exog_test, measure=measure)
savefig(f"images/B/ME_ARIMA/2_ARIMA_{measure}_study_MULTI.png", bbox_inches="tight")

# Avaliação e visualização
params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1, exog=exog_train)
prd_tst = best_model.forecast(steps=len(test), exog=exog_test)

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"ARIMA (p={params[0]}, d={params[1]}, q={params[2]})" + subject,
)
savefig(f"images/B/ME_ARIMA/2_ARIMA_{measure}_eval_MULTI.png")

plot_forecasting_series(
    train, test, prd_tst, title=f"ARIMA " + subject, xlabel=timecol, ylabel=target
)
savefig(f"images/B/ME_ARIMA/2_ARIMA_{measure}_forecast_MULTI.png")
