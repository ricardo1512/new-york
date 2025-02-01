import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from numpy import arange
from pandas import read_csv, DataFrame, Series
from sklearn.linear_model import LinearRegression

from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart, series_train_test_split, \
    plot_forecasting_eval, plot_forecasting_series

filename = "datasets/2_4_scaling_forecast_gdp_europe.csv"
subject = " [GDP Europe]"
data: DataFrame = read_csv(filename,
    index_col="Year")
data.index = pd.to_datetime(data.index)
target = "GDP"
exog_vars = [col for col in data.columns if col != target]

# Incluindo exógenas no modelo
series = data[[target] + exog_vars].values.astype("float32")
timecol: str = "Year"
measure: str = "MAPE"

def series_train_test_split(series: Series, trn_pct: float):
    trn_size = int(len(series) * trn_pct)
    train = series[:trn_size]
    test = series[trn_size:]
    return train, test

train, test = series_train_test_split(series, trn_pct=0.90)

from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

def prepare_dataset_for_lstm(series, seq_length: int = 4):
    setX: list = []
    setY: list = []
    for i in range(len(series) - seq_length):
        past = series[i : i + seq_length, :]  # Usando indexação numpy
        future = series[i + 1 : i + seq_length + 1, 0:1]  # Apenas o alvo no futuro
        setX.append(past)
        setY.append(future)
    return tensor(setX), tensor(setY)

class DS_LSTM(Module):
    def __init__(self, train, input_size: int, hidden_size: int = 50, num_layers: int = 1, length: int = 4):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
        self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=len(train) // 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def fit(self):
        self.train()
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def predict(self, X):
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]

input_size = len(exog_vars) + 1  # Inclui o target e variáveis exógenas
model = DS_LSTM(train, input_size=input_size, hidden_size=50, num_layers=1)
loss = model.fit()
print(loss)

from dslabs_functions import HEIGHT, plot_multiline_chart
from copy import deepcopy
from matplotlib.pyplot import figure, savefig, subplots

def lstm_study(train, test, nr_episodes: int = 1000, measure: str = "R2"):
    sequence_size = [2, 3, 4]
    nr_hidden_units = [25, 50, 100]

    step: int = nr_episodes // 10
    episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
    flag = measure == "MAPE" # measure == "R2" or
    best_model = None
    best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
    best_performance: float = -100000

    _, axs = subplots(1, len(sequence_size), figsize=(len(sequence_size) * HEIGHT, HEIGHT))
    plt.suptitle("LSTM with all variables" + subject, fontsize=14)

    for i in range(len(sequence_size)):
        length = sequence_size[i]
        tstX, tstY = prepare_dataset_for_lstm(test, seq_length=length)

        values = {}
        for hidden in nr_hidden_units:
            yvalues = []
            model = DS_LSTM(train, input_size=input_size, hidden_size=hidden)
            for n in range(0, nr_episodes + 1):
                model.fit()
                if n % step == 0:
                    prd_tst = model.predict(tstX)
                    eval: float = FORECAST_MEASURES[measure](test[length:, 0], prd_tst)
                    print(f"seq length={length} hidden_units={hidden} nr_episodes={n}", eval)
                    if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                        best_performance: float = eval
                        best_params["params"] = (length, hidden, n)
                        best_model = deepcopy(model)
                    yvalues.append(eval)
            values[hidden] = yvalues
        plot_multiline_chart(
            episodes,
            values,
            ax=axs[i],
            title=f"LSTM with all variables, seq length={length} ({measure})",
            xlabel="nr episodes",
            ylabel=measure,
            percentage=flag,
        )
    print(
        f"LSTM best results achieved with length={best_params['params'][0]} hidden_units={best_params['params'][1]} and nr_episodes={best_params['params'][2]}) ==> measure={best_performance:.2f}"
    )
    return best_model, best_params

best_model, best_params = lstm_study(train, test, nr_episodes=3000, measure=measure)

savefig(f"images/B/ME_LSTMs/2_LSTMs_{measure}_study_MULTI.png")

params = best_params["params"]
best_length = params[0]

# As variáveis 'train' e 'test' devem ser mantidas como DataFrames ou Series
train, test = series_train_test_split(series, trn_pct=0.90)

# Preparando os dados para o LSTM
trnX, trnY = prepare_dataset_for_lstm(train, seq_length=best_length)
tstX, tstY = prepare_dataset_for_lstm(test, seq_length=best_length)

# Fazendo as previsões
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

# Plotando a avaliação das previsões
plot_forecasting_eval(
    train[best_length:, 0],  # Usando DataFrame para acessar com iloc
    test[best_length:, 0],
    prd_trn,
    prd_tst,
    title=f"LSTM with all variables (length={best_length}, hidden={params[1]}, epochs={params[2]})" + subject,
)
savefig(f"images/B/ME_LSTMs/2_LSTMs_{measure}_eval_MULTI.png")

# Preparando a série para o gráfico de previsão
train_size = int(len(series) * 0.90)
series = data[[target] + exog_vars]  # Incluindo as variáveis exógenas
train, test = series[[target]].iloc[:train_size], series[[target]].iloc[train_size:]

# Criando a série de previsões
pred_series: Series = Series(prd_tst.numpy().ravel(), index=test.index[best_length:])

# Plotando a previsão final
plot_forecasting_series(
    train[best_length:],  # Somente a coluna target
    test[best_length:],   # Somente a coluna target
    pred_series,
    title=f"LSTMs with all variables" + subject,
    xlabel=timecol,
    ylabel=target,
)

savefig(f"images/B/ME_LSTMs/2_LSTMs_{measure}_forecast_MULTI.png")