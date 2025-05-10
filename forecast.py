# %%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# %%
simbol_actiuni = "BKNG"
data_inceput = "2020-01-01"

# %%
try:

    data_actiuni = yf.download(simbol_actiuni, start=data_inceput, auto_adjust=False)

    print("Coloanele din df:")
    print(data_actiuni.columns)

    print("\nPrimele randuri din df:")
    print(data_actiuni.head())

    # salveaz datele intr un fisier CSV
    data_actiuni.to_csv(f"{simbol_actiuni}_istoric.csv")

    # pretul de inchidere
    plt.figure(figsize=(12, 6))
    plt.plot(data_actiuni[('Close', 'BKNG')])
    plt.title(f'Evolutia pretului de inchidere pentru {simbol_actiuni}')
    plt.xlabel('Data')
    plt.ylabel('Pret')
    plt.grid(True)
    plt.show()

    # volumul tranzactionat
    plt.figure(figsize=(12, 6))
    plt.plot(data_actiuni[('Volume', 'BKNG')])
    plt.title(f'Volumul tranzactionat pentru {simbol_actiuni}')
    plt.xlabel('Data')
    plt.ylabel('Volum')
    plt.grid(True)
    plt.show()

    # elimin nivelul "Ticker"
    new_data = data_actiuni.copy()
    new_data.columns = [col[0] for col in new_data.columns]

    print("\ndf cu coloane simplificate:")
    print(new_data.head())

except Exception as e:
    print(f"A aparut o eroare: {e}")


# %%
import matplotlib.pyplot as plt

# pret de inchidere ajustat
plt.figure(figsize=(12, 6))
plt.plot(data_actiuni['Adj Close'])
plt.title(f'Evolutia pretului de inchidere ajustat pentru {simbol_actiuni}')
plt.xlabel('Data')
plt.ylabel('Pret')
plt.grid(True)
plt.show()

# %%
# media mobila simpla de 50 de zile
new_data['MA_50'] = new_data['Adj Close'].rolling(window=50).mean()

# media mobila simpla de 200 de zile
new_data['MA_200'] = new_data['Adj Close'].rolling(window=200).mean()

# deviatia standard a randamentelor zilnice
new_data['Daily_Return'] = new_data['Adj Close'].pct_change() # randamentul zilnic
new_data['Volatility'] = new_data['Daily_Return'].rolling(window=30).std() # fluctuatiile pe 30 de zile

print(new_data.head())
print(new_data.tail())

clean_data = new_data.dropna()

# obtin pretul din ziua urmatoare pe randul curent
clean_data['Target'] = clean_data['Adj Close'].shift(-1)

final_data = clean_data.dropna()

print("\ndf cu coloana Target:")
print(final_data.head())
print("\nUltimele randuri:")
print(final_data.tail())

from sklearn.model_selection import train_test_split

# %%
features = ['Adj Close', 'Volume', 'MA_50', 'MA_200', 'Volatility', 'Open', 'High', 'Low', 'Close']
target = 'Target'

X = final_data[features]
y = final_data[target]

# 80% antrenare, 20% testare

train_size = int(len(final_data) * 0.8)

# split pentru antrenare si testare
X_train, X_test = X[0:train_size], X[train_size:len(final_data)]
y_train, y_test = y[0:train_size], y[train_size:len(final_data)]

print(f"\nDimensiunea setului de antrenament (X_train): {X_train.shape}")
print(f"\nDimensiunea setului de testare (X_test): {X_test.shape}")

# %%
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler 

# LSTM urile sunt sensibile la scara datelor
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

time_steps = 60 # modelul se va uita la ultimele 60 de zile pentru a prezice ziua urmatoare

# secvente de date pentru LSTM
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :]
        Xs.append(v)
        ys.append(y[i + time_steps]) # targetul este valoarea din ziua de dupa secventa
    return np.array(Xs), np.array(ys)

# seturile de antrenare si testare
X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test_scaled, time_steps)

print(f"\nDimensiunea setului de antrenare (X_train_lstm) dupa crearea secventelor: {X_train_lstm.shape}")
print(f"\nDimensiunea setului de antrenare (y_train_lstm) dupa crearea secventelor: {y_train_lstm.shape}")
print(f"\nDimensiunea setului de testare (X_test_lstm) dupa crearea secventelor: {X_test_lstm.shape}")
print(f"\nDimensiunea setului de testare (y_test_lstm) dupa crearea secventelor: {y_test_lstm.shape}")

# %%
# --- Modelul LSTM ---

model = Sequential()

# units: nr de neuroni LSTM
# return_sequences: true - vreau ca urmatorul strat LSTM sa primeasca secvente
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, X_train_lstm.shape[2])))
model.add(Dropout(0.2)) # prevenirea overfittingului

model.add(LSTM(units=50, return_sequences=False)) # ultimul strat nu returneaza secvente
model.add(Dropout(0.2))

# units: 1 -> prezic o singura valoare (pretul de inchidere ajustat)
model.add(Dense(units=1))

# %%
# compilarea modelului
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# antrenarea modelului

# epochs: nr cicluri complete de antrenare pe tot setul de date
# batch_size: nr exemple de antrenament utilizate intr o iteratie
# validation_split: procent din datele de antrenament folosite pentru validare in timpul antrenarii
history = model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=32, validation_split=0.1)

print("\nAntrenarea modelului LSTM a fost finalizata cu succes.")

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# --- Evaluarea modelului ---

print("\nEvaluarea Modelului pe setul de testare...")

# predictii pe setul de testare
predictions_scaled = model.predict(X_test_lstm)

# aplic inversa scalarii pentru a obtine predictiile in unitatea monetara reala
predictions = scaler_y.inverse_transform(predictions_scaled)

actual_prices = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1))


# evaluarea performantei modelului
# RMSE (Radacina Mediei Patrate a Erorilor)
# RMSE este o masura a diferentei dintre valorile prezise de model si valorile reale
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))

# MAE (Media Erorilor Absolute)
# MAE este o masura a diferentei mediei absolute dintre valorile prezise de model si cele reale
# MAE este mai putin sensibil la valorile extreme decat RMSE
mae = mean_absolute_error(actual_prices, predictions)

print(f"RMSE pe setul de testare: {rmse:.2f}")
print(f"MAE pe setul de testare: {mae:.2f}")

# %%
# vizualizarea predictiilor si a valorilor reale

# datele de testare incep de la indexul train_size + time_steps pentru a exclude datele de antrenare
test_dates = final_data.index[train_size + time_steps:]

plt.figure(figsize=(14, 7))
plt.plot(test_dates, actual_prices, label='Pret real (Adj Close)', color='blue')
plt.plot(test_dates, predictions, label='Predictie Model LSTM', color='red', linestyle='--')
plt.title(f'Predictia Preturilor Actiunilor {simbol_actiuni} vs Valori Reale')
plt.xlabel('Data')
plt.ylabel('Pret')
plt.legend()
plt.grid(True)
plt.show()

print("\nEvaluarea a fost finalizata.")

# %%
# --- Predictia pretului pentru urmatoarea zi ---
# obtinerea ultimelor 60 de zile din setul de date  
last_60_days = final_data[-time_steps:][features].values
last_60_days_scaled = scaler_X.transform(last_60_days)
last_60_days_scaled = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], last_60_days_scaled.shape[1]))
# predictia pretului pentru urmatoarea zi
predicted_price_scaled = model.predict(last_60_days_scaled)
predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
print(f"\nPretul prezis pentru urmatoarea zi este: {predicted_price[0][0]:.2f} USD")
print("\nPredictia a fost finalizata.")
