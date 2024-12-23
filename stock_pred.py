import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prophet

data=pd.read_csv(r"C:\Users\shani\OneDrive\Pictures\WRK\myself\google-stock-price-prediction\goog.csv")
data.head()

plt.style.use("fivethirtyeight")
plt.figure(figsize=(16,8))
plt.title("google closing stock price")
plt.plot(data["Close"])
plt.xlabel("data",fontsize=18)
plt.ylabel("close price usd($)",fontsize=18)
plt.show()


data=data[["Date","Close"]]
data=data.rename(columns={"Date":"ds","Close":"y"})
data.head()

from prophet import Prophet
m=Prophet(daily_seasonality=True)
m.fit(data)

future=m.make_future_dataframe(periods=365)
predictions=m.predict(future)
m.plot(predictions)
plt.title("prediction of google stock price")
plt.xlabel("date")
plt.ylabel("closing stock price")
plt.show()


m.plot_components(predictions)
plt.show()
