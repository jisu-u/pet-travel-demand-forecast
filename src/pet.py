import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/multiTimeline_services_KR.csv")

df = df.iloc[:, :3]
df.columns = ["Time","Airline","Boarding"]

df["Time"] = pd.to_datetime(df["Time"])
df["t"] = np.arange(len(df))

services = ["Airline","Boarding"]

plt.figure(figsize=(10,6))

for s in services:
    X = df[["t"]]
    y = df[s]

    model = LinearRegression()
    model.fit(X,y)

    future = np.arange(len(df),len(df)+12).reshape(-1,1)
    pred = model.predict(future)

    plt.plot(df["t"],y,label=s+" actual")
    plt.plot(future,pred,"--",label=s+" forecast")

plt.title("South Korea Pet Travel Services Demand")
plt.legend()
plt.show()