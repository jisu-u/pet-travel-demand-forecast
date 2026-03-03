import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. CSV 읽기
df = pd.read_csv("multiTimeline.csv", skiprows=2)
df = df.iloc[:, 0:2]
df.columns = ["Time", "Search"]

# 2. 날짜 변환
df["Time"] = pd.to_datetime(df["Time"])

# 3. 시간 숫자 인덱스 생성
df["t"] = np.arange(len(df))

# 4. 모델 학습
X = df[["t"]]
y = df["Search"]

model = LinearRegression()
model.fit(X, y)

# 5. 미래 12개월 예측
future_t = np.arange(len(df), len(df) + 12).reshape(-1, 1)
future_pred = model.predict(future_t)

# 6. 시각화
plt.plot(df["t"], y, label="Actual")
plt.plot(future_t, future_pred, label="Predicted")
plt.legend()
plt.title("Pet Travel Demand Forecast")
plt.show()