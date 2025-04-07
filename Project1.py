import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error



df = pd.read_csv("PJME_hourly.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')


df.plot(style='.', figsize=(15, 5), title='PJME Energy Consumption')
plt.ylabel('MW')
plt.show()

df_model = df.copy()


df_model['hour'] = df_model.index.hour
df_model['dayofweek'] = df_model.index.dayofweek
df_model['month'] = df_model.index.month

df_model['lag_1'] = df_model['PJME_MW'].shift(1)
df_model['lag_24'] = df_model['PJME_MW'].shift(24)
df_model['lag_168'] = df_model['PJME_MW'].shift(168)  # 1 week

df_model = df_model.dropna()

split_date = '2015-01-01'
train = df_model[df_model.index < split_date]
test = df_model[df_model.index >= split_date]

X_train = train.drop('PJME_MW', axis=1)
y_train = train['PJME_MW']
X_test = test.drop('PJME_MW', axis=1)
y_test = test['PJME_MW']

model = XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=100)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Test RMSE: {rmse:.2f} MW")

plt.figure(figsize=(15, 5))
plt.plot(y_test.index, y_test, label='Actual', color='blue', linewidth=2, alpha=0.7)
plt.plot(y_test.index, preds, label='Predicted', color='orange', linewidth=2, alpha=0.7, linestyle='--')
plt.legend()
plt.title('XGBoost Energy Forecast - PJME')
plt.ylabel('MW')
plt.tight_layout()
plt.show()
