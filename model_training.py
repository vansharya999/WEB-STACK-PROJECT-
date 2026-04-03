import pandas as pd
from sklearn.linear_model import LinearRegression   
import joblib                                        
df = pd.read_csv("used_car_price_dataset_extended.csv")   
print(df.head())
print(df.isnull().sum())

df["insurance_valid"].fillna(df["insurance_valid"].mode()[0], inplace=True)  

print(df.duplicated().sum())

print(df.info())

print(df.describe())
df = df[df["price_usd"] > 0]

Q1 = df["price_usd"].quantile(0.25)
Q3 = df["price_usd"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["price_usd"] >= Q1 - 1.5 * IQR) & (df["price_usd"] <= Q3 + 1.5 * IQR)]
df = pd.get_dummies(df, drop_first=True)

print(df.head())
print(df.shape)
X = df.drop("price_usd", axis=1)
y = df["price_usd"]

print(df.columns)
print(df.info())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

print("MAE :", mae)
print("R2  :", r2)

joblib.dump(model,          "car_price_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("Model saved to car_price_model.pkl")
print("Columns saved to model_columns.pkl")
