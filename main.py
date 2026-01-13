import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("car_price_prediction.csv")

def cleanData(df):
    # Trun Levy Values of "-" to 0 then turns col to int
    df["Levy"] = df["Levy"].replace("-", "0").astype(int)

    # Maps Leather Interior to 0/1
    df["Leather interior"] = df["Leather interior"].map({"Yes": 1, "No": 0})

    # Splits Engine Volume and Turbo
    df[['Engine volume', 'Turbo']] = df['Engine volume'].str.split(" ", expand=True)

    # Turns Engine Volume into float
    # Turns Turbo into 0/1
    df["Engine volume"] = df["Engine volume"].astype(float)
    df["Turbo"] = df["Turbo"].apply(lambda x: 1 if str(x) == "Turbo" else 0)

    # Splits Mileage and KM
    df[["Mileage", "KM"]] = df["Mileage"].str.split(" ", expand=True)

    # Turns Mileage into int
    df["Mileage"] = df["Mileage"].astype(int)


    # Drops KM and ID
    df = df.drop(columns=["KM", "ID"])

    # Removes Junk and Super Cars
    df = df[df["Price"] > 1000]
    df = df[df["Price"] < 200000]

    return df

data = cleanData(data)

# Sets Features and Target Split
x = data.drop(columns=["Price"])
y = data["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=500
)

# Catgorical and Numeric Columns
categorical_cols = [
    "Manufacturer", "Model", "Category", "Fuel type",
    "Gear box type", "Drive wheels", "Doors", "Wheel", "Color"
]

numeric_cols = [
    "Levy", "Prod. year", "Engine volume", "Mileage",
    "Cylinders", "Airbags", "Leather interior", "Turbo"
]

# Preprocessing
preprocess = ColumnTransformer(
    # One Hot Encoder for Categorical Cols
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

preprocess.fit(x_train)

x_train_final = preprocess.transform(x_train) 
x_test_final = preprocess.transform(x_test)

# Random Forest Regressor Model

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=500,
    n_jobs=-1
)

model.fit(x_train_final, y_train)
y_pred = model.predict(x_test_final)

print("R2 Score:", r2_score(y_test, y_pred))
