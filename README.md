# Car Price Prediction

A machine learning project that predicts car prices using a Random Forest Regressor model with comprehensive data preprocessing and feature engineering.

## Project Overview

This project analyzes car pricing data and builds a predictive model using scikit-learn. The model processes various car attributes including manufacturer, model, production year, mileage, and technical specifications to predict market prices.

## Features

- **Data Cleaning Pipeline**: Handles missing values, converts data types, and removes outliers
- **Feature Engineering**: 
  - Extracts turbo information from engine volume
  - Converts categorical features to binary indicators
  - Separates mileage values from units
- **One-Hot Encoding**: Transforms categorical variables (manufacturer, model, fuel type, etc.)
- **Random Forest Regressor**: Ensemble model with 300 estimators for accurate predictions

## Dataset

The dataset (`car_price_prediction.csv`) contains the following features:
- Price (target variable)
- Manufacturer, Model, Category
- Production year
- Engine volume, Cylinders, Turbo
- Mileage, Levy
- Fuel type, Gear box type, Drive wheels
- Doors, Wheel, Color
- Leather interior, Airbags

## Installation

1. Create a virtual environment:
```bash
python -m venv sklearn-env
```

2. Activate the environment:
```bash
# Windows
sklearn-env\Scripts\activate

# Linux/Mac
source sklearn-env/bin/activate
```

3. Install dependencies:
```bash
pip install pandas scikit-learn
```

## Usage

Run the prediction model:
```bash
python main.py
```

The script will:
1. Load and clean the data
2. Split data into training and test sets (80/20)
3. Apply preprocessing transformations
4. Train the Random Forest model
5. Output the R² score

## Model Performance

The model uses:
- **Algorithm**: Random Forest Regressor
- **Estimators**: 300 trees
- **Train/Test Split**: 80/20
- **Random State**: 500

## Data Preprocessing

1. **Levy**: Replaces "-" with 0, converts to integer
2. **Leather Interior**: Maps Yes/No to 1/0
3. **Engine Volume**: Extracts numeric value, converts to float
4. **Turbo**: Binary indicator (1 if turbo, 0 otherwise)
5. **Mileage**: Extracts numeric value, converts to integer
6. **Price Filtering**: Removes outliers (<$1,000 or >$200,000)
7. **Dropped Columns**: ID, KM (unit label)

## Output

- Console displays the R² score of the model
- Generates `cleaned_car_prices.csv` with preprocessed data

## Requirements

- Python 3.7+
- pandas
- scikit-learn

## License

This project is for educational purposes.
