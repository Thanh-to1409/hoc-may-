from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Global scaler for use across models and predictions
scaler = MinMaxScaler()

# Function to load data and clean NaNs and outliers
def load_data():
    df = pd.read_csv('bmi_data.csv')
    
    # Convert columns to numeric, forcing errors to NaN
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Height(Inches)'] = pd.to_numeric(df['Height(Inches)'], errors='coerce')
    df['Weight(Pounds)'] = pd.to_numeric(df['Weight(Pounds)'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')

    # Drop rows with NaN values after conversion
    bmi_data_cleaned = df.dropna()

    # Detect and remove outliers using IQR
    Q1 = bmi_data_cleaned[['Age', 'Height(Inches)', 'Weight(Pounds)', 'BMI']].quantile(0.25)
    Q3 = bmi_data_cleaned[['Age', 'Height(Inches)', 'Weight(Pounds)', 'BMI']].quantile(0.75)
    IQR = Q3 - Q1

    outliers = (bmi_data_cleaned[['Age', 'Height(Inches)', 'Weight(Pounds)', 'BMI']] < (Q1 - 1.5 * IQR)) | (bmi_data_cleaned[['Age', 'Height(Inches)', 'Weight(Pounds)', 'BMI']] > (Q3 + 1.5 * IQR))
    bmi_data_cleaned = bmi_data_cleaned[~outliers.any(axis=1)]

    features = bmi_data_cleaned[['Age', 'Height(Inches)', 'Weight(Pounds)']]
    target = bmi_data_cleaned['BMI']
    return features, target

# Function to train and evaluate linear regression model
def train_and_evaluate_linear_model():
    features, target = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return model, mse, mae, r2

# Function to train Ridge Regression model
def train_ridge_model():
    features, target = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    # Scale data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    model = RidgeCV(alphas=alphas)
    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return model, mse, mae, r2

# Function to train Neural Network model
def train_neural_model():
    features, target = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    # Scale data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=3000, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    y_test_pred = mlp.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return mlp, mse, mae, r2

# Function to train Stacking model
def train_stacking_model():
    features, target = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    # Scale data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(random_state=1, max_iter=3000)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
    linreg = LinearRegression()

    stacking_model = StackingRegressor(
        estimators=[('mlp', mlp), ('ridge', ridge), ('linreg', linreg)],
        final_estimator=RidgeCV()
    )

    stacking_model.fit(X_train_scaled, y_train)

    y_test_pred = stacking_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return stacking_model, mse, mae, r2

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction and return evaluation metrics
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form and convert to float
        age = float(request.form['Age'])
        height = float(request.form['Height(Inches)'])
        weight = float(request.form['Weight(Pounds)'])

        selected_model = request.form['model_type']

        # Log received input for debugging
        print(f"Received input: Age={age}, Height(Inches)={height}, Weight(Pounds)={weight}, Model={selected_model}")

        if selected_model == 'linear':
            model, mse, mae, r2 = train_and_evaluate_linear_model()
        elif selected_model == 'ridge':
            model, mse, mae, r2 = train_ridge_model()
        elif selected_model == 'neural':
            model, mse, mae, r2 = train_neural_model()
        elif selected_model == 'stacking':
            model, mse, mae, r2 = train_stacking_model()
        else:
            return render_template('index.html', prediction_text="Invalid model selection")

        # Prepare input data for prediction
        input_features = pd.DataFrame([[age, height, weight]], columns=['Age', 'Height(Inches)', 'Weight(Pounds)'])

        # Apply scaling if the selected model is not linear regression
        if selected_model != 'linear':
            input_features = scaler.transform(input_features)

        predicted_bmi = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f'Predicted BMI: {predicted_bmi}',
                               mse=f'MSE: {mse}',
                               mae=f'MAE: {mae}',
                               r2=f'R²: {r2}')
    except ValueError as ve:
        return render_template('index.html', prediction_text=f"Input error: {str(ve)}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
