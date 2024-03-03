import yfinance as yf
from keras.models import load_model
import pickle
import numpy as np

def get_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.strip():  # Check if input is not empty
            return user_input.strip()
        else:
            print("Input cannot be empty. Please try again.")

try:
    # User input for start date, end date, and stock code
    start_date = get_user_input("Enter the start date (YYYY-MM-DD): ")
    end_date = get_user_input("Enter the end date (YYYY-MM-DD): ")
    stock_code = get_user_input("Enter the stock code: ")

    # Download stock data
    df = yf.download(stock_code, start=start_date, end=end_date).reset_index()

    # Prompt user to select category to predict
    print("Available categories: Open, High, Low, Close, Adj Close, Volume")
    category_input = get_user_input("Enter the category you want to predict: ").capitalize()

    if category_input not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        raise ValueError("Invalid category selected.")

    # Extract selected category data
    df_category = df[category_input].tolist()

    # Load scaler object
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    # Load trained model
    loaded_model = load_model('model.h5')

    # Reshape user input data into a 2D array with one row and three columns
    user_data_reshaped = np.array(df_category).reshape(-1, 1)

    # Scale user input using loaded scaler object
    user_data_scaled = scaler.transform(user_data_reshaped)

    # Use loaded model to make predictions
    predictions = loaded_model.predict(user_data_scaled)

    # Inverse transform predicted value
    predicted_value_original_scale = scaler.inverse_transform(predictions)

    # Display predicted value in original scale
    print("Predicted value in original scale:", predicted_value_original_scale[0][0])

except Exception as e:
    print("An error occurred:", e)
