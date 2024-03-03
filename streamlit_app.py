import streamlit as st
import pickle
import yfinance as yf
from keras.models import load_model
import numpy as np
from datetime import datetime, timedelta

# Load the Scaler object
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Load the model
model = load_model("model.h5")


# Define prediction function
def predict(stock_code, category_input):
    try:
        # Get current date
        current_date = datetime.now().date()

        # Calculate end date (5 days from today)
        end_date = current_date + timedelta(days=5)

        # Download stock data
        df = yf.download(stock_code, end=end_date).reset_index()

        # Extract selected category data
        df_category = df[category_input].tolist()

        # Reshape user input data into a 2D array with one column and as many rows as there are data points
        user_data_reshaped = np.array(df_category).reshape(-1, 1)

        # Scale user input using loaded scaler object
        user_data_scaled = scaler.transform(user_data_reshaped)

        # Use loaded model to make predictions
        predictions = model.predict(user_data_scaled)

        # Inverse transform predicted values
        predicted_values_original_scale = scaler.inverse_transform(predictions)

        # Display predicted values in original scale
        st.write("Predicted values for the next 5 days:")
        for i in range(5):
            prediction_date = (current_date + timedelta(days=i)).strftime('%Y-%m-%d')
            st.write("Date:", prediction_date, "-", category_input + ":", predicted_values_original_scale[i][0])

    except Exception as e:
        st.error("An error occurred: {}".format(e))


# Define main function
def main():
    st.title("Stock Price Analysis")

    # Get user input
    stock_code = st.text_input("Enter Stock code:")
    category_input = st.selectbox("Select category to predict:",
                                  ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    if st.button("Predict for Next 5 Days"):
        predict(stock_code, category_input)


if __name__ == "__main__":
    main()
