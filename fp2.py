import streamlit as st
import requests
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import calendar

# Set up the Streamlit interface
st.title('Electricity Consumption Forecast using Azure Auto ML REST Endpoint')
st.write('View historical data and select date range for predictions')

# Fetching historical data from GitHub
github_raw_url_industrial = 'https://raw.githubusercontent.com/adarshb3/FP2/main/elec_industrial_github2.csv'
github_raw_url_commercial = 'https://raw.githubusercontent.com/adarshb3/FP2/main/elec_commercial_github.csv'

try:
    historical_data_industrial = pd.read_csv(github_raw_url_industrial)
    historical_data_industrial['Date'] = pd.to_datetime(historical_data_industrial['Date'], format='%d-%m-%Y')
    historical_data_commercial = pd.read_csv(github_raw_url_commercial)
    historical_data_commercial['Date'] = pd.to_datetime(historical_data_commercial['Date'], format='%d-%m-%Y')
except Exception as e:
    st.error(f"Failed to load historical data from GitHub. Error: {e}")

# Year and Month Selector
years = sorted(set(historical_data_industrial['Date'].dt.year.unique()) | set(historical_data_commercial['Date'].dt.year.unique()), reverse=True)

# Mapping month numbers to names
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

start_year = st.selectbox("Select Start Year", years)
start_month = st.selectbox("Select Start Month", list(month_names.keys()), format_func=lambda x: month_names[x])
end_year = st.selectbox("Select End Year", years)
end_month = st.selectbox("Select End Month", list(month_names.keys()), format_func=lambda x: month_names[x])

start_date = datetime.date(start_year, start_month, 1)
end_day = calendar.monthrange(end_year, end_month)[1]
end_date = datetime.date(end_year, end_month, end_day)

# Prepare the data for prediction
dates_range = pd.date_range(start_date, end_date, freq='MS').tolist()
formatted_dates = [d.strftime('%Y-%m-%dT%H:%M:%S.000Z') for d in dates_range]

data = {
    "Inputs": {
        "data": [{"Date": date} for date in formatted_dates]
    },
    "GlobalParameters": {
        "quantiles": [0.025, 0.975]
    }
}

# REST API endpoints
url_industrial = 'http://f3a4ef57-ecd3-4b16-9100-874b20af60a3.eastus.azurecontainer.io/score'
url_commercial = 'http://ed5e0b71-9c23-4eeb-829f-d0daad0f4e2c.eastus.azurecontainer.io/score'

# When the user clicks the 'Predict' button
if st.button('Predict'):
    # Prepare the data for prediction with the selected month range
    # Create a list of the first day of each month in the range
    dates_range = pd.date_range(start_date, end_date, freq='MS').tolist()
    formatted_dates = [d.strftime('%Y-%m-%dT%H:%M:%S.000Z') for d in dates_range]
    
    data = {
        "Inputs": {
            "data": [{"Date": date} for date in formatted_dates]
        },
        "GlobalParameters": {
            "quantiles": [0.025, 0.975]
        }
    }

    # Make the POST requests with error handling for both sectors
    try:
        headers = {'Content-Type': 'application/json'}
        industrial_response = requests.post(url_industrial, json=data, headers=headers, timeout=120)
        commercial_response = requests.post(url_commercial, json=data, headers=headers, timeout=120)

        # Check if responses are successful
        if industrial_response.status_code == 200 and commercial_response.status_code == 200:
            industrial_predictions = industrial_response.json()
            commercial_predictions = commercial_response.json()

            if 'Results' in industrial_predictions and 'Results' in commercial_predictions:
                ind_forecast = industrial_predictions['Results']['forecast']
                com_forecast = commercial_predictions['Results']['forecast']
                ind_intervals = industrial_predictions['Results']['prediction_interval']
                com_intervals = commercial_predictions['Results']['prediction_interval']


                # Plotting the forecast
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))

                # Industrial Plot
                ax[0].plot(historical_data_industrial['Date'], historical_data_industrial['Total Energy Consumed by the Industrial Sector, Monthly'], label='Historical - Industrial')
                ax[0].plot(pd.to_datetime(formatted_dates), ind_forecast, label='Forecast - Industrial')
                ax[0].set_title('Industrial Sector Electricity Consumption')
                ax[0].legend()

                # Commercial Plot
                ax[1].plot(historical_data_commercial['Date'], historical_data_commercial['Total Energy Consumed by the Commercial Sector, Monthly'], label='Historical - Commercial')
                ax[1].plot(pd.to_datetime(formatted_dates), com_forecast, label='Forecast - Commercial')
                ax[1].set_title('Commercial Sector Electricity Consumption')
                ax[1].legend()

                for a in ax:
                    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    a.figure.autofmt_xdate()

                st.pyplot(fig)

                # Formatting dates for the DataFrame
                formatted_dates_df = pd.to_datetime(formatted_dates).strftime('%B %Y')

                # Combine forecasts and confidence intervals into a DataFrame
                combined_predictions_df = pd.DataFrame({
                    'Date': formatted_dates_df,
                    'Industrial Forecast': ind_forecast,
                    'Industrial Confidence Interval': ind_intervals,
                    'Commercial Forecast': com_forecast,
                    'Commercial Confidence Interval': com_intervals
                })
                st.write("Forecasted Energy Consumption:")
                st.dataframe(combined_predictions_df)
            else:
                st.error('Prediction data is not in the expected format.')
        else:
            st.error('Error in API response: Industrial - {}, Commercial - {}'.format(industrial_response.status_code, commercial_response.status_code))

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API endpoint. Error: {e}")
