# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # Load data from Google Sheet
# scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# creds = ServiceAccountCredentials.from_json_keyfile_name('D:\\Study\\Automated-power-consumption-analysis\\Streamlit\\credentials.json', scope)
# client = gspread.authorize(creds)

# sheet_url = 'https://docs.google.com/spreadsheets/d/1iI1RS3f3wBkvSMzKF8MmXkZapHVXWWE2JTZis12sWDs/edit'
# sheet = client.open_by_url(sheet_url).sheet1
# data = pd.DataFrame(sheet.get_all_records())

# # Preprocess the data
# data["DATE"] = pd.to_datetime(data["DATE"])
# data = data.set_index("DATE")

# # Streamlit app
# st.set_page_config(page_title="Energy Consumption Dashboard", layout="wide")

# # Add a title
# st.title("PowerSight")

# # Add a sidebar for user inputs
# st.sidebar.title("Filters")
# selected_venue = st.sidebar.selectbox("Select a Venue", options=data.columns[7:])

# # Date range input
# start_date, end_date = st.sidebar.date_input("Select a Date Range", value=[data.index.min(), data.index.max()])

# # Filter the data based on user inputs
# filtered_data = data.loc[start_date:end_date, [selected_venue]]
# filtered_data = filtered_data.rename(columns={selected_venue: "Energy Consumption"})

# # Display visualizations
# st.header(f"Energy Consumption for {selected_venue}")

# # Line chart
# fig, ax = plt.subplots(figsize=(12, 6))
# ax = sns.lineplot(data=filtered_data, x=filtered_data.index, y="Energy Consumption")
# ax.set_xlabel("Date")
# ax.set_ylabel("Energy Consumption")
# ax.set_title(f"Energy Consumption over Time for {selected_venue}")
# st.pyplot(fig)

# # Interactive bar chart
# fig = px.bar(filtered_data, x=filtered_data.index, y="Energy Consumption", title=f"Energy Consumption for {selected_venue}")
# st.plotly_chart(fig)

# # Statistical summary
# st.subheader("Statistical Summary")
# st.write(filtered_data.describe())

# # Maximum and minimum values
# st.subheader("Maximum and Minimum Values")
# max_value = filtered_data["Energy Consumption"].max()
# min_value = filtered_data["Energy Consumption"].min()
# st.write(f"Maximum Energy Consumption: {max_value}")
# st.write(f"Minimum Energy Consumption: {min_value}")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import urllib3
import requests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

urllib3.disable_warnings()

s = requests.Session()

response = s.get('https://docs.google.com/spreadsheets/d/1iI1RS3f3wBkvSMzKF8MmXkZapHVXWWE2JTZis12sWDs/edit', verify=False)

xy = response.text

# Load data from Google Sheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('D:\\Study\\Automated-power-consumption-analysis\\Streamlit\\credentials.json', scope)
client = gspread.authorize(creds)

sheet_url = xy
sheet = client.open_by_url(sheet_url).sheet1
data = pd.DataFrame(sheet.get_all_records())

# Preprocess the data
data["DATE"] = pd.to_datetime(data["DATE"])
data = data.set_index("DATE")

# Streamlit app
st.set_page_config(page_title="PowerSight", layout="wide")

# Add a title
st.title("PowerSight")

# Add a sidebar for user inputs
st.sidebar.title("Filters")
selected_venue = st.sidebar.selectbox("Select a Venue", options=data.columns[7:])

graph_type = st.sidebar.radio("Select Graph Type", ["Energy Consumption", "Voltage/Amp"])

if graph_type == "Energy Consumption":
    if st.sidebar.button("Show Energy Consumption"):
        # Date range input
        start_date, end_date = st.sidebar.date_input("Select a Date Range", value=[data.index.min(), data.index.max()])

        # Filter the data based on user inputs
        filtered_data = data.loc[start_date:end_date, [selected_venue]]
        filtered_data = filtered_data.rename(columns={selected_venue: "Energy Consumption"})

        # Display visualizations
        st.header(f"Energy Consumption for {selected_venue}")

        # Line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.lineplot(data=filtered_data, x=filtered_data.index, y="Energy Consumption", label="Energy Consumption")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy Consumption")
        ax.set_title(f"Energy Consumption over Time for {selected_venue}")
        ax.legend()
        st.pyplot(fig)

        # Interactive bar chart
        fig = px.bar(filtered_data, x=filtered_data.index, y="Energy Consumption", title=f"Energy Consumption for {selected_venue}")
        st.plotly_chart(fig)

        # Statistical summary
        st.subheader("Statistical Summary")
        st.write(filtered_data.describe())

        # Maximum and minimum values
        st.subheader("Maximum and Minimum Values")
        max_value = filtered_data["Energy Consumption"].max()
        min_value = filtered_data["Energy Consumption"].min()
        st.write(f"Maximum Energy Consumption: {max_value}")
        st.write(f"Minimum Energy Consumption: {min_value}")

elif graph_type == "Voltage/Amp":
    parameter = st.sidebar.selectbox("Select Parameter", ["VOLTAGE", "AMP"])

    # Check if selected venue and parameter exist in the DataFrame
    if selected_venue in data.columns and parameter in data.columns:
        if st.sidebar.button("Show Voltage/Amp"):
            start_date, end_date = st.sidebar.date_input("Select a Date Range", value=[data.index.min(), data.index.max()])

            # Filter the data based on user inputs
            filtered_data = data.loc[start_date:end_date, [selected_venue, parameter]]
            filtered_data = filtered_data.rename(columns={selected_venue: "Energy Consumption", parameter: "Parameter"})

            # Display visualizations
            st.header(f"{parameter} for {selected_venue}")

            # Line chart for voltage or amp
            fig, ax = plt.subplots(figsize=(12, 6))
            ax = sns.lineplot(data=filtered_data, x=filtered_data.index, y="Parameter", label=parameter)
            ax.set_xlabel("Date")
            ax.set_ylabel(parameter)
            ax.set_title(f"{parameter} over Time for {selected_venue}")
            ax.legend()
            st.pyplot(fig)

            # Interactive bar chart
            fig = px.bar(filtered_data, x=filtered_data.index, y="Parameter", title=f"{parameter} for {selected_venue}")
            st.plotly_chart(fig)

            # Statistical summary
            st.subheader("Statistical Summary")
            st.write(filtered_data.describe())

            # Maximum and minimum values
            st.subheader("Maximum and Minimum Values")
            max_value = filtered_data["Parameter"].max()
            min_value = filtered_data["Parameter"].min()
            st.write(f"Maximum {parameter}: {max_value}")
            st.write(f"Minimum {parameter}: {min_value}")
    else:
        st.sidebar.error("Selected venue or parameter not found in the data.")



