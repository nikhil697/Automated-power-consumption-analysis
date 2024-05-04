# import streamlit as st
# import pandas as pd 
# import streamlit.components.v1 as stc
# import plotly.express as px
# import time
# from streamlit_option_menu import option_menu
# from numerize.numerize import numerize
# import plotly.express as px
# import plotly.subplots as sp
# import plotly.graph_objects as go
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # Page behavior
# st.set_page_config(page_title="Descriptive Analytics ", page_icon="🌎", layout="wide")  

# # Remove default theme
# theme_plotly = None # None or streamlit

# # CSS Style
# with open('D:\Study\Automated-power-consumption-analysis\power_consumption\dashboard\streamlit\dashboardstyle.css') as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # Load Excel file
# scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# creds = ServiceAccountCredentials.from_json_keyfile_name('D:\Study\Automated-power-consumption-analysis\power_consumption\dashboard\streamlit\credentials.json', scope)
# client = gspread.authorize(creds)

# # Load the Google Sheet
# sheet_url = 'https://docs.google.com/spreadsheets/d/1jnDltb2tKZDTEOwaLA9RL0_ovmwEVJXJkc3-QpyxBrA/edit'
# sheet = client.open_by_url(sheet_url)
# worksheet = sheet.get_worksheet(0)  # Assuming the data is on the first sheet

# # Get all values from the sheet
# rows = worksheet.get_all_values()

# # Convert to DataFrame
# df = pd.DataFrame(rows[1:], columns=rows[0])

# # Sidebar filters
# st.sidebar.header("Please Filter Here:")
# region = st.sidebar.multiselect(
#     "Select the Region:",
#     options=df["Region"].unique(),
#     default=df["Region"].unique()
# )
# location = st.sidebar.multiselect(
#     "Select the Location:",
#     options=df["Location"].unique(),
#     default=df["Location"].unique(),
# )
# construction = st.sidebar.multiselect(
#     "Select the Construction:",
#     options=df["Construction"].unique(),
#     default=df["Construction"].unique()  
# )
# df_selection = df.query(
#     "Region == @region & Location == @location & Construction == @construction"
# )

# df_selection['Investment'] = pd.to_numeric(df_selection['Investment'])
# df_selection['Rating'] = pd.to_numeric(df_selection['Rating'], errors='coerce')

# # Define functions
# def HomePage():
#     # Print dataframe
#     with st.expander("🧭 My database"):
#         shwdata = st.multiselect('Filter :', df_selection.columns, default=[])
#         st.dataframe(df_selection[shwdata], use_container_width=True)

#     # Compute top Analytics
#     total_investment = df_selection['Investment'].sum()
#     investment_mode = df_selection['Investment'].mode().iloc[0]
#     investment_mean = df_selection['Investment'].mean()
#     investment_median = df_selection['Investment'].median() 
#     rating = df_selection['Rating'].sum()

#     # Display metrics
#     total1, total2, total3, total4, total5 = st.columns(5, gap='large')
#     with total1:
#         st.info('Total Investment', icon="🔍")
#         st.metric(label='sum TZS', value=f"{total_investment:,.0f}")

#     with total2:
#         st.info('Most frequently', icon="🔍")
#         st.metric(label='Mode TZS', value=f"{investment_mode:,.0f}")

#     with total3:
#         st.info('Investment Average', icon="🔍")
#         st.metric(label='Mean TZS', value=f"{investment_mean:,.0f}")

#     with total4:
#         st.info('Investment Marging', icon="🔍")
#         st.metric(label='Median TZS', value=f"{investment_median:,.0f}")

#     with total5:
#         st.info('Ratings of the data', icon="🔍")
#         st.metric(label='Rating', value=f"{rating:,.0f}", help=f"Total rating: {rating}")

#     st.markdown("""---""")

# def Graphs():
#     # Bar graph: Investment by Business Type
#     investment_by_businessType = (
#         df_selection.groupby(by=["BusinessType"]).count()[["Investment"]].sort_values(by="Investment")
#     )
#     fig_investment = px.bar(
#         investment_by_businessType,
#         x="Investment",
#         y=investment_by_businessType.index,
#         orientation="h",
#         title="Investment by Business Type",
#         color_discrete_sequence=["#0083B8"] * len(investment_by_businessType),
#         template="plotly_white",
#     )
#     fig_investment.update_layout(
#         plot_bgcolor="rgba(0,0,0,0)",
#         xaxis=(dict(showgrid=False))
#     )

#     # Line graph: Investment by Region
#     investment_by_state = df_selection.groupby(by=["State"]).count()[["Investment"]]
#     fig_state = px.line(
#         investment_by_state,
#         x=investment_by_state.index,
#         orientation="v",
#         y="Investment",
#         title="Investment by Region ",
#         color_discrete_sequence=["#0083B8"] * len(investment_by_state),
#         template="plotly_white",
#     )
#     fig_state.update_layout(
#         xaxis=dict(tickmode="linear"),
#         plot_bgcolor="rgba(0,0,0,0)",
#         yaxis=(dict(showgrid=False)),
#     )

#     # Pie chart: Regions by Ratings
#     fig = px.pie(df_selection, values='Rating', names='State', title='Regions by Ratings')
#     fig.update_layout(legend_title="Regions", legend_y=0.9)
#     fig.update_traces(textinfo='percent+label', textposition='inside')

#     # Display graphs
#     left_column, right_column, center = st.columns(3)
#     left_column.plotly_chart(fig_state, use_container_width=True)
#     right_column.plotly_chart(fig_investment, use_container_width=True)
#     with center:
#         st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

# def ProgressBar():
#     st.markdown("""<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #99ff99 , #FFFF00)}</style>""", unsafe_allow_html=True)
#     target = 3000000000
#     current = df_selection['Investment'].sum()
#     percent = round((current / target * 100))
#     my_bar = st.progress(0)

#     if percent > 100:
#         st.subheader("Target 100 completed")
#     else:
#         st.write(f"You have {percent}% of {format(target, ',d')} TZS")
#         for percent_complete in range(percent):
#             time.sleep(0.1)
#             my_bar.progress(percent_complete + 1, text="Target percentage")

# # Print dataframe and sidebar
# HomePage()

# # Print graphs
# Graphs()

# # Print progress bar
# ProgressBar()

# # Footer
# footer = """
# <style>
# a:hover,  a:active {
#     color: red;
#     background-color: transparent;
#     text-decoration: underline;
# }

# .footer {
#     position: fixed;
#     left: 0;
#     height: 5%;
#     bottom: 0;
#     width: 100%;
#     background-color: #243946;
#     color: white;
#     text-align: center;
# }
# </style>
# """
# st.markdown(footer, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go
import pymysql
from datetime import datetime
from datetime import time
import plotly.express as px

# import urllib3
# import requests
# from urllib3.exceptions import InsecureRequestWarning
# from urllib3 import disable_warnings

# urllib3.disable_warnings()

# s = requests.Session()

# response = s.get('https://docs.google.com/spreadsheets/d/1iI1RS3f3wBkvSMzKF8MmXkZapHVXWWE2JTZis12sWDs/edit', verify=False)

# xy = response.text

# # Load data from Google Sheet
# scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# creds = ServiceAccountCredentials.from_json_keyfile_name('D:\\Study\\Automated-power-consumption-analysis\\Streamlit\\credentials.json', scope)
# client = gspread.authorize(creds)

# sheet_url = xy
# sheet = client.open_by_url(sheet_url).sheet1
# data = pd.DataFrame(sheet.get_all_records())
# Custom CSS
custom_css = """
<style>
h1 {
    color: #4c72b0;
    font-weight: bold;
}
h2 {
    color: #55a868;
    font-weight: bold;
}
.metric-label {
    color: #6c757d;
    font-weight: bold;
}
.metric-value {
    color: #ffff00;
    font-size: 1.2rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)




# st.set_page_config(page_title="PowerSight", layout="wide")

# Establish connection to MySQL database
connection = pymysql.connect(host='powersight.cdy8ikaymuro.ap-south-1.rds.amazonaws.com',
                             user='admin',
                             password='Nikhil2002',
                             database='powersight',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

# Execute SQL query to fetch data
with connection.cursor() as cursor:
    sql_query = "SELECT * FROM elecread;"
    cursor.execute(sql_query)
    data = cursor.fetchall()

# Close connection
df = pd.DataFrame(data)

# Drop the default index column if present
# if 'id' in df.columns:
#     df = df.drop(columns=['id'])
st.data_editor(
    df,
    hide_index=True,
)


df = pd.DataFrame(data)

# Data preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df['TIME'] = pd.to_datetime(df['DATE'].dt.date.astype(str) + ' ' + df['TIME'], format='%Y-%m-%d %H:%M')
df['KWH'] = df['KWH'].astype(float)

# Sidebar filters
st.sidebar.title("Filters")
date_range = st.sidebar.date_input("Select Date Range", value=[df['DATE'].min(), df['DATE'].max()], min_value=df['DATE'].min(), max_value=df['DATE'].max())
venue_options = ['Sixth_FLOOR_CS', 'PROFESSOR_QUATERS', 'HOSTEL_N_ROAD_SIDE', 'lc_1', 'lc_2']
venue_filter = st.sidebar.radio("Select Venue", options=venue_options)




# Filter data based on user selections
filtered_df = df[(df['DATE'].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))) & (df[venue_filter] > 0)]


# Main content
st.title("PowerSight Dashboard")

# KPI cards
total_kwh = filtered_df['KWH'].sum()
avg_voltage = filtered_df['VOLTAGE'].mean().round(2)
avg_amp = filtered_df['AMP'].mean().round(2)

col1, col2, col3 = st.columns(3)
col1.metric("Total KWH", f"{total_kwh:,.2f}")
col2.metric("Average Voltage", f"{avg_voltage:,.2f} V")
col3.metric("Average Ampere", f"{avg_amp:,.2f} A")


# Filter DataFrame for TR_NO 1 and 2
filtered_df_tr1 = filtered_df[filtered_df['TR_NO'] == 1]
filtered_df_tr2 = filtered_df[filtered_df['TR_NO'] == 2]

# Create double bar graph
fig = go.Figure(data=[
    go.Bar(name='TR_NO 1', x=filtered_df_tr1['TIME'], y=filtered_df_tr1[venue_filter]),
    go.Bar(name='TR_NO 2', x=filtered_df_tr2['TIME'], y=filtered_df_tr2[venue_filter])
])

# Update layout
fig.update_layout(barmode='group', title="Energy Consumption Over Time",
                  xaxis_title="Time", yaxis_title="Energy Consumption", legend_title="Venue")

# Add markers to show exact values with different colors for TR_NO 1 and TR_NO 2
colors = ['blue', 'red']
for i, trace in enumerate(fig.data):
    trace['text'] = [f'<b>{val}</b>' for val in trace['y']]
    trace['textposition'] = 'outside'
    trace['textfont'] = dict(color=colors[i])
    
# Show the plotly chart
st.plotly_chart(fig, use_container_width=True)

# # Pie Chart
# pie_date = st.sidebar.date_input("Select Date for Pie Chart", value=df['DATE'].max(), min_value=df['DATE'].min(), max_value=df['DATE'].max())
# pie_df = df[df['DATE'].dt.date == pie_date]
# st.subheader(f"Pie Charts for {pie_date}")

# # Pie chart for TR_NO 1
# pie_data_1 = pie_df[pie_df['TR_NO'] == 1].groupby(venue_filter)['KWH'].sum().reset_index()
# fig = px.pie(pie_data_1, values='KWH', names=venue_filter, title=f"Distribution of KWH for TR_NO 1 on {pie_date}")
# st.plotly_chart(fig, use_container_width=True)

# # Pie chart for TR_NO 2
# pie_data_2 = pie_df[pie_df['TR_NO'] == 2].groupby(venue_filter)['KWH'].sum().reset_index()
# fig = px.pie(pie_data_2, values='KWH', names=venue_filter, title=f"Distribution of KWH for TR_NO 2 on {pie_date}")
# st.plotly_chart(fig, use_container_width=True)

st.subheader("Statistical Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='color:#55a868;'>Minimum Values</h2>", unsafe_allow_html=True)
    st.write(f"Minimum Voltage: <span class='metric-value'>{filtered_df['VOLTAGE'].min():.2f} V</span>", unsafe_allow_html=True)
    st.write(f"Minimum Ampere: <span class='metric-value'>{filtered_df['AMP'].min():.2f} A</span>", unsafe_allow_html=True)
    st.write(f"Minimum KWH: <span class='metric-value'>{filtered_df['KWH'].min():.2f}</span>", unsafe_allow_html=True)
    st.write(f"Minimum O_L: <span class='metric-value'>{filtered_df['O_L'].min():.2f}</span>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='color:#55a868;'>Maximum Values</h2>", unsafe_allow_html=True)
    st.write(f"Maximum Voltage: <span class='metric-value'>{filtered_df['VOLTAGE'].max():.2f} V</span>", unsafe_allow_html=True)
    st.write(f"Maximum Ampere: <span class='metric-value'>{filtered_df['AMP'].max():.2f} A</span>", unsafe_allow_html=True)
    st.write(f"Maximum KWH: <span class='metric-value'>{filtered_df['KWH'].max():.2f}</span>", unsafe_allow_html=True)
    st.write(f"Maximum O_L: <span class='metric-value'>{filtered_df['O_L'].max():.2f}</span>", unsafe_allow_html=True)