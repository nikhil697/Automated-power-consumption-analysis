# import streamlit as st
# import mysql.connector
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta

# # Database connection function
# def get_db_connection():
#     return mysql.connector.connect(
#         host="powersight-nikhilchadha1534-9076.c.aivencloud.com",
#         user="avnadmin",
#         password="AVNS_U6onSK7r0jsSIvu5DIg",
#         database="defaultdb",
#         port=22162
#     )

# # Function to load data
# def load_data(start_date, end_date):
#     conn = get_db_connection()
#     query = f"""
#     SELECT * FROM readings 
#     WHERE Timestamps BETWEEN '{start_date}' AND '{end_date}'
#     ORDER BY Timestamps
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# # Function to calculate statistics
# def calculate_stats(df, column):
#     stats = {
#         'Maximum': df[column].max(),
#         'Minimum': df[column].min(),
#         'Average': df[column].mean(),
#         'Standard Deviation': df[column].std(),
#         'Variance': df[column].var()
#     }
#     return stats

# # Set page config
# st.set_page_config(page_title="Power Monitoring Dashboard", layout="wide")

# # Title
# st.title("Power Monitoring Dashboard")

# # Sidebar controls
# st.sidebar.header("Controls")

# # Date range selector
# default_start_date = datetime.now() - timedelta(days=7)
# default_end_date = datetime.now()

# start_date = st.sidebar.date_input("Start Date", default_start_date)
# end_date = st.sidebar.date_input("End Date", default_end_date)

# # Graph type selector
# graph_type = st.sidebar.selectbox(
#     "Select Graph Type",
#     ["Line Plot", "Bar Chart", "Scatter Plot", "Box Plot"]
# )

# # Parameter selector
# parameter = st.sidebar.selectbox(
#     "Select Parameter",
#     ["ampere", "wattage_kwh", "pf"]
# )

# # Load data
# try:
#     df = load_data(start_date, end_date)
    
    

#     # Create visualization based on selection
#     st.header(f"{graph_type} of {parameter}")
    
#     if graph_type == "Line Plot":
#         fig = px.line(df, x='Timestamps', y=parameter,
#                      title=f'{parameter} Over Time')
        
#     elif graph_type == "Bar Chart":
#         fig = px.bar(df, x='Timestamps', y=parameter,
#                      title=f'{parameter} Distribution')
        
#     elif graph_type == "Scatter Plot":
#         fig = px.scatter(df, x='Timestamps', y=parameter,
#                         title=f'{parameter} Distribution')
        
#     elif graph_type == "Box Plot":
#         fig = px.box(df, y=parameter,
#                     title=f'{parameter} Distribution')

#     # Update layout
#     fig.update_layout(
#         xaxis_title="Timestamp",
#         yaxis_title=parameter.capitalize(),
#         height=500
#     )
    
#     # Display plot
#     st.plotly_chart(fig, use_container_width=True)

#     # Display basic stats in columns
#     st.header(f"Statistics for {parameter}")
#     stats = calculate_stats(df, parameter)
    
#     col1, col2, col3, col4, col5 = st.columns(5)
    
#     with col1:
#         st.metric("Maximum", f"{stats['Maximum']:.2f}")
#     with col2:
#         st.metric("Minimum", f"{stats['Minimum']:.2f}")
#     with col3:
#         st.metric("Average", f"{stats['Average']:.2f}")
#     with col4:
#         st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
#     with col5:
#         st.metric("Variance", f"{stats['Variance']:.2f}")
    
#     # Display raw data with pagination
#     st.header("Raw Data")
#     page_size = 10
#     total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
#     page_number = st.number_input('Page Number', min_value=1, max_value=total_pages, value=1)
#     start_idx = (page_number - 1) * page_size
#     end_idx = min(start_idx + page_size, len(df))
#     st.dataframe(df.iloc[start_idx:end_idx])

# except Exception as e:
#     st.error(f"An error occurred: {str(e)}")
#     st.error("Please check your database connection and try again.")

# import streamlit as st
# import mysql.connector
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta

# # Database connection function
# def get_db_connection():
#     return mysql.connector.connect(
#         host="powersight-nikhilchadha1534-9076.c.aivencloud.com",
#         user="avnadmin",
#         password="AVNS_U6onSK7r0jsSIvu5DIg",
#         database="defaultdb",
#         port=22162
#     )

# # Function to load data
# def load_data(start_date, end_date):
#     conn = get_db_connection()
#     query = f"""
#     SELECT * FROM readings 
#     WHERE Timestamps BETWEEN '{start_date}' AND '{end_date}'
#     ORDER BY Timestamps
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# # Function to calculate statistics
# def calculate_stats(df, column):
#     stats = {
#         'Maximum': df[column].max(),
#         'Minimum': df[column].min(),
#         'Average': df[column].mean(),
#         'Standard Deviation': df[column].std(),
#         'Variance': df[column].var()
#     }
#     return stats

# # Set page config
# st.set_page_config(page_title="Power Monitoring Dashboard", layout="wide")

# # Title
# st.title("Power Monitoring Dashboard")
# st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# # Sidebar controls
# st.sidebar.header("Controls")

# # Get today's date
# today = datetime.now().date()

# # Date range selector with today as default
# start_date = st.sidebar.date_input("Start Date", today)
# end_date = st.sidebar.date_input("End Date", today)

# # Manual refresh button
# if st.sidebar.button("Refresh Data"):
#     st.rerun()

# # Graph type selector
# graph_type = st.sidebar.selectbox(
#     "Select Graph Type",
#     ["Line Plot", "Bar Chart", "Scatter Plot", "Box Plot"]
# )

# # Parameter selector
# parameter = st.sidebar.selectbox(
#     "Select Parameter",
#     ["ampere", "wattage_kwh", "pf"]
# )

# # Load data
# try:
#     # Adjust end_date to include the entire day
#     end_datetime = datetime.combine(end_date, datetime.max.time())
#     start_datetime = datetime.combine(start_date, datetime.min.time())
    
#     df = load_data(start_datetime, end_datetime)
    
#     # Create visualization based on selection
#     st.header(f"{graph_type} of {parameter}")
    
#     if graph_type == "Line Plot":
#         fig = px.line(df, x='Timestamps', y=parameter,
#                      title=f'{parameter} Over Time')
        
#     elif graph_type == "Bar Chart":
#         fig = px.bar(df, x='Timestamps', y=parameter,
#                      title=f'{parameter} Distribution')
        
#     elif graph_type == "Scatter Plot":
#         fig = px.scatter(df, x='Timestamps', y=parameter,
#                         title=f'{parameter} Distribution')
        
#     elif graph_type == "Box Plot":
#         fig = px.box(df, y=parameter,
#                     title=f'{parameter} Distribution')

#     # Update layout
#     fig.update_layout(
#         xaxis_title="Timestamp",
#         yaxis_title=parameter.capitalize(),
#         height=500
#     )
    
#     # Display plot
#     st.plotly_chart(fig, use_container_width=True)

#     # Display basic stats in columns
#     st.header(f"Statistics for {parameter}")
#     stats = calculate_stats(df, parameter)
    
#     col1, col2, col3, col4, col5 = st.columns(5)
    
#     with col1:
#         st.metric("Maximum", f"{stats['Maximum']:.2f}")
#     with col2:
#         st.metric("Minimum", f"{stats['Minimum']:.2f}")
#     with col3:
#         st.metric("Average", f"{stats['Average']:.2f}")
#     with col4:
#         st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
#     with col5:
#         st.metric("Variance", f"{stats['Variance']:.2f}")
    
#     # Display raw data with pagination
#     st.header("Raw Data")
#     page_size = 10
#     total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
#     page_number = st.number_input('Page Number', min_value=1, max_value=total_pages, value=1)
#     start_idx = (page_number - 1) * page_size
#     end_idx = min(start_idx + page_size, len(df))
#     st.dataframe(df.iloc[start_idx:end_idx])

# except Exception as e:
#     st.error(f"An error occurred: {str(e)}")
#     st.error("Please check your database connection and try again.")
# -----------------------------------------------------------------------------------------------------------------------------------------
# import streamlit as st
# import mysql.connector
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta

# # Database connection function
# def get_db_connection():
#     return mysql.connector.connect(
#         host="powersight-nikhilchadha1534-9076.c.aivencloud.com",
#         user="avnadmin",
#         password="AVNS_U6onSK7r0jsSIvu5DIg",
#         database="defaultdb",
#         port=22162
#     )

# # Function to load data
# def load_data(start_date, end_date):
#     conn = get_db_connection()
#     query = f"""
#     SELECT * FROM readings 
#     WHERE Timestamps BETWEEN '{start_date}' AND '{end_date}'
#     ORDER BY Timestamps
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# # Function to calculate statistics
# def calculate_stats(df, column):
#     stats = {
#         'Maximum': df[column].max(),
#         'Minimum': df[column].min(),
#         'Average': df[column].mean(),
#         'Standard Deviation': df[column].std(),
#         'Variance': df[column].var()
#     }
#     return stats

# # Set page config
# st.set_page_config(page_title="Power Monitoring Dashboard", layout="wide")

# # Title
# st.title("Power Monitoring Dashboard")
# st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# # Sidebar controls
# st.sidebar.header("Controls")

# # Get today's date
# today = datetime.now().date()

# # Date range selector with today as default
# start_date = st.sidebar.date_input("Start Date", today)
# end_date = st.sidebar.date_input("End Date", today)

# # Manual refresh button
# if st.sidebar.button("Refresh Data"):
#     st.rerun()

# # Graph type selector
# graph_type = st.sidebar.selectbox(
#     "Select Graph Type",
#     ["Line Plot", "Area Plot", "Bar Chart", "Scatter Plot", "Box Plot"]
# )

# # Parameter selector with "all parameters" option
# parameter = st.sidebar.selectbox(
#     "Select Parameter",
#     ["all parameters", "ampere", "wattage_kwh", "pf"]
# )

# # Load data
# try:
#     # Adjust end_date to include the entire day
#     end_datetime = datetime.combine(end_date, datetime.max.time())
#     start_datetime = datetime.combine(start_date, datetime.min.time())
    
#     df = load_data(start_datetime, end_datetime)
    
#     # Create visualization based on selection
#     if parameter == "all parameters":
#         st.header(f"{graph_type} of All Parameters")
        
#         if graph_type == "Line Plot":
#             # Create a figure with secondary y-axis
#             fig = go.Figure()
            
#             # Add traces for each parameter
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df['ampere'], name='Ampere',
#                           line=dict(color='blue'))
#             )
            
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
#                           line=dict(color='red'))
#             )
            
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df['pf'], name='Power Factor',
#                           line=dict(color='green'))
#             )
            
#             # Update layout
#             fig.update_layout(
#                 title='All Parameters Over Time',
#                 xaxis_title='Timestamp',
#                 yaxis_title='Value',
#                 height=500,
#                 showlegend=True
#             )
            
#         elif graph_type == "Area Plot":
#             fig = go.Figure()
            
#             # Add traces for each parameter
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df['ampere'], name='Ampere',
#                           fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)',
#                           line=dict(color='blue'))
#             )
            
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
#                           fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
#                           line=dict(color='red'))
#             )
            
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df['pf'], name='Power Factor',
#                           fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)',
#                           line=dict(color='green'))
#             )
            
#             # Update layout
#             fig.update_layout(
#                 title='All Parameters Over Time',
#                 xaxis_title='Timestamp',
#                 yaxis_title='Value',
#                 height=500,
#                 showlegend=True
#             )
            
#         elif graph_type == "Bar Chart":
#             # Create subplots for bar charts
#             fig = go.Figure()
            
#             fig.add_trace(go.Bar(x=df['Timestamps'], y=df['ampere'], name='Ampere',
#                                 marker_color='blue'))
#             fig.add_trace(go.Bar(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
#                                 marker_color='red'))
#             fig.add_trace(go.Bar(x=df['Timestamps'], y=df['pf'], name='Power Factor',
#                                 marker_color='green'))
            
#             fig.update_layout(
#                 barmode='group',
#                 title='All Parameters Distribution',
#                 xaxis_title='Timestamp',
#                 yaxis_title='Value',
#                 height=500
#             )
            
#         elif graph_type == "Scatter Plot":
#             fig = go.Figure()
            
#             fig.add_trace(go.Scatter(x=df['Timestamps'], y=df['ampere'], name='Ampere',
#                                    mode='markers', marker=dict(color='blue')))
#             fig.add_trace(go.Scatter(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
#                                    mode='markers', marker=dict(color='red')))
#             fig.add_trace(go.Scatter(x=df['Timestamps'], y=df['pf'], name='Power Factor',
#                                    mode='markers', marker=dict(color='green')))
            
#             fig.update_layout(
#                 title='All Parameters Distribution',
#                 xaxis_title='Timestamp',
#                 yaxis_title='Value',
#                 height=500
#             )
            
#         elif graph_type == "Box Plot":
#             fig = go.Figure()
            
#             fig.add_trace(go.Box(y=df['ampere'], name='Ampere',
#                                marker_color='blue'))
#             fig.add_trace(go.Box(y=df['wattage_kwh'], name='Wattage (kWh)',
#                                marker_color='red'))
#             fig.add_trace(go.Box(y=df['pf'], name='Power Factor',
#                                marker_color='green'))
            
#             fig.update_layout(
#                 title='All Parameters Distribution',
#                 yaxis_title='Value',
#                 height=500
#             )
#     else:
#         # Single parameter plotting logic
#         st.header(f"{graph_type} of {parameter}")
        
#         if graph_type == "Line Plot":
#             fig = px.line(df, x='Timestamps', y=parameter,
#                          title=f'{parameter} Over Time')
            
#         elif graph_type == "Area Plot":
#             fig = go.Figure()
#             fig.add_trace(
#                 go.Scatter(x=df['Timestamps'], y=df[parameter],
#                           fill='tozeroy',
#                           fillcolor='rgba(0, 100, 255, 0.3)',
#                           line=dict(color='blue'),
#                           name=parameter)
#             )
#             fig.update_layout(
#                 title=f'{parameter} Over Time',
#                 xaxis_title='Timestamp',
#                 yaxis_title=parameter.capitalize(),
#                 height=500
#             )
            
#         elif graph_type == "Bar Chart":
#             fig = px.bar(df, x='Timestamps', y=parameter,
#                          title=f'{parameter} Distribution')
            
#         elif graph_type == "Scatter Plot":
#             fig = px.scatter(df, x='Timestamps', y=parameter,
#                             title=f'{parameter} Distribution')
            
#         elif graph_type == "Box Plot":
#             fig = px.box(df, y=parameter,
#                         title=f'{parameter} Distribution')

#         # Update layout
#         fig.update_layout(
#             xaxis_title="Timestamp",
#             yaxis_title=parameter.capitalize(),
#             height=500
#         )
    
#     # Display plot
#     st.plotly_chart(fig, use_container_width=True)

#     # Display basic stats
#     if parameter != "all parameters":
#         st.header(f"Statistics for {parameter}")
#         stats = calculate_stats(df, parameter)
        
#         col1, col2, col3, col4, col5 = st.columns(5)
        
#         with col1:
#             st.metric("Maximum", f"{stats['Maximum']:.2f}")
#         with col2:
#             st.metric("Minimum", f"{stats['Minimum']:.2f}")
#         with col3:
#             st.metric("Average", f"{stats['Average']:.2f}")
#         with col4:
#             st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
#         with col5:
#             st.metric("Variance", f"{stats['Variance']:.2f}")
#     else:
#         # Display stats for all parameters
#         st.header("Statistics for All Parameters")
#         parameters = ['ampere', 'wattage_kwh', 'pf']
#         for param in parameters:
#             st.subheader(param.capitalize())
#             stats = calculate_stats(df, param)
            
#             col1, col2, col3, col4, col5 = st.columns(5)
            
#             with col1:
#                 st.metric("Maximum", f"{stats['Maximum']:.2f}")
#             with col2:
#                 st.metric("Minimum", f"{stats['Minimum']:.2f}")
#             with col3:
#                 st.metric("Average", f"{stats['Average']:.2f}")
#             with col4:
#                 st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
#             with col5:
#                 st.metric("Variance", f"{stats['Variance']:.2f}")
    
#     # Display raw data with pagination
#     st.header("Raw Data")
#     page_size = 10
#     total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
#     page_number = st.number_input('Page Number', min_value=1, max_value=total_pages, value=1)
#     start_idx = (page_number - 1) * page_size
#     end_idx = min(start_idx + page_size, len(df))
#     st.dataframe(df.iloc[start_idx:end_idx])

# except Exception as e:
#     st.error(f"An error occurred: {str(e)}")
#     st.error("Please check your database connection and try again.")

# ---------------------------------------------------------------------------------------

import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import joblib

# Load the pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model(r'D:\Study\Automated-power-consumption-analysis\power_consumption\dashboard\streamlit\future_values_prediction_model.h5')
        scaler = joblib.load(r'D:\Study\Automated-power-consumption-analysis\power_consumption\dashboard\streamlit\scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Prediction function
def predict_values(input_timestamp, model, scaler, seq_length=3):
    # Database connection function
    def get_db_connection():
        return mysql.connector.connect(
            host="powersight-nikhilchadha1534-9076.c.aivencloud.com",
            user="avnadmin",
            password="AVNS_U6onSK7r0jsSIvu5DIg",
            database="defaultdb",
            port=22162
        )

    # Fetch recent historical data
    # conn = get_db_connection()
    # query = f"""
    # SELECT Timestamps, ampere, wattage_kwh, pf
    # FROM readings 
    # WHERE Timestamps < '{input_timestamp}'
    # ORDER BY Timestamps DESC
    # LIMIT {seq_length}
    # """
    # recent_data = pd.read_sql(query, conn)
    # conn.close()

    

# Load data from the CSV file
    csv_file_path = r"dashboard\streamlit\new123.csv"  # Replace with the path to your CSV file
    data = pd.read_csv(csv_file_path)

    # Convert Timestamps column to datetime (if not already in datetime format)
    data['Timestamps'] = pd.to_datetime(data['Timestamps'])

    # Filter the data
    filtered_data = data[data['Timestamps'] < input_timestamp]

    # Sort by Timestamps in descending order
    filtered_data = filtered_data.sort_values(by='Timestamps', ascending=False)

    # Limit to the specified number of rows
    recent_data = filtered_data.head(seq_length)

    # Display the result
    # print(recent_data)

    # Check if enough historical data exists
    if len(recent_data) < seq_length:
        st.warning(f"Not enough historical data. Need at least {seq_length} previous records.")
        return None

    # Prepare data for prediction
    features = ['ampere', 'wattage_kwh', 'pf']
    
    # Scale the recent data
    scaled_data = scaler.transform(recent_data[features])
    
    # Reshape for LSTM input
    input_sequence = scaled_data.reshape(1, seq_length, len(features))
    
    # Make prediction
    predicted_scaled = model.predict(input_sequence, verbose=0)[0]
    
    # Inverse transform to get actual values
    predicted_values = scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]
    
    # Create a result dictionary
    prediction_result = {
        'Timestamp': input_timestamp,
        'Predicted Ampere': predicted_values[0],
        'Predicted Wattage (kWh)': predicted_values[1],
        'Predicted Power Factor': predicted_values[2]
    }
    
    return prediction_result

# Existing Streamlit app code remains the same until the end...

# Add Prediction Section
def prediction_section():
    st.header("Power Measurements Prediction")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Could not load model. Please ensure the model files exist.")
        return
    
    # Input for prediction timestamp
    # Remove the default=datetime.now() to allow pure user selection
    prediction_date = st.date_input("Select Prediction Date")
    prediction_time = st.time_input("Select Prediction Time")
    
    # Combine date and time
    prediction_timestamp = datetime.combine(prediction_date, prediction_time)
    
    # Predict button
    if st.button("Predict Power Measurements"):
        # Call prediction function
        prediction = predict_values(prediction_timestamp, model, scaler)
        
        # Display predictions
        if prediction:
            st.subheader("Prediction Results")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Ampere", 
                          f"{prediction['Predicted Ampere']:.2f}")
            with col2:
                st.metric("Predicted Wattage (kWh)", 
                          f"{prediction['Predicted Wattage (kWh)']:.2f}")
            with col3:
                st.metric("Predicted Power Factor", 
                          f"{prediction['Predicted Power Factor']:.2f}")
            
            # Optional: Visualization of prediction
            pred_df = pd.DataFrame([prediction])
            fig = go.Figure()
            
            # Add traces for each parameter
            fig.add_trace(
                go.Bar(x=['Ampere'], y=[prediction['Predicted Ampere']], 
                       name='Ampere', marker_color='blue')
            )
            fig.add_trace(
                go.Bar(x=['Wattage (kWh)'], y=[prediction['Predicted Wattage (kWh)']], 
                       name='Wattage', marker_color='red')
            )
            fig.add_trace(
                go.Bar(x=['Power Factor'], y=[prediction['Predicted Power Factor']], 
                       name='Power Factor', marker_color='green')
            )
            
            fig.update_layout(
                title='Predicted Power Measurements',
                xaxis_title='Parameter',
                yaxis_title='Value',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Modify the main app to include prediction section
def main():
    # Existing page config and title
    st.set_page_config(page_title="Power Monitoring Dashboard", layout="wide")
    st.title("Power Monitoring Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create tabs
    tab1, tab2 = st.tabs(["Data Visualization", "Prediction"])

    with tab1:
        def get_db_connection():
            return mysql.connector.connect(
                host="powersight-nikhilchadha1534-9076.c.aivencloud.com",
                user="avnadmin",
                password="AVNS_U6onSK7r0jsSIvu5DIg",
                database="defaultdb",
                port=22162
            )

        def load_data(start_date, end_date):
            conn = get_db_connection()
            query = f"""
            SELECT * FROM readings 
            WHERE Timestamps BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY Timestamps
            """
            df = pd.read_sql(
                
                
                
                query, conn)
            conn.close()
            return df

        def calculate_stats(df, column):
            stats = {
                'Maximum': df[column].max(),
                'Minimum': df[column].min(),
                'Average': df[column].mean(),
                'Standard Deviation': df[column].std(),
                'Variance': df[column].var()
            }
            return stats

        # Set page config
       

        # Title
        st.title("Power Monitoring Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Sidebar controls
        st.sidebar.header("Controls")

        # Get today's date
        today = datetime.now().date()

        # Date range selector with today as default
        start_date = st.sidebar.date_input("Start Date", today)
        end_date = st.sidebar.date_input("End Date", today)

        # Manual refresh button
        if st.sidebar.button("Refresh Data"):
            st.rerun()

        # Graph type selector
        graph_type = st.sidebar.selectbox(
            "Select Graph Type",
            ["Line Plot", "Area Plot", "Bar Chart", "Scatter Plot", "Box Plot"]
        )

        # Parameter selector with "all parameters" option
        parameter = st.sidebar.selectbox(
            "Select Parameter",
            ["all parameters", "ampere", "wattage_kwh", "pf"]
        )

        # Load data
        try:
            # Adjust end_date to include the entire day
            end_datetime = datetime.combine(end_date, datetime.max.time())
            start_datetime = datetime.combine(start_date, datetime.min.time())
            
            df = load_data(start_datetime, end_datetime)
            
            # Create visualization based on selection
            if parameter == "all parameters":
                st.header(f"{graph_type} of All Parameters")
                
                if graph_type == "Line Plot":
                    # Create a figure with secondary y-axis
                    fig = go.Figure()
                    
                    # Add traces for each parameter
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df['ampere'], name='Ampere',
                                line=dict(color='blue'))
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
                                line=dict(color='red'))
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df['pf'], name='Power Factor',
                                line=dict(color='green'))
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='All Parameters Over Time',
                        xaxis_title='Timestamp',
                        yaxis_title='Value',
                        height=500,
                        showlegend=True
                    )
                    
                elif graph_type == "Area Plot":
                    fig = go.Figure()
                    
                    # Add traces for each parameter
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df['ampere'], name='Ampere',
                                fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)',
                                line=dict(color='blue'))
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
                                fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
                                line=dict(color='red'))
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df['pf'], name='Power Factor',
                                fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)',
                                line=dict(color='green'))
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='All Parameters Over Time',
                        xaxis_title='Timestamp',
                        yaxis_title='Value',
                        height=500,
                        showlegend=True
                    )
                    
                elif graph_type == "Bar Chart":
                    # Create subplots for bar charts
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(x=df['Timestamps'], y=df['ampere'], name='Ampere',
                                        marker_color='blue'))
                    fig.add_trace(go.Bar(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
                                        marker_color='red'))
                    fig.add_trace(go.Bar(x=df['Timestamps'], y=df['pf'], name='Power Factor',
                                        marker_color='green'))
                    
                    fig.update_layout(
                        barmode='group',
                        title='All Parameters Distribution',
                        xaxis_title='Timestamp',
                        yaxis_title='Value',
                        height=500
                    )
                    
                elif graph_type == "Scatter Plot":
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(x=df['Timestamps'], y=df['ampere'], name='Ampere',
                                        mode='markers', marker=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df['Timestamps'], y=df['wattage_kwh'], name='Wattage (kWh)',
                                        mode='markers', marker=dict(color='red')))
                    fig.add_trace(go.Scatter(x=df['Timestamps'], y=df['pf'], name='Power Factor',
                                        mode='markers', marker=dict(color='green')))
                    
                    fig.update_layout(
                        title='All Parameters Distribution',
                        xaxis_title='Timestamp',
                        yaxis_title='Value',
                        height=500
                    )
                    
                elif graph_type == "Box Plot":
                    fig = go.Figure()
                    
                    fig.add_trace(go.Box(y=df['ampere'], name='Ampere',
                                    marker_color='blue'))
                    fig.add_trace(go.Box(y=df['wattage_kwh'], name='Wattage (kWh)',
                                    marker_color='red'))
                    fig.add_trace(go.Box(y=df['pf'], name='Power Factor',
                                    marker_color='green'))
                    
                    fig.update_layout(
                        title='All Parameters Distribution',
                        yaxis_title='Value',
                        height=500
                    )
            else:
                # Single parameter plotting logic
                st.header(f"{graph_type} of {parameter}")
                
                if graph_type == "Line Plot":
                    fig = px.line(df, x='Timestamps', y=parameter,
                                title=f'{parameter} Over Time')
                    
                elif graph_type == "Area Plot":
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=df['Timestamps'], y=df[parameter],
                                fill='tozeroy',
                                fillcolor='rgba(0, 100, 255, 0.3)',
                                line=dict(color='blue'),
                                name=parameter)
                    )
                    fig.update_layout(
                        title=f'{parameter} Over Time',
                        xaxis_title='Timestamp',
                        yaxis_title=parameter.capitalize(),
                        height=500
                    )
                    
                elif graph_type == "Bar Chart":
                    fig = px.bar(df, x='Timestamps', y=parameter,
                                title=f'{parameter} Distribution')
                    
                elif graph_type == "Scatter Plot":
                    fig = px.scatter(df, x='Timestamps', y=parameter,
                                    title=f'{parameter} Distribution')
                    
                elif graph_type == "Box Plot":
                    fig = px.box(df, y=parameter,
                                title=f'{parameter} Distribution')

                # Update layout
                fig.update_layout(
                    xaxis_title="Timestamp",
                    yaxis_title=parameter.capitalize(),
                    height=500
                )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)

            # Display basic stats
            if parameter != "all parameters":
                st.header(f"Statistics for {parameter}")
                stats = calculate_stats(df, parameter)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Maximum", f"{stats['Maximum']:.2f}")
                with col2:
                    st.metric("Minimum", f"{stats['Minimum']:.2f}")
                with col3:
                    st.metric("Average", f"{stats['Average']:.2f}")
                with col4:
                    st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
                with col5:
                    st.metric("Variance", f"{stats['Variance']:.2f}")
            else:
                # Display stats for all parameters
                st.header("Statistics for All Parameters")
                parameters = ['ampere', 'wattage_kwh', 'pf']
                for param in parameters:
                    st.subheader(param.capitalize())
                    stats = calculate_stats(df, param)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Maximum", f"{stats['Maximum']:.2f}")
                    with col2:
                        st.metric("Minimum", f"{stats['Minimum']:.2f}")
                    with col3:
                        st.metric("Average", f"{stats['Average']:.2f}")
                    with col4:
                        st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
                    with col5:
                        st.metric("Variance", f"{stats['Variance']:.2f}")
            
            # Display raw data with pagination
            st.header("Raw Data")
            page_size = 10
            total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
            page_number = st.number_input('Page Number', min_value=1, max_value=total_pages, value=1)
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            st.dataframe(df.iloc[start_idx:end_idx])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your database connection and try again.")


    with tab2:
        # Prediction section
        prediction_section()

if __name__ == "__main__":
    main()