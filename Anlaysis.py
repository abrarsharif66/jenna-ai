import streamlit as st
import pandas as pd
from datetime import datetime

# Function to get the current month and year
def get_current_month_year():
    current_month = datetime.now().strftime("%B_%Y")
    return current_month

# Read the CSV file
current_month_year = get_current_month_year()
filename = f"{current_month_year}.csv"

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    st.warning(f"No data found for {current_month_year}.")
    st.stop()

# Title and description
st.title("Sentiment and Domain Analysis")
st.write("This app displays the sentiment and domain analysis from the conversation.")

# Display the data
st.dataframe(df)

# Count plots
st.subheader("Sentiment Distribution")
sentiment_count = df["Sentiment"].value_counts()
st.bar_chart(sentiment_count)

st.subheader("Domain Distribution")
domain_count = df["Domain"].value_counts()
st.bar_chart(domain_count)