import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
from wordcloud import WordCloud
from PIL import Image
import altair as alt

# Load dataset
df = pd.read_csv(r"E:\PYTHON PROJECTS\V-S Code Projects\Netflix_Data_Analysis\netflix_dataset.csv")
df_copy = df.copy()

# Set page configuration for title and layout
st.set_page_config(page_title="Netflix Data Dashboard", layout="wide")

# Load Netflix logo
logo = Image.open(r"E:\PYTHON PROJECTS\V-S Code Projects\Netflix_Data_Analysis\pngwing.com.png")

# Sidebar: Netflix logo
st.sidebar.image(logo, use_column_width=True)

# Sidebar: Add a title with Netflix Red color
st.sidebar.markdown("<h2 style='color: #E50914;'>Explore Netflix Data</h2>", unsafe_allow_html=True)

# Sidebar: Add a button or interaction element in Netflix red
if st.sidebar.button('Analyze Content'):
    st.sidebar.write("Analysis started!")

# Main Title
st.markdown("<h1 style='text-align: center; color: #E50914;'>Netflix Data Dashboard</h1>", unsafe_allow_html=True)

# Divider line with custom color
st.markdown("<hr style='border: 2px solid #E50914;'>", unsafe_allow_html=True)

# Sample Text Section in Netflix Theme
st.markdown("""
    <h3 style='color: #FFFFFF;'>Key Insights</h3>
    <ul style='color: #B3B3B3;'>
        <li>Movies dominate the Netflix library.</li>
        <li>Top genres include Drama, Comedy, and Documentary.</li>
        <li>The US, India, and the UK contribute the most content.</li>
    </ul>
""", unsafe_allow_html=True)

# Custom CSS for dark background and specific text colors
st.markdown(
    """
    <style>
    /* Set sidebar background color */
    .css-1d391kg {  
        background-color: #141414; /* Dark background */
        color: white; /* Text color */
    }
    /* Set main background color */
    .css-1outpf7 {  
        background-color: #000000; /* Black background */
        color: white; /* Text color */
    }
    /* All body text */
    body {
        color: #FFFFFF; /* White text color */
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF; /* Header text color */
    }
    /* List item text color */
    li {
        color: #B3B3B3; /* Light gray for list items */
    }
    /* Plot background */
    .plotly .plot {
        background-color: #000000; /* Dark background for plots */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your Streamlit app content goes here
content_type = st.sidebar.selectbox("Select Content Type:", ["Movies", "TV Shows"])
selected_plot_type = st.sidebar.selectbox("Select Plot Type:", ["Top Genres", "Content Type Distribution", "Country"])

# Data preparation
show_type = df_copy["type"].value_counts()
top_genres = df_copy["listed_in"].str.split(',').explode().value_counts().head(10)
country = df_copy["country"].value_counts()

# Sample data preparation for country distribution
country_data = {
    "Country": country.index[:10],
    "Percentage": country.values[:10]
}
df_country = pd.DataFrame(country_data)

# Function to get latitude and longitude
def get_latitude(country):
    latitudes = {
        "United States": 37.0902,
        "India": 20.5937,
        "United Kingdom": 55.3781,
        "Pakistan": 30.3753,
        "Japan": 36.2048,
        "Mexico": 23.6345,
        "Canada": 56.1304,
        "Spain": 40.4637,
    }
    return latitudes.get(country, 0)

def get_longitude(country):
    longitudes = {
        "United States": -95.7129,
        "India": 78.9629,
        "United Kingdom": -3.4360,
        "Pakistan": 69.3451,
        "Japan": 138.2529,
        "Mexico": -102.5528,
        "Canada": -106.3468,
        "Spain": -3.7038,
    }
    return longitudes.get(country, 0)

if content_type == "Movies":
    df_filtered = df_copy[df_copy['type'] == 'Movie']
else:
    df_filtered = df_copy[df_copy['type'] == 'TV Show']

if selected_plot_type == "Top Genres":
    top_genres = df_filtered['listed_in'].str.split(',').explode().value_counts().head(10)

    st.subheader(f"Top Genres for {content_type}")
    df_genres = pd.DataFrame({'Genre': top_genres.index, 'Frequency': top_genres.values})

    fig_top_genres = alt.Chart(df_genres).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Frequency'),
        y=alt.Y('Genre:N', title='Genres', sort='-x'),
        color='Frequency:Q',
        tooltip=['Genre', 'Frequency']
    ).properties(
        title=f'Top Genres for {content_type}',
        width=700,
        height=400
    )

    st.altair_chart(fig_top_genres, use_container_width=True)

elif selected_plot_type == "Content Type Distribution":
    show_type = df_copy["type"].value_counts()

    st.title("Netflix Style Dashboard")
    st.write("This is a sample dashboard with a Netflix-style theme.")

    col1, col2 = st.columns(2)

    with col1:
        df_content_type = pd.DataFrame({'Type': show_type.index, 'Frequency': show_type.values})
        fig_content_type = alt.Chart(df_content_type).mark_bar().encode(
            x=alt.X('Type:N', title='Type'),
            y=alt.Y('Frequency:Q', title='Frequency'),
            color='Frequency:Q',
            tooltip=['Type', 'Frequency']
        ).properties(
            title='Distribution of Type on Netflix',
            width=400,
            height=300
        ).configure_title(
            fontSize=16,
            anchor='middle',
            font='Arial'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=12
        )
        
        st.altair_chart(fig_content_type, use_container_width=True)

# Create a row with two columns for country plots
col3, col4 = st.columns(2)

# Only show country plots if the user selects "Country" in the sidebar
if selected_plot_type == "Country":
    # Bar Plot for Top 10 Countries using Plotly
    with col3:
        fig_country = px.bar(
            df_country,
            x="Country",
            y="Percentage",
            title="Top 10 Countries on Netflix",
            labels={"Country": "Countries", "Percentage": "Frequency"},
            color="Percentage",
            color_continuous_scale='Reds'
        )
        fig_country.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            xaxis_title='Countries',
            yaxis_title='Frequency',
            title_font=dict(size=16, family='Arial, sans-serif', color='black'),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_country)

    # Pie Chart for Top 10 Countries using Plotly
    with col4:
        fig_pie = px.pie(
            df_country,
            names='Country',
            values='Percentage',
            title="Top 10 Countries Distribution",
            color='Country',
            color_discrete_sequence=px.colors.sequential.Reds
        )
        fig_pie.update_layout(
            plot_bgcolor='red',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(size=16, family='Arial, sans-serif', color='black')
        )
        st.plotly_chart(fig_pie)

# Interactive Map for Countries with Pie Chart
st.subheader(f"Map of Netflix {content_type} Content by Country")
map_col, pie_col = st.columns([2, 1])  # Adjust column sizes as needed

with map_col:
    m = folium.Map(location=[20.0, 0.0], zoom_start=2, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(m)

    for index, row in df_country.iterrows():
        folium.Marker(
            location=[get_latitude(row['Country']), get_longitude(row['Country'])],
            popup=f"<strong>{row['Country']}</strong><br>Percentage: {row['Percentage']}%",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    st_folium(m, width=700, height=500)  # Adjusted width for better fit

# Count plot for Ratings using Plotly
rating = df_copy['rating'].value_counts().reset_index()
rating.columns = ['Rating', 'Frequency']

fig_count = px.bar(
    rating,
    x='Rating',
    y='Frequency',
    title="Rating Distribution of Netflix Content",
    color='Frequency',
    color_continuous_scale=px.colors.sequential.Reds
)
fig_count.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    title_font=dict(size=16, family='Arial, sans-serif', color='black')
)

st.plotly_chart(fig_count)
