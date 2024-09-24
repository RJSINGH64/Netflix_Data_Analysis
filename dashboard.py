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
import os
import pickle
#from model_training import initiate_model_training
import sys,os
# Load dataset with a relative path
file_path = os.path.join(os.getcwd(), 'netflix_dataset.csv')
df = pd.read_csv(file_path)
df_copy = df.copy()

#try:
    #initiate_model_training() #triggering tarining pipeline

#except Exception as e:
   # print(e)


# Set page configuration for title and layout
st.set_page_config(
    page_title="Netflix Data Visualization Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .st-folium {
        margin-bottom: 0;  /* Remove margin below the map */
    }
    </style>
    """,
    unsafe_allow_html=True
)

image_path = os.path.join(os.getcwd(), 'pngwing.com.png')
# Load Netflix logo
logo = Image.open(image_path)

# Sidebar: Netflix logo
st.sidebar.image(logo, use_column_width=True)

# Sidebar: Add a title with Netflix Red color
st.sidebar.markdown(
    "<h2 style='color: #E50914;'>Explore Netflix Data</h2>", unsafe_allow_html=True)

# Sidebar: Add a button or interaction element in Netflix red
if st.sidebar.button('Analyze Content'):
    st.sidebar.write("Analysis started!")

# Main Title
st.markdown("<h1 style='text-align: center; color: #E50914;'>Netflix Data Dashboard</h1>",
            unsafe_allow_html=True)

# Divider line with custom color
st.markdown("<hr style='border: 2px solid #E50914;'>", unsafe_allow_html=True)

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


# Sidebar option to show the dataset
show_data = st.sidebar.checkbox("Show Dataset")
st.markdown("""
    <h3 style='color: #FFFFFF;'>Key Insights</h3>
    <ul style='color: #B3B3B3;'>
        <li>Movies dominate the Netflix library.</li>
        <li>Top genres include Drama, Comedy, and Documentary.</li>
        <li>The US, India, and the UK contribute the most content.</li>
        <li>Asian countries like South Korea appear in this list, reflecting the global rise in popularity of Korean TV shows (K-dramas)..</li>
        <li>The focus on international shows is visible, with countries like Japan and India producing many TV shows for Netflix’s global audience..</li>
        <li>Documentaries and Crime TV Shows follow closely, indicating a growing trend in true crime and investigative series.</li>
           
    </ul>
""", unsafe_allow_html=True)


# Display the dataset if the checkbox is selected
if show_data:
    # Custom title with red color and bold
    st.markdown("<h2 style='color: #E50914; font-weight: bold;'>Netflix Dataset</h2>", unsafe_allow_html=True)
    st.dataframe(df_copy)  # Display the entire DataFrame

# Load the trained model, scaler, and encoders
with open("trained_model.pkl", 'rb') as file:
    model_rf = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("label_encoder.pkl", 'rb') as file:
    label_encoders = pickle.load(file)

# Load feature names
with open("feature_names.pkl", 'rb') as file:
    feature_names = pickle.load(file)


# Load the original dataset to get unique values for dropdowns
df = pd.read_csv("netflix_dataset.csv")

# Unique values for dropdowns
countries = df["country"].unique().tolist()
ratings = df["rating"].unique().tolist()
directors = df["director"].unique().tolist()
genres = df['listed_in'].str.get_dummies(sep=', ').columns.tolist()




# Input fields for the prediction

st.markdown("<h2 style='color: #E50914; font-weight: bold;'>Netflix Show type Prediction</h2>", unsafe_allow_html=True)

# Dropdowns for inputs
selected_country = st.selectbox("Select Country", countries)
selected_rating = st.selectbox("Select Rating", ratings)
selected_director = st.selectbox("Select Director", directors)

# Genre selection as multiple choice (can select multiple genres)
selected_genres = st.multiselect("Select Genres", genres)

# Duration input
duration = st.number_input("Duration (minutes)", min_value=0, value=30)

# Prepare input data for the model
if st.button("Predict"):
    # Prepare the input data as a dictionary
    input_data = {
        "country": label_encoders["country"].transform([selected_country])[0],
        "rating": label_encoders["rating"].transform([selected_rating])[0],
        "duration": duration,
    }

    # One-hot encode the selected genres
    for genre in genres:
        input_data[genre] = 1 if genre in selected_genres else 0

    # One-hot encode the selected director
    for director in directors:
        input_data[f"director_{director}"] = 1 if director == selected_director else 0

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Ensure all feature names are included, filling missing ones with 0
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
            
    # Reorder the DataFrame to match the training feature order
    input_df = input_df[feature_names]

    # Standardize the features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model_rf.predict(input_scaled)
    
    # Decode the prediction
    prediction_label = label_encoders["type"].inverse_transform(prediction)[0]

    # Display the result
    st.success(f"The predicted type is: **{prediction_label}**")

# Your Streamlit app content goes here
content_type = st.sidebar.selectbox(
    "Select Content Type:", ["Movies", "TV Shows"])
selected_plot_type = st.sidebar.selectbox(
    "Select Plot Type:", ["Top Genres", "Content Type Distribution", "Country"])

# Data preparation
show_type = df_copy["type"].value_counts()
top_genres = df_copy["listed_in"].str.split(
    ',').explode().value_counts().head(10)
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
    top_genres = df_filtered['listed_in'].str.split(
        ',').explode().value_counts().head(10)

    st.subheader(f"Top Genres for {content_type}")
    df_genres = pd.DataFrame(
        {'Genre': top_genres.index, 'Frequency': top_genres.values})

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
        df_content_type = pd.DataFrame(
            {'Type': show_type.index, 'Frequency': show_type.values})
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
            title_font=dict(
                size=16, family='Arial, sans-serif', color='black'),
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

st.markdown("<h2 style='color: black; font-weight: bold;'>Top 10 Countries </h2>", unsafe_allow_html=True)
# Interactive Map for Countries with Pie Chart
st.subheader(f"Map of Netflix {content_type} Content by Country")
map_col = st.columns(1)  # Adjust column sizes as needed

with map_col[0]:
    m = folium.Map(location=[20.0, 0.0], zoom_start=2,
                   tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(m)

    for index, row in df_country.iterrows():
        folium.Marker(
            location=[get_latitude(row['Country']),
                      get_longitude(row['Country'])],
            popup=f"<strong>{row['Country']}</strong><br>Percentage: {row['Percentage']}%",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    st_folium(m, width=900, height=400)  # Adjusted width for better fit

# Count plot for Ratings using Plotly
rating = df_copy['rating'].value_counts().reset_index()
rating.columns = ['Rating', 'Frequency']

fig_count = px.bar(
    rating,
    x='Rating',
    y='Frequency',
    color='Frequency',
    title="Movies and Shows Rating",
    color_continuous_scale=px.colors.sequential.Blues,
    labels={"Rating": "Rating on Netflix", "Frequency": "Frequency"}
)

fig_count.update_layout(
    xaxis_title="Rating on Netflix",
    yaxis_title="Frequency",
    title_font=dict(size=20, color='black'),
    xaxis_tickangle=-45
)

# Pie chart for Ratings distribution using Plotly
fig_pie = px.pie(
    rating,
    names='Rating',
    values='Frequency',
    title="Top Ratings Distribution",
    color='Rating',
    color_discrete_sequence=px.colors.sequential.Plasma
)

# Show plots in columns
st.subheader("Ratings Distribution")
col5, col6 = st.columns(2)

with col5:
    st.plotly_chart(fig_count)

with col6:
    st.plotly_chart(fig_pie)


# Function to redirect to a goodbye message

def redirect_to_goodbye():
    st.write("Thank you for using the app! You can close this tab now.")
    st.stop()  # Stops further execution of the script


# Custom exit button in sidebar
if st.sidebar.button("Exit Dashboard"):
    redirect_to_goodbye()

# Sidebar option for selecting the number of years to display
num_years = st.sidebar.slider(
    "Select Number of Release Years to Display:", min_value=1, max_value=20, value=11)

# Checking most releases by year
release_year = df_copy["release_year"].value_counts().head(num_years)

# Create a Plotly bar chart
fig = px.bar(
    x=release_year.index,
    y=release_year.values,
    labels={'x': 'Release Year', 'y': 'Frequency of Release Year'},
    title='Most Number of Releases by Year',
    color=release_year.index,
    color_continuous_scale='Viridis'
)

# Update layout for better aesthetics
fig.update_layout(
    xaxis_title='Release Year',
    yaxis_title='Frequency of Release Year',
    title_font=dict(size=20, family='Arial, sans-serif', color='black'),
    xaxis_tickangle=-45,
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Sidebar option for selecting the number of countries to display
num_countries = st.sidebar.slider(
    "Select Number of Countries to Display:", min_value=1, max_value=20, value=10)

# Counting releases by country
country_release_count = df_copy['country'].value_counts().head(num_countries)

# Create a Plotly bar chart for releases by country
fig_country = px.bar(
    x=country_release_count.index,
    y=country_release_count.values,
    labels={'x': 'Country', 'y': 'Number of Releases'},
    title='Number of Releases by Country',
    color=country_release_count.index,
    color_continuous_scale='Blues'
)

# Update layout for better aesthetics
fig_country.update_layout(
    xaxis_title='Country',
    yaxis_title='Number of Releases',
    title_font=dict(size=20, family='Arial, sans-serif', color='black'),
    xaxis_tickangle=-45,
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig_country)

# Optionally, you can create a line plot for releases over time by country
# You can choose a specific country from the unique values in the country column
selected_country = st.sidebar.selectbox(
    "Select a Country:", df_copy['country'].unique())

# Filter the DataFrame for the selected country
country_data = df_copy[df_copy['country'] == selected_country]
release_year_country = country_data['release_year'].value_counts().sort_index()

# Create a Plotly line chart for the selected country's releases over the years
fig_line = px.line(
    x=release_year_country.index,
    y=release_year_country.values,
    labels={'x': 'Release Year', 'y': 'Number of Releases'},
    title=f'Number of Releases in {selected_country} by Year'
)

# Update layout for better aesthetics
fig_line.update_layout(
    xaxis_title='Release Year',
    yaxis_title='Number of Releases',
    title_font=dict(size=20, family='Arial, sans-serif', color='black'),
)

# Display the line chart
st.plotly_chart(fig_line)

# Sidebar option for country selection
selected_country = st.sidebar.selectbox(
    "select country for top Directors:", df_copy["country"].unique())

# Filter the dataframe based on the selected country
filtered_data = df_copy[df_copy["country"] == selected_country]

# Director Data Preparation
director = filtered_data["director"].value_counts().reset_index()
director.columns = ['Director', 'Count']  # Rename columns for clarity
director = director.sort_values(
    by="Count", ascending=False).iloc[1:20]  # Top 10 directors

# Create a Plotly bar chart for top directors
fig_director = px.bar(
    director,
    x='Director',
    y='Count',
    labels={'Director': 'Director Name', 'Count': 'Count'},
    title=f'Top 10 Directors on Netflix in {selected_country}',
    color='Count',
    color_continuous_scale='Viridis'
)

# Update layout for better aesthetics
fig_director.update_layout(
    xaxis_title='Director Name',
    yaxis_title='Count',
    title_font=dict(size=20, family='Arial, sans-serif', color='black'),
    xaxis_tickangle=-45,
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig_director)


# Data Preparation
movies_director = df_copy[df_copy["type"] == "Movie"]["director"]
tv_show_director = df_copy[df_copy["type"] == "TV Show"]["director"]

# Sidebar option to select between Movie and TV Show
selected_type = st.sidebar.selectbox(
    "Select Type for Wordcloud:", ["Movie", "TV Show"])


# Assuming you have already defined movies_director and tv_show_director
movies_director = df_copy[df_copy['type'] ==
                          'Movie']['director'].dropna().tolist()
tv_show_director = df_copy[df_copy['type'] ==
                           'TV Show']['director'].dropna().tolist()

if selected_type == "Movie":
    wordcloud = WordCloud(width=800, height=400, background_color='black',
                          colormap='viridis').generate(' '.join(movies_director))
    st.header("WordCloud of Movie Directors")
else:
    wordcloud = WordCloud(width=800, height=400, background_color='black',
                          colormap='viridis').generate(' '.join(tv_show_director))
    st.header("WordCloud of TV Show Directors")

st.markdown("<h2 style='color:black; font-weight: bold;'>Directors Wordcloud</h2>", unsafe_allow_html=True)
# Plot WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)
# Footer with custom message
st.markdown("<h4 style='text-align: center; color: #B3B3B3;'>Created by Rajat Singh</h4>",
            unsafe_allow_html=True)
