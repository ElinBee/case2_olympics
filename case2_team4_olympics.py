import kaggle
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


api = KaggleApi()
api.authenticate()
api.dataset_download_file("stefanydeoliveira/summer-olympics-medals-1896-2024",
                         file_name='olympics_dataset.csv')

with zipfile.ZipFile('olympics_dataset.csv.zip','r') as zipref:
    zipref.extractall()

    
df = pd.read_csv('olympics_dataset.csv')
df2 = pd.read_csv('olympics_dataset.csv')

landenlijst = [
    "Algeria", "Andorra", "Argentina", "Armenia", "Australia", "Azerbaijan", "Belgium", "Bermuda", 
    "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Canada", "Chile", "China", "Chinese Taipei (Taiwan)", 
    "Colombia", "Cyprus", "Denmark", "Germany", "Estonia", "Ethiopia", "Finland", "France", 
    "Georgia", "Ghana", "Greece", "United Kingdom", "Hungary", "Hong Kong", "Ireland", "Iceland", 
    "India", "Iran", "Israel", "Italy", "Jamaica", "Japan", "Cayman Islands", "Kazakhstan", "Kyrgyzstan", 
    "Croatia", "Latvia", "Lebanon", "Liechtenstein", "Lithuania", "Macedonia", "Morocco", "Mexico", "Moldova", 
    "Monaco", "Mongolia", "Montenegro", "Netherlands", "Nepal", "New Zealand", "North Korea", "Norway", 
    "Ukraine", "Uzbekistan", "Austria", "Pakistan", "Peru", "Poland", "Portugal", "Romania", "Russia", 
    "San Marino", "Senegal", "Singapore", "Slovenia", "Slovakia", "Spain", "Tajikistan", "Czech Republic", 
    "Turkey", "United States", "Belarus", "South Africa", "South Korea", "Sweden", "Switzerland","Soviet Union"
]

df = df[df['Team'].isin(landenlijst)]
df = df.drop_duplicates(subset=['Event', 'Medal', 'Sport', 'Year'], keep='first').reset_index(drop=True)


#---Streamlit interface---------------------------------------------------------------------------------------------

st.title("128 jaar van de :red[Olympische Spelen] \n:first_place_medal: :second_place_medal: :third_place_medal:")
tab1, tab2, tab3, tab4 = st.tabs(["Voorspelling", "Sporten", "Atleten", "Medailles"])


#---Voorspellingsgrafiek--------------------------------------------------------------------------------------------

teams = df['Team'].unique()
teams.sort()

#dropdown
with tab1:
    selected_team = st.selectbox('Selecteer een land', options=teams)

country = df[df['Team'] == selected_team]     # filter op gekozen land in dropdown

medal_list = ['Gold', 'Silver', 'Bronze']
filtered_prediction_df = country[country['Medal'].isin(medal_list)]
medal_count_reg = filtered_prediction_df.groupby('Year').size().reset_index(name='Total')
medal_count_reg = medal_count_reg.set_index('Year')

# Lineaire regressie
X = medal_count_reg.index.values.reshape(-1, 1) 

last_year = 2024
next_10_years = np.arange(last_year + 4, last_year + 41, 4).reshape(-1, 1)  # 2028, 2032, 2036, ..., 2064
y = medal_count_reg['Total'].values
model = LinearRegression().fit(X, y)
predicted_total = model.predict(next_10_years)
predicted_total = np.round(predicted_total).astype(int)     # afronden

# Maak een DataFrame met de voorspelde gegevens
predicted_data = pd.DataFrame({
    'Year': next_10_years.flatten(),
    'predicted_total': predicted_total
})

trendline_values = model.predict(X)
trendline_total = np.round(trendline_values).astype(int)

# Maak een DataFrame met de trendlijn voor historische data
trendline_df = pd.DataFrame({
    'Year': medal_count_reg.index,
    'Trendline': trendline_total
})

historical_data = medal_count_reg.reset_index().rename(columns={'Total': 'Total Medailles'})
last_historical_year = historical_data['Year'].max()  # Dit zou 2024 moeten zijn

# Maak een nieuwe voorspelling dataset die begint bij 2024
predicted_data = pd.DataFrame({
    'Year': np.insert(next_10_years.flatten(), 0, last_historical_year),
    'Trendline': np.insert(predicted_total, 0, trendline_total[-1])  # Voorspelling start bij de laatste waarde van de historische trendlijn
})

line_fig = go.Figure()

# lijnen in grafiek toevoegen
line_fig.add_trace(go.Scatter(
    x=historical_data['Year'],
    y=historical_data['Total Medailles'],
    mode='lines+markers',
    name='Aantal behaalde medailles',
    line=dict(color='blue', width=2),
))
line_fig.add_trace(go.Scatter(
    x=historical_data['Year'],
    y=trendline_total,
    mode='lines',
    name='Trendlijn behaalde medailles',
    line=dict(color='lightblue', dash='5px, 2px', width=2)
))
line_fig.add_trace(go.Scatter(
    x=predicted_data['Year'],
    y=predicted_data['Trendline'],
    mode='lines',
    name='Voorspelling medailles',
    line=dict(color='orange', dash='5px, 2px', width=2)
))
line_fig.add_trace(go.Scatter(
    x=[last_historical_year], 
    y=[trendline_total[-1]],   
    mode='markers',
    name='Start voorspelling',
    marker=dict(color='orange', size=8, symbol='circle')
))

olympic_years = list(range(1896, 2064 + 1, 12))

line_fig.update_layout(
    title=f'Voorspelling aantal medailles voor {selected_team}',
    xaxis_title='Jaar',
    yaxis_title='Totaal aantal medailles',
    xaxis=dict(
        tickmode='array',
        tickvals=olympic_years,
        ticktext=[str(year) for year in olympic_years],
        tickangle=-45,
        tickfont=dict(size=10)
    ),
    autosize=False,
    width=1400,
    height=500,
    margin=dict(l=80, r=80, b=100, t=50),
)

with tab1:
    st.plotly_chart(line_fig)


#---Aantal atleten en aantal sporten----------------------------------------------------------------------------------

country = df2[df2['Team'] == selected_team]
unique_athletes_per_year = country.groupby('Year')['Name'].nunique().reset_index(name='Unique Athletes')

line_fig_athletes = go.Figure()

line_fig_athletes.add_trace(go.Scatter(
    x=unique_athletes_per_year['Year'],
    y=unique_athletes_per_year['Unique Athletes'],
    mode='lines+markers',
    name='Aantal Unieke Atleten',
    line=dict(color='green', width=2)
))
line_fig_athletes.update_layout(
    title=f'Aantal Unieke Atleten per Jaar voor {selected_team}',
    xaxis_title='Jaar',
    yaxis_title='Aantal atleten',
    xaxis=dict(
        tickmode='array',
        tickvals=olympic_years,
        ticktext=[str(year) for year in olympic_years],
        tickangle=-45,
        tickfont=dict(size=10)
    )
)

unique_events_per_year = df2.groupby('Year')['Event'].nunique().reset_index(name='Unique events')
line_fig_events = go.Figure()

line_fig_events.add_trace(go.Scatter(
    x=unique_events_per_year['Year'],
    y=unique_events_per_year['Unique events'],
    mode='lines+markers',
    name='Aantal unieke sport events',
    line=dict(color='green', width=2)
))
line_fig_events.update_layout(
    title=f'Aantal sport events per Jaar',
    xaxis_title='Jaar',
    yaxis_title='Aantal sport events',
    xaxis=dict(
        tickmode='array',
        tickvals=olympic_years,
        ticktext=[str(year) for year in olympic_years],
        tickangle=-45,
        tickfont=dict(size=10)
    )
)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(line_fig_athletes)
    with col2:
        st.plotly_chart(line_fig_events)


#---Pie Chart---------------------------------------------------------------------------------------------------------

sports = df['Sport'].unique()
sports.sort()

# Dropdown voor sportselectie
with tab2:
    selected_sport = st.selectbox('Selecteer een Sport:', sports)

# Filter op geselecteerde sport, tel het aantal medailles per land en sorteer
medals_per_country = df[df['Sport'] == selected_sport].groupby('Team').size().reset_index(name='Total Medals')
top_5_countries = medals_per_country.sort_values(by='Total Medals', ascending=False).head(5)

# Maak een pie chart met Plotly Express
fig = px.pie(top_5_countries, 
              names='Team', 
              values='Total Medals', 
              title=f'Top 5 Landen met meeste medailles in {selected_sport}',
              hover_data=['Total Medals'], 
              color='Total Medals',
              hole=0.5)

# Voeg een hovertemplate toe om alleen het percentage te tonen
fig.update_traces(
    textposition='inside',  # Tekst binnen de chart
    textinfo='label+value',  # Toon label en waarde binnenin de chart
    hovertemplate='%{percent}',  # Alleen het percentage weergeven bij hover
)
with tab2:
    st.plotly_chart(fig)


#---Atleten-----------------------------------------------------------------------------------------------------------

with tab3:
    gender_option = st.selectbox(
        'Selecteer geslacht:',
        ('Alleen mannen', 'Alleen vrouwen', 'Beide geslachten')
    )

    # Checkboxes voor goud, zilver, en brons
    goud = st.checkbox('Goud', value=True)
    zilver = st.checkbox('Zilver', value=True)
    brons = st.checkbox('Brons', value=True)

# Functie om medailles te tellen per atleet
def tel_medailles(df2, medal_types):
    return df2[df2['Medal'].isin(medal_types)].groupby(['Name', 'Sport']).size().reset_index(name='Aantal Medailles')

geselecteerde_medailles = []

if goud:
    geselecteerde_medailles.append('Gold')
if zilver:
    geselecteerde_medailles.append('Silver')
if brons:
    geselecteerde_medailles.append('Bronze')

# Alleen doorgaan als er minstens één medailletype is geselecteerd
if geselecteerde_medailles:
    # Filteren op geslacht
    if gender_option == 'Alleen mannen':
        df2 = df2[df2['Sex'] == 'M']  # Filter alleen mannen
    elif gender_option == 'Alleen vrouwen':
        df2 = df2[df2['Sex'] == 'F']  # Filter alleen vrouwen

    # Filteren op basis van geselecteerde medailletypes en het aantal medailles per atleet tellen
    df_geselecteerde_medailles = tel_medailles(df2, geselecteerde_medailles)

    # Bepaal de top 10 atleten op basis van de gefilterde data
    top_10_athletes = df_geselecteerde_medailles.groupby('Name')['Aantal Medailles'].sum().sort_values(ascending=False).head(10)

    fig = go.Figure()
    kleuren = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}

    # Voor elk medaille type een trace toevoegen aan de bar chart
    for medal_type in geselecteerde_medailles:
        df_medal_type = df2[df2['Medal'] == medal_type]

        # Groepeer op naam en sport en tel het aantal medailles
        df_grouped = df_medal_type.groupby(['Name', 'Sport']).size().reset_index(name='Aantal Medailles')

        # Filter alleen de top 10 atleten
        df_grouped = df_grouped[df_grouped['Name'].isin(top_10_athletes.index)]
        
        sporten_per_atleet = df_grouped.groupby('Name')['Sport'].apply(lambda x: ', '.join(x.unique())).reindex(top_10_athletes.index, fill_value='')

        # Bereken de totale medailles per atleet om de volgorde correct te houden
        total_medals = df_grouped.groupby('Name')['Aantal Medailles'].sum().reindex(top_10_athletes.index, fill_value=0)

        # Voeg een trace toe voor de bar chart
        fig.add_trace(go.Bar(
            x=total_medals.index,
            y=total_medals.values,
            name='Brons' if medal_type == 'Bronze' else 'Zilver' if medal_type == 'Silver' else 'Goud',  # Medaille naam in het Nederlands
            marker_color=kleuren[medal_type],
            hovertemplate='<b>%{x}</b><br>Aantal: %{y}<br>Sport: %{customdata}',
            customdata= sporten_per_atleet  # Sport toevoegen aan customdata voor hover
        ))

    fig.update_layout(
        barmode='stack',  # Stacked bar chart
        title='Top 10 Atleten - Aantal Medailles',
        xaxis_title='Atleet',
        yaxis_title='Aantal Medailles',
        xaxis_tickangle=-45,  # Namen roteren voor betere leesbaarheid
        legend_title_text='Medaille Type',  # Aangepaste titel voor de legenda
        width = 1400,
        height = 500,
        font=dict(
            family="Arial, sans-serif",  # Consistent lettertype
            size=12,
            color="black"
        )
    )
 
    with tab3:
        st.plotly_chart(fig)
else:
    st.write("Selecteer minstens één medaille om de grafiek te zien.")


#---Top 10 landen medailles-------------------------------------------------------------------------------------------

jaar = df.groupby("Year")["Medal"].count().reset_index()
jaar.sort_values(by = "Year", ascending= True)

years_with_medals = df['Year'].unique().tolist()
years_with_medals.sort()

with tab4:
    teams = df['Team'].unique()
    selected_team1 = st.selectbox('Selecteer een land', options=teams)   # Dropdown
    country1 = df[df['Team'] == selected_team1]  
    
    medal_count = country1['Medal'].value_counts()
    gold_medals = medal_count.get('Gold', 0)
    silver_medals = medal_count.get('Silver', 0)
    bronze_medals = medal_count.get('Bronze', 0)

# Function to create a styled box
def styled_box(title, count, emoji):
    return f"""
    <div style="border: 1px solid #ccc; border-radius: 5px; padding: 10px; text-align: center; background-color: #f9f9f9;">
        <h3 style="margin: 0;">{emoji} {title}</h3>
        <p style="font-size: 20px; font-weight: bold;">{count}</p>
    </div>
    """
# Display medal counts in styled boxes
with tab4:
    st.write(f"### Medailleverdeling voor {selected_team1}:")
    col11,col12,col13= st.columns(3)
    with col11:
        col11.markdown(styled_box("Goud", gold_medals, "🥇"), unsafe_allow_html=True)
    with col12:
        col12.markdown(styled_box("Zilver", silver_medals, "🥈"), unsafe_allow_html=True)
    with col13:
        col13.markdown(styled_box("Brons", bronze_medals, "🥉"), unsafe_allow_html=True)
    for i in range(3): st.text(" ")
    st.write('Kies een jaar om de top 10 landen met de meeste Olympische medailles te zien.')
    

# Slider to select a year
with tab4:
    year = st.select_slider('Year', options=years_with_medals , value=years_with_medals[0])
    df_year = df[df['Year'] == year]
    medals_by_country = df_year.groupby(['Team', 'Medal']).size().unstack(fill_value=0)
    for medal in ['Gold', 'Silver', 'Bronze']:
        if medal not in medals_by_country.columns:
            medals_by_country[medal] = 0

# Calculate total medals and get top 10 countries
medals_by_country['Totaal'] = medals_by_country[['Gold', 'Silver', 'Bronze']].sum(axis=1)
top_countries = medals_by_country.sort_values(by='Totaal', ascending=False).head(10)

# Reset index for Plotly
top_countries = top_countries.reset_index()
top_countries = top_countries.rename(columns={'Gold': 'Goud', 'Silver': 'Zilver', 'Bronze': 'Brons'})

top_countries.index = top_countries.index + 1
# Melt the dataframe for Plotly
top_countries_melted = top_countries.melt(id_vars=['Team', 'Totaal'], value_vars=['Goud', 'Zilver', 'Brons'], var_name='Medal', value_name='Count')

# Create a stacked bar plot using Plotly
fig = px.bar(top_countries_melted, x='Team', y='Count', color='Medal', 
             color_discrete_map={'Goud': '#FFD700', 'Zilver': '#C0C0C0', 'Brons': '#CD7F32'},
             title=f'Top 10 landen op basis van totale medailles in {year}',
             labels={'Count': 'Aantal Medailles', 'Team': 'Land'},
             category_orders={'Medal': ['Goud', 'Zilver', 'Brons']})

fig.update_layout(barmode='stack',
                  width = 700
                  )

with tab4:
    col15,col16 = st.columns([4,2])
    with col15:
        st.plotly_chart(fig)
    with col16:
        st.write(top_countries[['Team', 'Goud', 'Zilver', 'Brons', 'Totaal']])