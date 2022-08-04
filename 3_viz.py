
# load in libraries
import dash
from dash import Dash, html
# from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

modeling_df = pd.read_csv('./modeling_df.csv')


# Elements to include: 

# world map
# country dropdown menu
# timeframe dropdown menu
# ner
# sentiment analysis
# topic modeling
# global regional analysis 



app = Dash(__name__, use_pages=True)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'}

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

fig = px.scatter(df, x="gdp per capita", y="life expectancy",
                 size="population", color="continent", hover_name="country",
                 log_x=True, size_max=60)

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div([
    dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    