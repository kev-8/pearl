
# load in libraries
import dash
from dash import Dash, html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output
# import plotly.express as px
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


df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')


app.layout = html.Div([
    dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    