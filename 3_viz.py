# load in libraries
import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

# from dash.dependencies import Input, Output
# import plotly.express as px


# Elements to include:

# world map
# country dropdown menu
# timeframe dropdown menu
# ner
# sentiment analysis
# topic modeling
# global regional analysis


app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG])


app.layout = html.Div([dash.page_container])

if __name__ == "__main__":
    app.run_server(debug=True)
