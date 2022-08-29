import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import html, dcc

dash.register_page(__name__, path='/app', title='pearl', name='app_page')

# TODO: 
    # use dbc to align the dropdown menus
    # decrease map space -> top half 30%, bottom half 70%
    # load in df and create filters based on menu selection

url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '

colors = {'text': '#7FDBFF'}


layout = html.Div([
    dl.Map(dl.TileLayer(url=url, minZoom=1, maxZoom=4, attribution=attribution),
           style={'width': '50%', 'height': '500px', 'float': 'left'}),
    
    html.Div(children=[
        html.Label('Country'),
        dcc.Dropdown(['Canada', 'France', 'United States'], 'Canada',
                     placeholder='Select a country'),
        html.Br(),
        html.Label('Year'),
        dcc.Dropdown(['2022', '2021', '2020'], '2022',
                     placeholder='Select a year')
        ],
        style={'width': '25%', 'float': 'right'})
    
        ])