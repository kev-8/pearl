import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/app', title='pearl', name='app_page')

# TODO: load in df and create filters based on menu selection

url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '

colors = {'text': '#7FDBFF'}

modeling_df = pd.read_csv('./modeling_df.csv')
modeling_df['country'] = 'Canada'

fig = px.bar(modeling_df, x="top_entity_names", y="country")

layout = html.Div([
    
    # dbc.Row([
    #     dbc.Col(
    #         dbc.DropdownMenu(
    #             label='Country',
    #             menu_variant='dark',
    #             toggle_style={'background': '#d4b1aa'},
    #             children=[
    #                 dbc.DropdownMenuItem('Canada'),
    #                 dbc.DropdownMenuItem('United States'),
    #                 dbc.DropdownMenuItem('France')
    #             ]
    #         ),
            
    #         dbc.DropdownMenu(
    #             label='Year',
    #             menu_variant='dark',
    #             toggle_style={'background': '#d4b1aa'},
    #             children=[
    #                 dbc.DropdownMenuItem('2022'),
    #                 dbc.DropdownMenuItem('2021'),
    #                 dbc.DropdownMenuItem('2020')
    #             ]
    #         ),
    #         align='center',
    #         width=3
    #     ),
        
    #     dbc.Col(
    #         dl.Map(dl.TileLayer(url=url, minZoom=1, maxZoom=4, attribution=attribution)),
    #         width=9
    #         )
    # ])
    
    dbc.Card(
    dbc.CardBody([
    dbc.Row([
        dbc.Col(
            [
            html.Label('Country'),    
            dcc.Dropdown(['Canada', 'France', 'United States'], 'Canada',
                         placeholder='Select a country')
            ]
        ),
        html.Br(),
        html.Br(),
        dbc.Col(
            [
            html.Label('Year'),
            dcc.Dropdown(['2022', '2021', '2020'], '2022',
                         placeholder='Select a year')    
            ]
        ),
        
        # TODO: make map move based on country selection
        # TODO: add border around the map
        dl.Map(dl.TileLayer(url=url, minZoom=1, maxZoom=6, attribution=attribution),
                style={'width': '55%', 'height': '300px', 'float': 'left'})
    ])
    ]),
    color='#000'
    ),
    
    dbc.Card(
    dbc.CardBody([
    dbc.Row([
        dbc.Col(
            [
            html.Label('Entities'),
            dcc.Graph(figure=fig)
            ]
        ),
        dbc.Col(
            [
            html.Label('Topics'),
            dcc.Graph(figure=fig)
            ]
        ),
        dbc.Col(
            [
            html.Label('Sentiment'),
            dcc.Graph(figure=fig)
            ]
        )
    ])    
    ]),
    color='#000'    
    )

    
    
    # Dropdown menus using DCC instead of DBC. might need later for callbacks
    
    # html.Div(children=[
    #     html.Label('Country'),
    #     dcc.Dropdown(['Canada', 'France', 'United States'], 'Canada',
    #                   placeholder='Select a country'),
    #     html.Br(),
    #     html.Label('Year'),
    #     dcc.Dropdown(['2022', '2021', '2020'], '2022',
    #                   placeholder='Select a year')
    #     ],
    #     style={'width': '25%', 'float': 'right'})
    
])