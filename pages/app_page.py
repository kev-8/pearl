import dash
from dash import html

dash.register_page(__name__, path='/app', title='pearl', name='app_page')

# TODO: 
    # Add clickable world map
    # Add country down menus
    # Add timeframe dropdown menu

colors = {'text': '#7FDBFF'}

layout = html.Div(children=[
    html.H1(children='placeholder',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
    )
])
