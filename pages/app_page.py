import dash
from dash import html

dash.register_page(__name__, path='/app', title='pearl')

colors = {'background': '#111111', 'text': '#7FDBFF'}

layout = html.Div(children=[
    html.H1(children='placeholder',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
    )
])
