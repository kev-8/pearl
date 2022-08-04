import dash
from dash import html

dash.register_page(__name__, path='/', title='pearl')


# TODO: 
    # center title
    # make font larger
    # make name hyperlink to app_page
    # upload logo under title
    # use loading shimmer custom component

colors = {'background': '#111111', 'text': '#7FDBFF'}

layout = html.Div(children=[
    html.H1(children='pearl',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
    )
])
