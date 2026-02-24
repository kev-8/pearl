import dash
from dash import html, dcc

dash.register_page(__name__, path='/', title='pearl')

layout = html.Div(
    className='page-center landing-page',
    children=[
        html.H1('pearl', className='pearl-text'),
        html.P(
            "Haiti's story is too often told through headlines alone. "
            "This constant stream of negative coverage shapes how the world sees the country, "
            "and misses so much of what makes it extraordinary. Pearl was created to change that: "
            "to celebrate Haiti's rich history, its resilient people, and its vibrant culture. "
            "The name is a nod to Haiti's nickname, the Pearl of the Antilles.",
            className='landing-description',
        ),
        html.A('Enter', href='/query', className='enter-btn'),
    ],
)
