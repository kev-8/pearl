import dash
from dash import html

dash.register_page(__name__, path='/', title='pearl')

layout = html.Div(
    className='page-center',
    children=[
        html.H1('pearl', className='shimmer-title'),
        html.P(
            "In this day and age, most of the news that comes out about Haiti is negative. "
            "This consistent stream of negative media skews perception of the country. "
            "This tool was created to uplift history and positive news about Haiti and show "
            "the world what a beautiful country it truly is. The name pearl comes from Haiti's "
            "nickname: Pearl of the Antilles.",
            className='landing-description',
        ),
        html.A('Enter', href='/query', className='enter-btn'),
    ],
)
