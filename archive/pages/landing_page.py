import dash
from dash import html

dash.register_page(__name__, path='/', title='pearl')


colors = {'text': '#d4b1aa'}

layout = html.Div(
    children=[
    html.H1(
            className='shimmer',
            children='pearl'
            
    ),
    
    html.A(
        html.Img(
                src=r'assets/pearl.png',
                height = 250,
                width =  250
        ),
        href='/app'
    )
])
        


