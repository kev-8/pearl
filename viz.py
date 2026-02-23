import dash
from dash import Dash, html, dcc

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400&display=swap"
    ],
)

app.layout = html.Div([
    dcc.Store(id='initial-query', storage_type='session'),
    dcc.Store(id='thread-id', storage_type='session'),
    dash.page_container,
])

if __name__ == "__main__":
    app.run(debug=True)
