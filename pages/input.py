import uuid
import dash
from dash import html, dcc, Input, Output, State, callback, no_update

dash.register_page(__name__, path='/query', title='pearl')

PLACEHOLDERS = [
    "Tell me something about Haiti",
    "Di m yon bagay sou Ayiti",
    "Parlez-moi d'Haïti",
]

layout = html.Div(
    className='page-center query-page',
    children=[
        dcc.Interval(id='placeholder-interval', interval=3000, n_intervals=0),
        html.H1('pearl', className='pearl-text'),
        html.Div(
            className='query-input-wrapper',
            children=[
                html.Div(
                    className='query-input-box',
                    children=[
                        dcc.Input(
                            id='query-input',
                            type='text',
                            n_submit=0,
                            placeholder=PLACEHOLDERS[0],
                            className='query-input',
                            debounce=False,
                        ),
                    ],
                ),
            ],
        ),
        dcc.Location(id='query-redirect', refresh=True),
    ],
)


@callback(
    Output('query-input', 'placeholder'),
    Input('placeholder-interval', 'n_intervals'),
)
def rotate_placeholder(n_intervals):
    return PLACEHOLDERS[n_intervals % len(PLACEHOLDERS)]


@callback(
    Output('initial-query', 'data'),
    Output('thread-id', 'data'),
    Output('query-redirect', 'href'),
    Input('query-input', 'n_submit'),
    State('query-input', 'value'),
    State('thread-id', 'data'),
    prevent_initial_call=True,
)
def handle_submit(n_submit, value, thread_id):
    if not value or not value.strip():
        return no_update, no_update, no_update

    if not thread_id:
        thread_id = str(uuid.uuid4())

    return value.strip(), thread_id, '/chat'
