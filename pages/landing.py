import dash
from dash import html, Input, Output, State, callback, ctx

dash.register_page(__name__, path='/', title='pearl')

DESCRIPTIONS = {
    'en': "Pearl was created to celebrate Haiti's rich history, vibrant culture, and resilient people. The name is a nod to Haiti's nickname, the Pearl of the Antilles.",
    'ht': "Pearl te kreye pou selebre istwa rich Ayiti, kilti vibran li, ak pèp rezilyan li. Non an fè referans ak ti non Ayiti a, Pèl Zantiy yo.",
    'fr': "Pearl a été créée pour célébrer la riche histoire d'Haïti, sa culture dynamique et son peuple résilient. Le nom fait référence au surnom d'Haïti, la Perle des Antilles.",
}

ENTER_TEXT = {
    'en': 'Enter',
    'ht': 'Antre',
    'fr': 'Entrer',
}

SWITCHER_OPTIONS = {
    'en': ('HT', 'FR'),
    'ht': ('EN', 'FR'),
    'fr': ('HT', 'EN'),
}

layout = html.Div(
    className='page-center landing-page',
    children=[
        html.Div(html.H1('pearl', className='pearl-text'), className='pearl-text-wrapper'),
        html.P(
            DESCRIPTIONS['en'],
            id='landing-description',
            className='landing-description',
        ),
        html.A('Enter', href='/query', id='landing-enter', className='enter-btn'),
        html.Div(
            className='lang-switcher',
            children=[
                html.Span('HT', id='lang-btn-1', n_clicks=0, className='lang-option'),
                html.Span(' | ', className='lang-sep'),
                html.Span('FR', id='lang-btn-2', n_clicks=0, className='lang-option'),
            ],
        ),
        html.Div(
            className='info-container',
            children=[
                html.Span('ⓘ', id='info-toggle', n_clicks=0, className='info-icon'),
                html.Div(
                    id='info-panel',
                    className='info-panel',
                    style={'display': 'none'},
                    children=[
                        html.P('Data Sources: Le Nouvelliste via Digital Library of the Caribbean (dLOC),' \
                        '\nRadio Haïti broadcasts from the Duke University Libraries digital repository', className='info-text'),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output('answer-language', 'data'),
    Input('lang-btn-1', 'n_clicks'),
    Input('lang-btn-2', 'n_clicks'),
    State('lang-btn-1', 'children'),
    State('lang-btn-2', 'children'),
    prevent_initial_call=True,
)
def switch_lang(_, __, btn1_label, btn2_label):
    if ctx.triggered_id == 'lang-btn-1':
        selected = btn1_label.lower()
    elif ctx.triggered_id == 'lang-btn-2':
        selected = btn2_label.lower()
    else:
        return dash.no_update
    # English is the default landing view; returning to it clears the override.
    return None if selected == 'en' else selected


@callback(
    Output('landing-description', 'children'),
    Output('landing-enter', 'children'),
    Output('lang-btn-1', 'children'),
    Output('lang-btn-2', 'children'),
    Input('answer-language', 'data'),
)
def render_language_choice(selected_language):
    display_language = selected_language if selected_language in DESCRIPTIONS else 'en'
    btn1, btn2 = SWITCHER_OPTIONS[display_language]
    return DESCRIPTIONS[display_language], ENTER_TEXT[display_language], btn1, btn2


@callback(
    Output('info-panel', 'style'),
    Input('info-toggle', 'n_clicks'),
    prevent_initial_call=True,
)
def toggle_info(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    return {'display': 'none'}
