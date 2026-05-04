import dash
from dash import html, dcc, Input, Output, State, callback, ctx

dash.register_page(__name__, path='/', title='pearl')

DESCRIPTIONS = {
    'en': "Pearl was created to celebrate Haiti's rich history, vibrant culture, and resilient people. The name is a nod to Haiti's nickname, the Pearl of the Antilles.",
    'ht': "Pearl te kreye pou selebre istwa rich Ayiti, kilti pwosede ki vibwan, ak moun ki rezistan. Non an se yon ti non ann Ayiti, Pèl Zantiy yo.",
    'fr': "Pearl a été créée pour célébrer la riche histoire, sa culture dynamique et son peuple résilient. Le nom fait référence au surnom d'Haïti, la Perle des Antilles.",
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
        dcc.Store(id='landing-lang', data='en'),
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
    Output('landing-lang', 'data'),
    Output('landing-description', 'children'),
    Output('landing-enter', 'children'),
    Output('lang-btn-1', 'children'),
    Output('lang-btn-2', 'children'),
    Input('lang-btn-1', 'n_clicks'),
    Input('lang-btn-2', 'n_clicks'),
    State('landing-lang', 'data'),
    State('lang-btn-1', 'children'),
    State('lang-btn-2', 'children'),
    prevent_initial_call=True,
)
def switch_lang(_, __, current_lang, btn1_label, btn2_label):
    triggered = ctx.triggered_id
    new_lang = btn1_label.lower() if triggered == 'lang-btn-1' else btn2_label.lower()
    btn1_new, btn2_new = SWITCHER_OPTIONS[new_lang]
    return new_lang, DESCRIPTIONS[new_lang], ENTER_TEXT[new_lang], btn1_new, btn2_new


@callback(
    Output('info-panel', 'style'),
    Input('info-toggle', 'n_clicks'),
    prevent_initial_call=True,
)
def toggle_info(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    return {'display': 'none'}
