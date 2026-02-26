import dash
from dash import html, dcc, Input, Output, State, callback, clientside_callback, no_update

from modeling import conversational_agent

dash.register_page(__name__, path='/chat', title='pearl')

layout = html.Div(
    className='chat-page-wrapper',
    children=[
        # Top bar
        html.Div(
            className='chat-top-bar',
            children=[
                html.A('pearl', href='/', className='pearl-static'),
            ],
        ),
        # Scrollable message area
        html.Div(id='chat-messages', className='chat-container'),
        # Stores
        dcc.Store(id='conversation-history', data=[]),
        dcc.Store(id='pending-query', data=None),
        # One-shot interval — fires initial load after mount
        dcc.Interval(id='init-trigger', interval=300, max_intervals=1, n_intervals=0),
        # Fixed bottom input bar
        html.Div(
            className='chat-input-bar',
            children=[
                html.Div(
                    className='chat-input-box',
                    children=[
                        dcc.Input(
                            id='chat-input',
                            type='text',
                            n_submit=0,
                            placeholder='Ask a follow-up…',
                            className='chat-input',
                            debounce=False,
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def render_messages(history):
    """Convert list of {role, content} dicts to Dash bubble components."""
    bubbles = []
    for msg in history:
        role = msg.get('role', 'assistant')
        content = msg.get('content', '')
        if role == 'user':
            bubbles.append(html.Div(content, className='chat-bubble chat-bubble-user'))
        else:
            bubbles.append(
                dcc.Markdown(
                    content,
                    className='chat-bubble chat-bubble-assistant',
                    link_target='_blank',
                )
            )
    return bubbles


def _thinking_dots():
    return html.Div(
        [html.Span(className='dot'), html.Span(className='dot'), html.Span(className='dot')],
        className='chat-bubble chat-bubble-assistant chat-bubble-thinking',
    )


def _thinking_state(history, query):
    """Render history + user bubble + animated dots placeholder."""
    bubbles = render_messages(history)
    bubbles.append(html.Div(query, className='chat-bubble chat-bubble-user'))
    bubbles.append(_thinking_dots())
    return bubbles


def _invoke(query, thread_id):
    result = conversational_agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


# ── Callback 1: show thinking state immediately on page load ──────────────────

@callback(
    Output('chat-messages', 'children'),
    Output('pending-query', 'data'),
    Input('init-trigger', 'n_intervals'),
    State('initial-query', 'data'),
    State('conversation-history', 'data'),
    prevent_initial_call=True,
)
def initial_show(n_intervals, initial_query, history):
    if history or not initial_query:
        return render_messages(history or []), no_update
    return _thinking_state([], initial_query), initial_query


# ── Callback 2: show thinking state immediately on follow-up submit ───────────

@callback(
    Output('chat-messages', 'children', allow_duplicate=True),
    Output('chat-input', 'value'),
    Output('pending-query', 'data', allow_duplicate=True),
    Input('chat-input', 'n_submit'),
    State('chat-input', 'value'),
    State('conversation-history', 'data'),
    prevent_initial_call=True,
)
def submit_show(n_submit, value, history):
    if not value or not value.strip():
        return no_update, no_update, no_update
    query = value.strip()
    return _thinking_state(history or [], query), '', query


# ── Callback 3: invoke model once pending-query is set ────────────────────────

@callback(
    Output('conversation-history', 'data'),
    Output('chat-messages', 'children', allow_duplicate=True),
    Output('pending-query', 'data', allow_duplicate=True),
    Input('pending-query', 'data'),
    State('thread-id', 'data'),
    State('conversation-history', 'data'),
    prevent_initial_call=True,
)
def process_query(pending_query, thread_id, history):
    if not pending_query:
        return no_update, no_update, no_update

    answer = _invoke(pending_query, thread_id)
    history = list(history or [])
    history.append({'role': 'user',      'content': pending_query})
    history.append({'role': 'assistant', 'content': answer})
    return history, render_messages(history), None


# ── Clientside: scroll to bottom on any message update ───────────────────────

clientside_callback(
    """
    function(children) {
        var el = document.getElementById('chat-messages');
        if (el) { el.scrollTop = el.scrollHeight; }
        return null;
    }
    """,
    Output('chat-messages', 'data-scroll'),
    Input('chat-messages', 'children'),
)
