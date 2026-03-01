import dash
from dash import html, dcc, Input, Output, State, callback, clientside_callback, no_update

from modeling import start_stream, get_stream_state

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
        # Streaming poll interval — enabled while a query is in flight
        dcc.Interval(id='stream-interval', interval=100, disabled=True, max_intervals=-1),
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


# ── Callback 3: kick off background stream when pending-query is set ──────────

@callback(
    Output('stream-interval', 'disabled'),
    Input('pending-query', 'data'),
    State('thread-id', 'data'),
    State('conversation-history', 'data'),
    prevent_initial_call=True,
)
def start_query_stream(pending_query, thread_id, history):
    if not pending_query:
        return True  # nothing to do — keep interval disabled
    start_stream(pending_query, history or [], thread_id or 'default')
    return False  # enable polling


# ── Callback 4: poll stream buffer and update display ─────────────────────────

@callback(
    Output('conversation-history', 'data'),
    Output('chat-messages', 'children', allow_duplicate=True),
    Output('pending-query', 'data', allow_duplicate=True),
    Output('stream-interval', 'disabled', allow_duplicate=True),
    Input('stream-interval', 'n_intervals'),
    State('pending-query', 'data'),
    State('thread-id', 'data'),
    State('conversation-history', 'data'),
    prevent_initial_call=True,
)
def poll_stream(n_intervals, pending_query, thread_id, history):
    if not pending_query:
        return no_update, no_update, no_update, True

    text, done = get_stream_state(thread_id or 'default')
    history_list = list(history or [])

    if done:
        # Finalise: persist to history, clear pending, disable interval
        final_history = history_list + [
            {'role': 'user',      'content': pending_query},
            {'role': 'assistant', 'content': text},
        ]
        return final_history, render_messages(final_history), None, True

    if text:
        # Stream in progress — show current partial content
        display = render_messages(history_list) + [
            html.Div(pending_query, className='chat-bubble chat-bubble-user'),
            dcc.Markdown(
                text,
                className='chat-bubble chat-bubble-assistant',
                link_target='_blank',
            ),
        ]
        return no_update, display, no_update, no_update

    # No text yet — keep showing thinking dots
    return no_update, no_update, no_update, no_update


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
