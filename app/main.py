"""Resumai — AI-powered CV interview service."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.graph import interview_graph
from app.graph.state import _empty_collected
from app.graph.nodes import generate_opening

app = FastAPI(title="Resumai", description="AI-powered CV interview service")

# ── In-memory session store (replaced by Mongo later) ─────────────
sessions: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────

class StartRequest(BaseModel):
    session_id: str
    target_role: str


class MessageRequest(BaseModel):
    session_id: str
    text: str


class InterviewResponse(BaseModel):
    question: str
    phase: str
    finished: bool


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    return CHAT_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/interview/start", response_model=InterviewResponse)
def start_interview(req: StartRequest):
    """Start a new interview session. Returns the first question."""
    first_q = generate_opening(req.target_role)

    state = {
        "session_id": req.session_id,
        "target_role": req.target_role,
        "messages": [AIMessage(content=first_q)],
        "collected": _empty_collected(),
        "current_question": first_q,
        "finished": False,
        "phase": "open",
        "deep_dive_items": [],
        "deep_dive_cursor": 0,
        "summary_confirmed": False,
    }
    sessions[req.session_id] = state

    return InterviewResponse(
        question=first_q,
        phase="open",
        finished=False,
    )


@app.post("/interview/message", response_model=InterviewResponse)
def send_message(req: MessageRequest):
    """Send a user message and get the next interview response."""
    state = sessions.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found. Call /interview/start first.")

    if state.get("finished"):
        raise HTTPException(status_code=400, detail="Interview already finished.")

    # Add user message to state
    state["messages"] = [*state["messages"], HumanMessage(content=req.text)]

    # Run the graph (routes to correct phase node)
    result = interview_graph.invoke(state)

    # Add AI response to messages
    result["messages"] = [*result["messages"], AIMessage(content=result["current_question"])]

    # Persist updated state
    sessions[req.session_id] = result

    return InterviewResponse(
        question=result["current_question"],
        phase=result.get("phase", "open"),
        finished=result.get("finished", False),
    )


@app.get("/interview/{session_id}/status")
def get_status(session_id: str):
    """Get current interview state (for debugging)."""
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": state["session_id"],
        "target_role": state["target_role"],
        "phase": state.get("phase", "open"),
        "collected": state["collected"],
        "finished": state.get("finished", False),
        "current_question": state["current_question"],
        "deep_dive_cursor": state.get("deep_dive_cursor", 0),
        "deep_dive_items": state.get("deep_dive_items", []),
        "message_count": len(state["messages"]),
    }


# ── Inline chat UI ───────────────────────────────────────────────

CHAT_HTML = """\
<!DOCTYPE html>
<html lang="he" dir="ltr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resumai</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

/* -- Header -- */
.header { background: #1a1a2e; padding: 16px 24px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #2a2a3e; }
.header h1 { font-size: 20px; color: #fff; }
.phase-badge { background: #16213e; color: #4fc3f7; padding: 4px 12px; border-radius: 12px; font-size: 13px; }

/* -- Setup -- */
.setup { display: flex; gap: 12px; padding: 20px 24px; background: #1a1a1a; border-bottom: 1px solid #222; }
.setup input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; background: #252525; color: #fff; font-size: 14px; }
.setup button { padding: 10px 20px; border-radius: 8px; border: none; background: #4fc3f7; color: #000; font-weight: 600; cursor: pointer; font-size: 14px; }
.setup button:hover { background: #81d4fa; }

/* -- Main layout -- */
.main { flex: 1; display: flex; overflow: hidden; }

/* -- Chat side -- */
.chat-side { flex: 1; display: flex; flex-direction: column; border-right: 1px solid #222; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 80%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
.msg.ai { align-self: flex-start; background: #1e293b; color: #e2e8f0; border-bottom-left-radius: 4px; }
.msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
.msg.loading { align-self: flex-start; background: #1e293b; color: #94a3b8; font-style: italic; }
.input-bar { display: flex; gap: 8px; padding: 16px; background: #1a1a1a; border-top: 1px solid #222; }
.input-bar input { flex: 1; padding: 12px 16px; border-radius: 20px; border: 1px solid #333; background: #252525; color: #fff; font-size: 14px; }
.input-bar input:focus { outline: none; border-color: #4fc3f7; }
.input-bar button { padding: 12px 20px; border-radius: 20px; border: none; background: #4fc3f7; color: #000; font-weight: 600; cursor: pointer; }
.input-bar button:disabled { opacity: 0.4; cursor: not-allowed; }

/* -- Data side -- */
.data-side { width: 380px; overflow-y: auto; padding: 20px; background: #111; }
.data-side h2 { font-size: 15px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
.category { margin-bottom: 20px; }
.category h3 { font-size: 14px; color: #4fc3f7; margin-bottom: 8px; }
.entry { background: #1a1a2e; padding: 10px 14px; border-radius: 8px; margin-bottom: 6px; font-size: 13px; line-height: 1.5; }
.entry .field { color: #94a3b8; }
.entry .value { color: #e0e0e0; }
.empty { color: #555; font-style: italic; font-size: 13px; }

/* -- Deep dive tracker -- */
.dive-tracker { margin-bottom: 20px; }
.dive-item { padding: 6px 10px; border-radius: 6px; margin-bottom: 4px; font-size: 13px; }
.dive-item.done { color: #4caf50; text-decoration: line-through; opacity: 0.6; }
.dive-item.current { color: #4fc3f7; font-weight: 600; background: #1a1a2e; }
.dive-item.pending { color: #555; }
</style>
</head>
<body>
<div class="header">
  <h1>Resumai</h1>
  <span class="phase-badge" id="phaseBadge">setup</span>
</div>

<div class="setup" id="setupBar">
  <input id="roleInput" placeholder="Target role (e.g. Backend Developer)" />
  <button onclick="startInterview()">Start</button>
</div>

<div class="main">
  <div class="chat-side">
    <div class="messages" id="messages"></div>
    <div class="input-bar">
      <input id="userInput" placeholder="Type your message..." onkeydown="if(event.key==='Enter')sendMsg()" disabled />
      <button id="sendBtn" onclick="sendMsg()" disabled>Send</button>
    </div>
  </div>
  <div class="data-side">
    <h2>Collected Data</h2>
    <div id="diveTracker"></div>
    <div id="collectedData"><p class="empty">Start an interview to see data here.</p></div>
  </div>
</div>

<script>
let sessionId = null;
let busy = false;

function addMsg(text, cls) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.textContent = text;
  document.getElementById('messages').appendChild(el);
  el.scrollIntoView({ behavior: 'smooth' });
  return el;
}

async function startInterview() {
  const role = document.getElementById('roleInput').value.trim();
  if (!role) return;
  sessionId = 'session_' + Date.now();
  document.getElementById('setupBar').style.display = 'none';

  const loading = addMsg('Starting...', 'loading');
  const res = await fetch('/interview/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, target_role: role })
  });
  const data = await res.json();
  loading.remove();
  addMsg(data.question, 'ai');
  updatePhase(data.phase);
  document.getElementById('userInput').disabled = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('userInput').focus();
  refreshStatus();
}

async function sendMsg() {
  if (busy) return;
  const input = document.getElementById('userInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg(text, 'user');

  busy = true;
  document.getElementById('sendBtn').disabled = true;
  const loading = addMsg('Thinking...', 'loading');

  const res = await fetch('/interview/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, text: text })
  });
  const data = await res.json();
  loading.remove();
  addMsg(data.question, 'ai');
  updatePhase(data.phase);

  if (data.finished) {
    input.disabled = true;
    document.getElementById('sendBtn').disabled = true;
  } else {
    document.getElementById('sendBtn').disabled = false;
    input.focus();
  }
  busy = false;
  refreshStatus();
}

function updatePhase(phase) {
  const badge = document.getElementById('phaseBadge');
  const labels = { open: 'Open Chat', summary: 'Summary', deep_dive: 'Deep Dive', done: 'Done' };
  badge.textContent = labels[phase] || phase;
}

async function refreshStatus() {
  if (!sessionId) return;
  const res = await fetch('/interview/' + sessionId + '/status');
  const data = await res.json();
  renderCollected(data.collected);
  renderDiveTracker(data.deep_dive_items || [], data.deep_dive_cursor || 0, data.phase);
}

function renderCollected(collected) {
  const el = document.getElementById('collectedData');
  let html = '';
  const labels = { work_history: 'Work History', education: 'Education', skills: 'Skills', highlight_project: 'Projects' };
  for (const [cat, entries] of Object.entries(collected)) {
    if (!entries.length) continue;
    html += '<div class="category"><h3>' + (labels[cat] || cat) + '</h3>';
    for (const entry of entries) {
      html += '<div class="entry">';
      for (const [k, v] of Object.entries(entry)) {
        if (!v || (Array.isArray(v) && !v.length)) continue;
        const val = Array.isArray(v) ? v.join(', ') : v;
        html += '<span class="field">' + k + ':</span> <span class="value">' + val + '</span><br>';
      }
      html += '</div>';
    }
    html += '</div>';
  }
  el.innerHTML = html || '<p class="empty">Nothing collected yet.</p>';
}

function renderDiveTracker(items, cursor, phase) {
  const el = document.getElementById('diveTracker');
  if (!items.length || phase === 'open' || phase === 'summary') { el.innerHTML = ''; return; }
  let html = '<div class="dive-tracker"><h3 style="color:#94a3b8;font-size:13px;margin-bottom:8px;">DEEP DIVE PROGRESS</h3>';
  items.forEach((item, i) => {
    let cls = 'pending';
    if (i < cursor) cls = 'done';
    else if (i === cursor) cls = 'current';
    html += '<div class="dive-item ' + cls + '">' + (i < cursor ? '\\u2713 ' : i === cursor ? '\\u25B6 ' : '  ') + item.label + '</div>';
  });
  html += '</div>';
  el.innerHTML = html;
}
</script>
</body>
</html>
"""
