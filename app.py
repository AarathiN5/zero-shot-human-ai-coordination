"""
app.py — DTS-ZSC Showcase Web UI
Run with:  python app.py
Then open: http://localhost:5000

This file does NOT change any existing code.
It just wraps the real environment in a Flask server
and serves a live dashboard in your browser.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, render_template_string
from env.collaborative_env import CollaborativeTaskEnv, OBS_DIM
from agent.ppo_agent        import PPOAgent
from agent.human_observer   import HumanBehaviorObserver, EXTRA_OBS_DIM
from env.task_definitions   import WAREHOUSE_TASKS, NUM_TASKS
from human.simulated_human  import HUMAN_PROFILES

import numpy as np

app = Flask(__name__)

# ── Global session state ──────────────────────────────────────────────────
session = {
    "env":    None,
    "agent":  None,
    "hbo":    None,
    "obs":    None,
    "info":   None,
    "done":   False,
    "reward": 0.0,
    "step":   0,
    "conflicts": 0,
    "sync":   0,
    "parallel": 0,
    "h_done": 0,
    "ai_done": 0,
    "role_switches": 0,
    "prev_assignee": None,
    "task_status":   [0]*NUM_TASKS,
    "task_progress": [0.0]*NUM_TASKS,
    "task_assignee": [None]*NUM_TASKS,
    "profile": "average",
}

def ces():
    s = session
    n = max(s["step"], 1)
    compl  = sum(1 for x in s["task_status"] if x == 2) / NUM_TASKS
    speed  = max(0, 1 - s["step"] / 200)
    sync   = s["parallel"] / n
    conf   = s["conflicts"] / n
    return round(0.4*compl + 0.3*speed + 0.2*sync - 0.1*conf, 3)

def reset_session(profile_name="average"):
    profile = HUMAN_PROFILES.get(profile_name, HUMAN_PROFILES["average"])
    env  = CollaborativeTaskEnv(human_profile=profile)
    hbo  = HumanBehaviorObserver(window=10)
    obs, info = env.reset(seed=42)
    hbo.reset()
    obs = hbo.enhance(obs)

    # Try loading a trained agent; fall back to untrained
    agent = PPOAgent(obs_dim=OBS_DIM + EXTRA_OBS_DIM)
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "agent_final.pt")
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        print(f"[UI] Loaded trained agent from {ckpt_path}")
    else:
        print("[UI] No checkpoint found — using untrained agent (run train.py first for best results)")

    session.update({
        "env": env, "agent": agent, "hbo": hbo,
        "obs": obs, "info": info, "done": False,
        "reward": 0.0, "step": 0, "conflicts": 0,
        "sync": 0, "parallel": 0, "h_done": 0, "ai_done": 0,
        "role_switches": 0, "prev_assignee": None,
        "task_status":   [0]*NUM_TASKS,
        "task_progress": [0.0]*NUM_TASKS,
        "task_assignee": [None]*NUM_TASKS,
        "profile": profile_name,
    })

def step_once():
    s = session
    if s["done"]:
        return

    agent: PPOAgent       = s["agent"]
    env:   CollaborativeTaskEnv = s["env"]
    hbo:   HumanBehaviorObserver = s["hbo"]

    action, logp, value = agent.select_action(s["obs"], s["info"])
    task_id  = action // 2
    assignee = action  % 2

    if s["prev_assignee"] is not None and assignee != s["prev_assignee"]:
        s["role_switches"] += 1
    s["prev_assignee"] = assignee

    obs, reward, terminated, truncated, info = env.step(action)
    hbo.update(info)
    obs = hbo.enhance(obs)

    s["obs"]    = obs
    s["info"]   = info
    s["reward"] += reward
    s["step"]   += 1

    if reward <= -1.9:
        s["conflicts"] += 1

    ai_busy    = info.get("ai_task") is not None
    human_busy = env.human.is_busy
    if ai_busy and human_busy:
        s["parallel"] += 1
        s["sync"]     += 1

    # Sync task display state from env internals
    for i in range(NUM_TASKS):
        raw = float(env._task_status[i])
        if raw == 1.0:
            s["task_status"][i] = 2       # done
        elif raw == 0.5:
            s["task_status"][i] = 1       # in progress
        else:
            s["task_status"][i] = 0       # pending

        if env._ai_current_task == i:
            prog = env._ai_time_on_task / max(env._ai_task_budget, 0.01)
            s["task_progress"][i] = min(prog, 1.0)
            s["task_assignee"][i] = "ai"
        elif env.human.current_task_id == i:
            prog = env.human.time_on_task / max(env.human.task_budget, 0.01)
            s["task_progress"][i] = min(prog, 1.0)
            s["task_assignee"][i] = "human"
        elif s["task_status"][i] == 2:
            s["task_progress"][i] = 1.0

    if terminated or truncated:
        s["done"] = True

# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    profile = request.json.get("profile", "average")
    reset_session(profile)
    return api_state()

@app.route("/api/step", methods=["POST"])
def api_step():
    count = request.json.get("count", 1)
    for _ in range(count):
        if session["done"]:
            break
        step_once()
    return api_state()

@app.route("/api/state")
def api_state():
    s   = session
    env = s["env"]
    hm  = env.human.get_metrics() if env and env.human else {}
    tasks = []
    for t in WAREHOUSE_TASKS:
        prereqs_met = all(s["task_status"][p] == 2 for p in t.prerequisites)
        tasks.append({
            "id":       t.id,
            "name":     t.name,
            "label":    t.name.replace("_", " ").title(),
            "diff":     t.difficulty,
            "prereqs":  t.prerequisites,
            "prereqs_met": prereqs_met,
            "status":   s["task_status"][t.id],
            "progress": round(s["task_progress"][t.id], 3),
            "assignee": s["task_assignee"][t.id],
        })
    return jsonify({
        "step":        s["step"],
        "reward":      round(s["reward"], 2),
        "ces":         ces() if s["step"] > 0 else None,
        "conflicts":   s["conflicts"],
        "sync":        s["sync"],
        "role_switches": s["role_switches"],
        "done":        s["done"],
        "profile":     s["profile"],
        "human": {
            "speed":      round(hm.get("speed", 0.5), 3),
            "fatigue":    round(hm.get("fatigue", 0.0), 3),
            "error_rate": round(hm.get("error_rate", 0.0), 3),
            "tasks_done": hm.get("tasks_done", 0),
            "is_busy":    env.human.is_busy if env and env.human else False,
            "current_task": env.human.current_task_id if env and env.human else None,
        },
        "ai": {
            "current_task": env._ai_current_task if env else None,
            "tasks_done":  s["ai_done"],
        },
        "tasks": tasks,
    })

# ── HTML ──────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DTS-ZSC: Live Showcase</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #f9f9f8;
    --surface:  #ffffff;
    --surface2: #f1f0eb;
    --border:   rgba(0,0,0,0.10);
    --text:     #1a1a18;
    --muted:    #6b6b66;
    --hint:     #9a9a93;
    --purple:   #534AB7;
    --purple-lt:#EEEDFE;
    --amber:    #BA7517;
    --amber-lt: #FAEEDA;
    --green:    #3B6D11;
    --green-lt: #EAF3DE;
    --red:      #A32D2D;
    --red-lt:   #FCEBEB;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
  .page { max-width: 960px; margin: 0 auto; padding: 24px 20px 48px; }

  /* Header */
  .hdr { display: flex; align-items: flex-start; justify-content: space-between;
         margin-bottom: 24px; flex-wrap: wrap; gap: 12px; }
  .hdr h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.3px; }
  .hdr p  { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  select, button { font-size: 13px; padding: 6px 12px; border-radius: 8px; cursor: pointer;
                   border: 1px solid var(--border); background: var(--surface);
                   color: var(--text); transition: background .15s; }
  button:hover { background: var(--surface2); }
  button.primary { background: var(--purple); color: #fff; border-color: var(--purple); }
  button.primary:hover { background: #3C3489; }
  button:disabled { opacity: 0.4; cursor: default; }

  /* Metric cards */
  .metrics { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; margin-bottom: 20px; }
  .mc { background: var(--surface2); border-radius: 8px; padding: 12px; }
  .mc .lbl { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .mc .val { font-size: 24px; font-weight: 600; }
  .mc .sub { font-size: 11px; color: var(--hint); margin-top: 2px; }

  /* Layout */
  .body-grid { display: grid; grid-template-columns: 1fr 260px; gap: 16px; }
  @media(max-width:680px){ .body-grid { grid-template-columns: 1fr; } .metrics { grid-template-columns: repeat(2,1fr); } }

  /* Task list */
  .section-label { font-size: 11px; color: var(--muted); margin-bottom: 6px;
                   display: grid; grid-template-columns: 24px 1fr 100px 80px;
                   gap: 10px; padding: 0 12px; }
  .task-list { display: flex; flex-direction: column; gap: 7px; }
  .task-row { display: grid; grid-template-columns: 24px 1fr 100px 80px;
              gap: 10px; align-items: center; padding: 10px 12px;
              border: 1px solid var(--border); border-radius: 10px;
              background: var(--surface); transition: border-color .2s, background .2s; }
  .task-row.active   { border-color: var(--purple); background: var(--purple-lt); }
  .task-row.done     { opacity: 0.55; }
  .task-row.locked   { opacity: 0.40; }
  .num { width: 24px; height: 24px; border-radius: 50%; font-size: 11px; font-weight: 600;
         display: flex; align-items: center; justify-content: center;
         background: var(--surface2); color: var(--muted); flex-shrink: 0; }
  .task-row.active .num { background: var(--purple); color: #fff; }
  .task-row.done   .num { background: var(--green); color: var(--green-lt); }
  .t-name { font-size: 13px; font-weight: 500; }
  .t-sub  { font-size: 11px; color: var(--hint); }
  .pill { font-size: 11px; font-weight: 500; padding: 3px 9px; border-radius: 12px; text-align: center; }
  .pill-pending { background: var(--surface2); color: var(--muted); }
  .pill-human   { background: var(--amber-lt); color: var(--amber); }
  .pill-ai      { background: var(--purple-lt); color: var(--purple); }
  .pill-done    { background: var(--green-lt); color: var(--green); }
  .prog-wrap { height: 4px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
  .prog-bar  { height: 100%; border-radius: 2px; background: var(--purple); transition: width .3s; }
  .task-row.done .prog-bar { background: var(--green); }

  /* Side panel */
  .side { display: flex; flex-direction: column; gap: 12px; }
  .card { border: 1px solid var(--border); border-radius: 10px; padding: 14px;
          background: var(--surface); }
  .card h3 { font-size: 13px; font-weight: 600; margin-bottom: 10px;
             display: flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .stat-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; }
  .stat-row:last-child { margin-bottom: 0; }
  .stat-row .sl { font-size: 12px; color: var(--muted); }
  .stat-row .sv { font-size: 12px; font-weight: 500; }
  .bar-w { flex: 1; margin: 0 8px; height: 3px; background: var(--border); border-radius: 2px; }
  .bar-f { height: 100%; border-radius: 2px; transition: width .4s; }
  .log-box { font-size: 11px; font-family: monospace; max-height: 150px;
             overflow-y: auto; display: flex; flex-direction: column-reverse; gap: 2px; }
  .log-line { color: var(--muted); padding: 1px 0; }
  .log-line.ok  { color: var(--green); }
  .log-line.bad { color: var(--red); }

  /* Status bar */
  .status-bar { margin-top: 16px; padding: 10px 14px; border-radius: 8px;
                background: var(--surface); border: 1px solid var(--border);
                font-size: 12px; color: var(--muted); display: flex;
                justify-content: space-between; align-items: center; }
  .status-bar.done-bar { background: var(--green-lt); border-color: var(--green); color: var(--green); }
</style>
</head>
<body>
<div class="page">
  <div class="hdr">
    <div>
      <h1>DTS-ZSC &mdash; Live Showcase</h1>
      <p>Dynamic Task Sequencing &middot; Zero-Shot Human-AI Coordination &middot; PPO Agent</p>
    </div>
    <div class="controls">
      <select id="profileSel" onchange="doReset()">
        <option value="average">Human: average</option>
        <option value="fast">Human: fast</option>
        <option value="slow">Human: slow</option>
        <option value="novice">Human: novice (high errors)</option>
        <option value="expert">Human: expert</option>
        <option value="tired">Human: tired (high fatigue)</option>
      </select>
      <button id="stepBtn" onclick="doStep(1)">Step &times;1</button>
      <button id="playBtn" class="primary" onclick="togglePlay()">&#9654; Run</button>
      <button onclick="doReset()">Reset</button>
    </div>
  </div>

  <div class="metrics">
    <div class="mc"><div class="lbl">Joint reward</div><div class="val" id="mReward">0.0</div><div class="sub">cumulative score</div></div>
    <div class="mc"><div class="lbl">CES score</div><div class="val" id="mCES">&mdash;</div><div class="sub">coordination efficiency</div></div>
    <div class="mc"><div class="lbl">Step</div><div class="val" id="mStep">0</div><div class="sub">of 200 max</div></div>
    <div class="mc"><div class="lbl">Conflicts</div><div class="val" id="mConflicts">0</div><div class="sub">invalid assignments</div></div>
  </div>

  <div class="body-grid">
    <div>
      <div class="section-label">
        <span></span><span>Task</span><span>Assigned to</span><span>Progress</span>
      </div>
      <div class="task-list" id="taskList"></div>
    </div>
    <div class="side">
      <div class="card">
        <h3><span class="dot" style="background:var(--amber)"></span> Human partner <span id="profBadge" style="font-size:10px;padding:2px 7px;border-radius:10px;background:var(--amber-lt);color:var(--amber);font-weight:500">average</span></h3>
        <div class="stat-row"><span class="sl">Speed</span><div class="bar-w"><div class="bar-f" id="bSpd" style="background:var(--amber);width:50%"></div></div><span class="sv" id="vSpd">0.50</span></div>
        <div class="stat-row"><span class="sl">Fatigue</span><div class="bar-w"><div class="bar-f" id="bFtg" style="background:var(--red);width:0%"></div></div><span class="sv" id="vFtg">0.00</span></div>
        <div class="stat-row"><span class="sl">Error rate</span><div class="bar-w"><div class="bar-f" id="bErr" style="background:var(--red);width:0%"></div></div><span class="sv" id="vErr">0.00</span></div>
        <div style="font-size:11px;color:var(--hint);margin-top:8px">Tasks done: <strong id="vHDone" style="color:var(--text)">0</strong> &nbsp;&middot;&nbsp; Currently: <strong id="vHTask" style="color:var(--text)">idle</strong></div>
      </div>
      <div class="card">
        <h3><span class="dot" style="background:var(--purple)"></span> RL agent (PPO)</h3>
        <div class="stat-row"><span class="sl">Current task</span><span class="sv" id="vAITask">idle</span></div>
        <div class="stat-row"><span class="sl">Role switches</span><span class="sv" id="vRoleSw">0</span></div>
        <div class="stat-row"><span class="sl">Sync steps</span><span class="sv" id="vSync">0</span></div>
      </div>
      <div class="card">
        <h3>Event log</h3>
        <div class="log-box" id="logBox"></div>
      </div>
    </div>
  </div>

  <div class="status-bar" id="statusBar">
    Ready &mdash; press Run to start the simulation
  </div>
</div>

<script>
let playing = false;
let timer   = null;
let logLines = [];

async function apiFetch(path, body={}) {
  const r = await fetch(path, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}

async function doReset() {
  stopPlay();
  const p = document.getElementById('profileSel').value;
  const d = await apiFetch('/api/reset', {profile: p});
  logLines = [];
  render(d);
}

async function doStep(n=1) {
  if (document.getElementById('playBtn').disabled) return;
  const d = await apiFetch('/api/step', {count: n});
  render(d);
}

function stopPlay() {
  playing = false;
  clearInterval(timer);
  document.getElementById('playBtn').textContent = '\u25B6 Run';
}

function togglePlay() {
  if (playing) { stopPlay(); return; }
  playing = true;
  document.getElementById('playBtn').textContent = '\u23F8 Pause';
  timer = setInterval(async () => {
    const d = await apiFetch('/api/step', {count:1});
    render(d);
    if (d.done) stopPlay();
  }, 150);
}

function pct(v) { return Math.round(v * 100) + '%'; }

function addLog(msg, cls='') {
  logLines.unshift({msg, cls});
  if (logLines.length > 80) logLines.pop();
}

function render(d) {
  document.getElementById('mReward').textContent    = d.reward.toFixed(1);
  document.getElementById('mCES').textContent       = d.ces !== null ? d.ces.toFixed(3) : '—';
  document.getElementById('mStep').textContent      = d.step;
  document.getElementById('mConflicts').textContent = d.conflicts;

  const h = d.human;
  document.getElementById('bSpd').style.width  = pct(h.speed);
  document.getElementById('vSpd').textContent  = h.speed.toFixed(2);
  document.getElementById('bFtg').style.width  = pct(h.fatigue);
  document.getElementById('vFtg').textContent  = h.fatigue.toFixed(2);
  document.getElementById('bErr').style.width  = pct(h.error_rate);
  document.getElementById('vErr').textContent  = h.error_rate.toFixed(2);
  document.getElementById('vHDone').textContent = h.tasks_done;
  document.getElementById('vHTask').textContent = h.current_task !== null
    ? d.tasks[h.current_task]?.label || ('task '+h.current_task) : 'idle';

  document.getElementById('profBadge').textContent  = d.profile;
  document.getElementById('vAITask').textContent     = d.ai.current_task !== null
    ? d.tasks[d.ai.current_task]?.label || ('task '+d.ai.current_task) : 'idle';
  document.getElementById('vRoleSw').textContent = d.role_switches;
  document.getElementById('vSync').textContent   = d.sync;

  // Task list
  const list = document.getElementById('taskList');
  list.innerHTML = '';
  d.tasks.forEach(t => {
    const isActive = t.status === 1;
    const isDone   = t.status === 2;
    const isLocked = t.status === 0 && !t.prereqs_met;
    let cls = '';
    if (isActive) cls = 'active';
    else if (isDone) cls = 'done';
    else if (isLocked) cls = 'locked';

    let pillHtml = '<span class="pill pill-pending">pending</span>';
    if (isActive && t.assignee === 'human') pillHtml = '<span class="pill pill-human">human</span>';
    if (isActive && t.assignee === 'ai')    pillHtml = '<span class="pill pill-ai">AI agent</span>';
    if (isDone)                             pillHtml = '<span class="pill pill-done">done</span>';

    const prereqTxt = t.prereqs.length
      ? (t.prereqs_met || t.status > 0 ? '' : 'needs #' + t.prereqs.map(p=>p+1).join(', #'))
      : 'ready to start';
    const subtxt = isDone ? 'completed' : (isActive ? 'in progress' : prereqTxt);

    const row = document.createElement('div');
    row.className = 'task-row ' + cls;
    row.innerHTML = `
      <div class="num">${t.id+1}</div>
      <div>
        <div class="t-name">${t.label}</div>
        <div class="t-sub">difficulty ${t.diff.toFixed(1)} &middot; ${subtxt}</div>
      </div>
      <div>${pillHtml}</div>
      <div class="prog-wrap"><div class="prog-bar" style="width:${Math.round(t.progress*100)}%"></div></div>
    `;
    list.appendChild(row);
  });

  // Log events
  if (d.done && d.step > 0) addLog('All tasks complete!', 'ok');
  if (d.conflicts > 0) addLog('[' + String(d.step).padStart(3,'0') + '] conflict detected', 'bad');
  const logEl = document.getElementById('logBox');
  logEl.innerHTML = logLines.map(l =>
    `<div class="log-line ${l.cls}">${l.msg}</div>`
  ).join('');

  // Status bar
  const bar = document.getElementById('statusBar');
  if (d.done) {
    const all = d.tasks.every(t => t.status === 2);
    bar.className = 'status-bar done-bar';
    bar.textContent = all
      ? `All 6 tasks completed in ${d.step} steps — CES: ${d.ces}`
      : `Episode ended (max steps reached) — CES: ${d.ces}`;
    document.getElementById('playBtn').disabled = true;
    document.getElementById('stepBtn').disabled = true;
  } else {
    bar.className = 'status-bar';
    bar.textContent = `Step ${d.step} — human profile: ${d.profile} — ${d.tasks.filter(t=>t.status===2).length}/6 tasks done`;
    document.getElementById('playBtn').disabled = false;
    document.getElementById('stepBtn').disabled = false;
  }
}

doReset();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    reset_session()
    print("\n" + "="*55)
    print("  DTS-ZSC Showcase UI")
    print("  Open in your browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
