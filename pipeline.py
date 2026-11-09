

import cv2
import numpy as np
import json
import os
import sys
import time
import threading
import webbrowser
import argparse
import logging
from collections import defaultdict, deque
from datetime import datetime

from ultralytics import YOLO




# --- PPE (Eq. 1.1 weights w_i) ---
PPE_WEIGHTS       = {"helmet": 0.6, "vest": 0.4}
ASSOC_THRESHOLDS  = {"helmet": 0.50, "vest": 0.50}   
MIN_COVERAGE      = 0.50                              
MIN_PERSON_HEIGHT = 80                                
GLOBAL_PPE_REQUIRED = ["helmet", "vest"]             
DEFAULT_ZONE_PPE    = ["helmet", "vest"]             


PPE_CONFIRMATION_S = 1.0   # seconds an item must hold a new state before committing

# --- Inactivity (§5.3.3.1, UC9). Research basis:
#       productivity idleness 15–30 s (CN122867059A)
#       industrial stationary warning 45 s / 90 s (CN122015951A)
#       60 s chosen as a safety-monitoring middle ground. ---
INACTIVITY_RADIUS_PX   = 50
INACTIVITY_DURATION_S  = 60.0
INACTIVITY_WINDOW_S    = 60.0   # rolling window kept per track

# --- Zone geometry ---
ZONE_RELOAD_INTERVAL_S = 2.0    # NFR6: changes take effect without restart

# --- Output ---
OBSERVATION_INTERVAL_S = 1.0    # one record per track per second (§5.3.2 time window)
ZONES_FILE             = "zones.json"
OBSERVATIONS_FILE      = "observations.jsonl"
LATEST_FILE            = "latest.json"
SNAPSHOT_FILE          = "snapshot.jpg"

# --- Zone editor server ---
EDITOR_PORT = 5001

# --- Colours (BGR) ---
C_GREEN  = (0, 200, 0)
C_RED    = (0, 0, 255)
C_AMBER  = (0, 165, 255)
C_GRAY   = (140, 140, 140)
C_CYAN   = (255, 255, 0)
C_WHITE  = (255, 255, 255)


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test. polygon = [[x,y], ...]."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Euclidean distance from a point to a line segment."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return float(np.hypot(px - x1, py - y1))
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))


def distance_to_polygon(px, py, polygon):
    """0 if inside; otherwise minimum distance to any polygon edge."""
    if len(polygon) < 3:
        return float("inf")
    if point_in_polygon(px, py, polygon):
        return 0.0
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < best:
            best = d
    return best


# =============================================================================
# ZONE MANAGER
# =============================================================================

class ZoneManager:
    """
    Loads zones.json (written by the browser editor) and answers two questions
    per worker: which zone am I in, and how far is the nearest zone boundary.

    zones.json format:
    {
      "version": 1,
      "cameras": {
        "CAM01": {
          "pixel_to_meter_scale": null,
          "zones": [
            {"zone_id": "Z1", "name": "Machinery A", "zone_type": "machinery",
             "ppe_required": ["helmet","vest"],
             "polygon": [[120,200],[400,200],[400,480],[120,480]]}
          ]
        }
      }
    }
    """

    def __init__(self, path, camera_id):
        self.path = path
        self.camera_id = camera_id
        self._mtime = None
        self.zones = []
        self.pixel_to_meter_scale = None
        self.reload(force=True)

    def reload(self, force=False):
        if not os.path.exists(self.path):
            if force:
                self.zones = []
                self.pixel_to_meter_scale = None
            return
        try:
            mtime = os.path.getmtime(self.path)
            if not force and mtime == self._mtime:
                return
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cam = (data.get("cameras") or {}).get(self.camera_id, {})
            self.zones = cam.get("zones", []) or []
            self.pixel_to_meter_scale = cam.get("pixel_to_meter_scale", None)
            self._mtime = mtime
        except Exception as e:
            print(f"[ZoneManager] Could not load {self.path}: {e}")

    def _check_reload(self):
        if not os.path.exists(self.path):
            return
        try:
            if os.path.getmtime(self.path) != self._mtime:
                self.reload()
        except OSError:
            pass

    def find_zone(self, px, py):
        """Return the zone dict the point is inside, else None (zones do not overlap)."""
        self._check_reload()
        for z in self.zones:
            if point_in_polygon(px, py, z.get("polygon", [])):
                return z
        return None

    def nearest_proximity_px(self, px, py):
        """Minimum distance to any zone boundary, in pixels. inf if no zones."""
        self._check_reload()
        if not self.zones:
            return float("inf")
        return min(distance_to_polygon(px, py, z.get("polygon", [])) for z in self.zones)

    def proximity_m(self, px):
        if self.pixel_to_meter_scale is None or px == float("inf"):
            return None
        return round(px * float(self.pixel_to_meter_scale), 2)


# =============================================================================
# INACTIVITY TRACKER
# =============================================================================

class InactivityTracker:
    """
    Per-track rolling window of foot positions. A worker is 'inactive' when the
    set of positions in the window fits inside a circle of radius R for at
    least D seconds (spatial extent, not path length — a worker pacing in a
    tight circle is still 'stationary' in the safety sense).
    """

    def __init__(self, radius_px, duration_s, window_s):
        self.radius = radius_px
        self.duration = duration_s
        self.window = window_s
        self._buf = defaultdict(deque)   # track_id -> deque[(t, x, y)]

    def update(self, track_id, x, y, t):
        buf = self._buf[track_id]
        buf.append((t, x, y))
        while buf and t - buf[0][0] > self.window:
            buf.popleft()

        if len(buf) < 2:
            return False, 0.0

        # Spatial extent: max distance between any two points in the window.
        xs = [p[1] for p in buf]
        ys = [p[2] for p in buf]
        extent = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))

        if extent > self.radius:
            # Moved; reset the clock by trimming to the current point.
            while len(buf) > 1:
                buf.popleft()
            return False, 0.0

        elapsed = t - buf[0][0]
        if elapsed >= self.duration:
            return True, round(elapsed, 1)
        return False, round(elapsed, 1)

    def reset(self, track_id=None):
        if track_id is None:
            self._buf.clear()
        else:
            self._buf.pop(track_id, None)


# =============================================================================
# PPE STATE MACHINE  (§5.3.3.1, Figure 5.3.3.1.2)
# =============================================================================

class PPEStateMachine:
    """
    Converts the raw per-frame association result into a confirmed status.

    Rules (from the report):
      - A single missed detection is not enough to declare 'missing'.
      - A worn/missing verdict is only assigned when the region is visible.
      - When the region becomes unclear, status returns to 'unknown' (after
        the same confirmation period) rather than holding stale evidence.
    """

    def __init__(self, confirmation_s):
        self.confirm = confirmation_s
        # key: (track_id, item_type) -> dict
        self._state = {}

    def update(self, track_id, item_type, raw_status, t):
        key = (track_id, item_type)
        st = self._state.get(key)
        if st is None:
            # First sighting: unknown until evidence accumulates.
            self._state[key] = {
                "current": "unknown", "pending": None, "pending_since": t
            }
            st = self._state[key]

        if raw_status == st["current"]:
            st["pending"] = None
            return st["current"]

        if st["pending"] != raw_status:
            st["pending"] = raw_status
            st["pending_since"] = t
            return st["current"]

        if t - st["pending_since"] >= self.confirm:
            st["current"] = raw_status
            st["pending"] = None
        return st["current"]

    def reset_track(self, track_id):
        for k in [k for k in self._state if k[0] == track_id]:
            del self._state[k]


# =============================================================================
# OBSERVATION WRITER
# =============================================================================

class ObservationWriter:
    """
    Appends to observations.jsonl (full history) and overwrites latest.json
    (live snapshot of current workers) on every observation tick.
    """

    def __init__(self, jsonl_path, latest_path, enabled=True):
        self.enabled = enabled
        self.jsonl_path = jsonl_path
        self.latest_path = latest_path
        self._fh = None
        if self.enabled:
            # Truncate previous run so demos start clean.
            self._fh = open(self.jsonl_path, "w", encoding="utf-8")
            print(f"[Writer] {jsonl_path} (appending) and {latest_path} (live)")

    def write(self, observations):
        if not self.enabled or not observations:
            return
        line = json.dumps(observations, ensure_ascii=False)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        try:
            with open(self.latest_path, "w", encoding="utf-8") as f:
                json.dump(observations, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def close(self):
        if self._fh:
            self._fh.close()


# =============================================================================
# ZONE EDITOR (Flask, runs in a daemon thread)
# =============================================================================

_EDITOR_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Define Zones</title>
<style>
body{font-family:system-ui,sans-serif;background:#1e1e1e;color:#eee;margin:0;padding:16px}
h2{margin:0 0 8px 0;font-size:18px}
p.hint{color:#aaa;font-size:13px;margin:4px 0 12px 0}
#wrap{position:relative;display:inline-block;border:1px solid #444}
canvas{display:block;max-width:100%;height:auto;cursor:crosshair}
button{background:#2d7dd2;color:#fff;border:0;padding:8px 14px;border-radius:4px;
       font-size:14px;cursor:pointer;margin:4px 4px 4px 0}
button.secondary{background:#555}
button.danger{background:#a33}
button:hover{filter:brightness(1.15)}
table{border-collapse:collapse;margin-top:12px;font-size:13px}
th,td{border:1px solid #444;padding:6px 10px;text-align:left}
select{background:#333;color:#eee;border:1px solid #555;padding:4px;border-radius:3px}
#status{margin-left:10px;font-size:13px;color:#8f8}
</style></head>
<body>
<h2>Define Hazard Zones — camera __CAMERA_ID__</h2>
<p class="hint">Click on the image to add polygon vertices. Close the polygon when
you have at least 3 points. Assign a type to each zone. Click Save when done.</p>

<div id="wrap"><canvas id="cv" width="__WIDTH__" height="__HEIGHT__"></canvas></div>

<div>
  <button id="closeBtn">Close Polygon</button>
  <button id="clearBtn" class="secondary">Clear All</button>
  <button id="saveBtn">Save Zones</button>
  <span id="status"></span>
</div>

<table id="tbl"><thead><tr><th>#</th><th>Type</th><th>Points</th><th></th></tr></thead>
<tbody></tbody></table>

<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const img = new Image();
const W = cv.width, H = cv.height;
let polygons = [];          // [{type, points:[[x,y],...]}]
let current = [];           // points being drawn
let mouse = null;

img.onload = redraw;
img.src = '/snapshot?t=' + Date.now();

function redraw() {
  ctx.clearRect(0, 0, W, H);
  if (img.complete) ctx.drawImage(img, 0, 0, W, H);

  polygons.forEach((p, i) => {
    const col = `hsl(${(i*67)%360},80%,55%)`;
    drawPoly(p.points, col, true);
    if (p.points.length) labelPoly(p.points, 'Z'+(i+1), col);
  });

  if (current.length) drawPoly(current, '#00e5ff', false);
  if (current.length && mouse) {
    ctx.beginPath();
    const last = current[current.length-1];
    ctx.moveTo(last[0], last[1]);
    ctx.lineTo(mouse[0], mouse[1]);
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawPoly(pts, colour, closed) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (closed) ctx.closePath();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2;
  ctx.stroke();
  if (closed) {
    ctx.fillStyle = colour.replace('55%','55%)').replace('hsl','hsla').replace(')',',0.15)');
    ctx.fill();
  }
  pts.forEach(p => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 3, 0, Math.PI*2);
    ctx.fillStyle = colour; ctx.fill();
  });
}

function labelPoly(pts, text, colour) {
  let cx=0, cy=0;
  pts.forEach(p=>{cx+=p[0];cy+=p[1];});
  cx/=pts.length; cy/=pts.length;
  ctx.font = 'bold 14px sans-serif';
  ctx.fillStyle = colour;
  ctx.fillText(text, cx-10, cy);
}

function toImageCoords(ev) {
  const r = cv.getBoundingClientRect();
  return [Math.round((ev.clientX-r.left) * (W / r.width)),
          Math.round((ev.clientY-r.top)  * (H / r.height))];
}

cv.addEventListener('click', ev => { current.push(toImageCoords(ev)); redraw(); });
cv.addEventListener('mousemove', ev => { mouse = toImageCoords(ev); redraw(); });

document.getElementById('closeBtn').onclick = () => {
  if (current.length < 3) { setStatus('Need at least 3 points'); return; }
  polygons.push({type: 'machinery', points: current.slice()});
  current = [];
  renderTable(); redraw(); setStatus('');
};

document.getElementById('clearBtn').onclick = () => {
  if (!confirm('Discard all polygons?')) return;
  polygons = []; current = []; renderTable(); redraw();
};

document.getElementById('saveBtn').onclick = async () => {
  const payload = {camera_id: '__CAMERA_ID__', zones: polygons};
  const r = await fetch('/save_zones', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const j = await r.json();
  setStatus(j.ok ? ('Saved ' + j.count + ' zones') : ('Error: ' + j.error));
};

function renderTable() {
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';
  polygons.forEach((p, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>Z${i+1}</td>
      <td><select data-i="${i}">
        <option value="machinery">machinery</option>
        <option value="restricted maintenance">restricted maintenance</option>
        <option value="unprotected edge">unprotected edge</option>
        <option value="elevated platform">elevated platform</option>
      </select></td>
      <td>${p.points.length} pts</td>
      <td><button class="danger" data-del="${i}">Delete</button></td>`;
    tb.appendChild(tr);
    const sel = tr.querySelector('select');
    sel.value = p.type;
    sel.onchange = e => { polygons[+e.target.dataset.i].type = e.target.value; };
    tr.querySelector('button').onclick = e => {
      polygons.splice(+e.target.dataset.del, 1);
      renderTable(); redraw();
    };
  });
}

function setStatus(s){ document.getElementById('status').textContent = s || ''; }
</script></body></html>
"""


class ZoneEditorServer:
    """
    Serves the snapshot + polygon editor and accepts POST /save_zones.
    Runs in a daemon thread so the OpenCV loop is never blocked.
    """

    def __init__(self, camera_id, zones_path, snapshot_path):
        self.camera_id = camera_id
        self.zones_path = zones_path
        self.snapshot_path = snapshot_path
        self._thread = None
        self._app = None
        self._lock = threading.Lock()

    def _build_app(self):
        from flask import Flask, request, Response, jsonify
        app = Flask(__name__)
        app.logger.setLevel(logging.ERROR)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.route("/")
        def index():
            with open(self.snapshot_path, "rb") as f:
                head = f.read(2)
            if head[:2] != b"\xff\xd8":
                return "No snapshot yet.", 404
            import struct
            # Read image dimensions cheaply with OpenCV
            img = cv2.imread(self.snapshot_path)
            if img is None:
                return "Snapshot unreadable.", 500
            h, w = img.shape[:2]
            html = (_EDITOR_HTML
                    .replace("__WIDTH__", str(w))
                    .replace("__HEIGHT__", str(h))
                    .replace("__CAMERA_ID__", self.camera_id))
            return Response(html, mimetype="text/html")

        @app.route("/snapshot")
        def snapshot():
            if not os.path.exists(self.snapshot_path):
                return "", 404
            return Response(open(self.snapshot_path, "rb").read(),
                            mimetype="image/jpeg")

        @app.route("/save_zones", methods=["POST"])
        def save_zones():
            try:
                body = request.get_json(force=True)
                raw = body.get("zones", [])
                cleaned = []
                for i, z in enumerate(raw):
                    pts = z.get("points", [])
                    if len(pts) < 3:
                        continue
                    cleaned.append({
                        "zone_id": f"Z{i+1}",
                        "name": f"Zone {i+1}",
                        "zone_type": z.get("type", "machinery"),
                        "ppe_required": DEFAULT_ZONE_PPE,
                        "polygon": [[int(p[0]), int(p[1])] for p in pts],
                    })
                data = {
                    "version": 1,
                    "cameras": {
                        self.camera_id: {
                            "pixel_to_meter_scale": None,
                            "zones": cleaned,
                        }
                    },
                }
                with self._lock:
                    with open(self.zones_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                return jsonify({"ok": True, "count": len(cleaned)})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 400

        return app

    def launch(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            print("[ZoneEditor] Flask not installed. Run: pip install flask")
            return
        if self._thread and self._thread.is_alive():
            webbrowser.open(f"http://127.0.0.1:{EDITOR_PORT}/")
            return
        self._app = self._build_app()
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                host="127.0.0.1", port=EDITOR_PORT,
                debug=False, use_reloader=False
            ),
            daemon=True,
        )
        self._thread.start()
        time.sleep(0.5)
        webbrowser.open(f"http://127.0.0.1:{EDITOR_PORT}/")


# =============================================================================
# ASSOCIATION  (Eq. 1.2 — unchanged logic from the working version)
# =============================================================================

def compute_containment(ppe_box, region):
    qx1, qy1, qx2, qy2 = ppe_box
    rx1, ry1, rx2, ry2 = region
    ix1, iy1 = max(qx1, rx1), max(qy1, ry1)
    ix2, iy2 = min(qx2, rx2), min(qy2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = (qx2 - qx1) * (qy2 - qy1)
    return inter / area if area > 0 else 0.0


def get_regions(person_box):
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    pw = px2 - px1
    return {
        "helmet": (px1, py1, px2, py1 + ph * 0.25),
        "vest":   (px1 + pw * 0.10, py1 + ph * 0.20,
                   px2 - pw * 0.10, py1 + ph * 0.75),
    }


def associate_ppe(people, ppe_detections):
    candidates = []
    for pbox, pid in people:
        if pbox[3] - pbox[1] < MIN_PERSON_HEIGHT:
            continue
        regions = get_regions(pbox)
        for idx, (qbox, qcls, conf) in enumerate(ppe_detections):
            if "hardhat" in qcls or "helmet" in qcls:
                ptype = "helmet"
            elif "vest" in qcls:
                ptype = "vest"
            else:
                continue
            c = compute_containment(qbox, regions[ptype])
            candidates.append((c * conf, c, pid, idx, qcls, ptype))

    candidates.sort(reverse=True)
    used_boxes, used_pt = set(), set()
    result = {pid: {} for _, pid in people}

    for _, c, pid, idx, qcls, ptype in candidates:
        if idx in used_boxes or (pid, ptype) in used_pt:
            continue
        if c < ASSOC_THRESHOLDS[ptype]:
            continue
        used_boxes.add(idx)
        used_pt.add((pid, ptype))
        result[pid][ptype] = {
            "status": "missing" if ("no-" in qcls or "no_" in qcls) else "worn",
            "class": qcls,
            "containment": round(c, 2),
            "confidence": None,      # filled from ppe_detections below
        }

    # attach confidence, and mark fully unassigned items as unknown
    for pid in result:
        for ptype in ("helmet", "vest"):
            if ptype not in result[pid]:
                result[pid][ptype] = {
                    "status": "unknown", "class": None,
                    "containment": 0.0, "confidence": None,
                }
    return result, used_boxes


# =============================================================================
# SCORE  (Eq. 1.1, 1.2, 1.3)
# =============================================================================

def compute_score(status_dict, required_items):
    if not required_items:
        return 0.0, 0.0, "Not Applicable"

    total_w = sum(PPE_WEIGHTS[i] for i in required_items)
    missing_w = 0.0
    known_w = 0.0
    for it in required_items:
        st = status_dict.get(it, {}).get("status", "unknown")
        if st == "missing":
            missing_w += PPE_WEIGHTS[it]
            known_w += PPE_WEIGHTS[it]
        elif st == "worn":
            known_w += PPE_WEIGHTS[it]

    if known_w == 0:
        return 0.0, 0.0, "Unavailable"

    coverage = known_w / total_w
    score = (missing_w / known_w) * 100.0
    status = "Unreliable" if coverage < MIN_COVERAGE else "Sufficient Coverage"
    return score, coverage, status


# =============================================================================
# PER-FRAME PROCESSING
# =============================================================================

class Pipeline:
    def __init__(self, camera_id, args):
        self.camera_id = camera_id
        self.args = args
        self.person_model = YOLO("yolo11n.pt")
        self.ppe_model = YOLO("best.pt")
        self.zones = ZoneManager(ZONES_FILE, camera_id)
        self.inactivity = InactivityTracker(
            INACTIVITY_RADIUS_PX, INACTIVITY_DURATION_S, INACTIVITY_WINDOW_S
        )
        self.ppe_sm = PPEStateMachine(PPE_CONFIRMATION_S)
        self.writer = ObservationWriter(
            OBSERVATIONS_FILE, LATEST_FILE, enabled=not args.no_write
        )
        self.zone_entry = {}          #tracking 
        self.last_obs_t = {}          
        self.last_frame_w = 0
        self.last_frame_h = 0

    # ------------------------------------------------------------------
    def _emit_observation(self, base, t):
        pid = base["temporary_id"]
        last = self.last_obs_t.get(pid, 0)
        if t - last < OBSERVATION_INTERVAL_S:
            return None
        self.last_obs_t[pid] = t
        base["observation_id"] = (
            f"{self.camera_id}-{pid}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        return base

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        fh, fw = frame.shape[:2]
        self.last_frame_w, self.last_frame_h = fw, fh
        t = time.time()

        # ---- 1. Person detection + tracking ----
        p_res = self.person_model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[0], conf=0.35, verbose=False
        )[0]
        people = [(b.xyxy[0].tolist(), int(b.id))
                  for b in p_res.boxes if b.id is not None]

        # ---- 2. PPE detection ----
        e_res = self.ppe_model.predict(frame, conf=0.15, verbose=False)[0]
        ppe_detections = []
        for b in e_res.boxes:
            cls = self.ppe_model.names[int(b.cls)].lower()
            if not any(k in cls for k in ("hardhat", "helmet", "vest")):
                continue
            ppe_detections.append((b.xyxy[0].tolist(), cls, float(b.conf)))

        # ---- 3. Association ----
        assoc, used_boxes = associate_ppe(people, ppe_detections)

        # ---- 4. Per-person: zones, inactivity, state machine, payload ----
        observations = []
        active_ids = set()

        for pbox, pid in people:
            active_ids.add(pid)
            px1, py1, px2, py2 = pbox
            foot_x, foot_y = (px1 + px2) / 2.0, py2

            # 4a. zone membership / proximity / time-in-zone
            zone = self.zones.find_zone(foot_x, foot_y)
            if zone:
                zone_id = zone.get("zone_id")
                zone_type = zone.get("zone_type")
                req = zone.get("ppe_required") or DEFAULT_ZONE_PPE
                key = (pid, zone_id)
                if key not in self.zone_entry:
                    self.zone_entry[key] = t
                time_in_zone = round(t - self.zone_entry[key], 1)
                # clear stale entries for this track
                for k in [k for k in self.zone_entry if k[0] == pid and k != key]:
                    del self.zone_entry[k]
            else:
                zone_id, zone_type = None, None
                req = GLOBAL_PPE_REQUIRED
                time_in_zone = 0.0

            prox_px = self.zones.nearest_proximity_px(foot_x, foot_y)

            # 4b. raw PPE statuses, with occlusion / small-person overrides
            raw = {}
            head_occluded = py1 < 5 or (py2 - py1) < MIN_PERSON_HEIGHT
            torso_occluded = (py2 > fh - 5) or (py2 - py1) < MIN_PERSON_HEIGHT
            for ptype in ("helmet", "vest"):
                if ptype == "helmet" and head_occluded:
                    raw[ptype] = "unknown"
                elif ptype == "vest" and torso_occluded:
                    raw[ptype] = "unknown"
                else:
                    raw[ptype] = assoc[pid][ptype]["status"]

            # 4c. confirmation period
            confirmed = {}
            for ptype in ("helmet", "vest"):
                confirmed[ptype] = self.ppe_sm.update(pid, ptype, raw[ptype], t)

            # 4d. score
            status_dict = {
                ptype: {
                    "status": confirmed[ptype],
                    "class": assoc[pid][ptype]["class"],
                    "containment": assoc[pid][ptype]["containment"],
                }
                for ptype in ("helmet", "vest")
            }
            score, coverage, score_status = compute_score(status_dict, req)

            # 4e. inactivity
            inactive, inactive_s = self.inactivity.update(pid, foot_x, foot_y, t)

            # 4f. build observation
            obs = {
                "camera_id": self.camera_id,
                "temporary_id": pid,
                "timestamp": now_iso(),
                "position_x": int(foot_x),
                "position_y": int(foot_y),
                "person_box": [int(v) for v in pbox],
                "zone_id": zone_id,
                "zone_type": zone_type,
                "hazard_proximity_px": (
                    None if prox_px == float("inf") else round(prox_px, 1)
                ),
                "hazard_proximity_m": self.zones.proximity_m(prox_px),
                "time_in_zone_s": time_in_zone,
                "ppe_status": {k: confirmed[k] for k in ("helmet", "vest")},
                "ppe_confidence": {
                    k: assoc[pid][k]["confidence"] for k in ("helmet", "vest")
                },
                "ppe_score": round(score, 2),
                "observation_coverage": round(coverage, 2),
                "score_status": score_status,
                "association": {
                    k: {
                        "class": assoc[pid][k]["class"],
                        "containment": assoc[pid][k]["containment"],
                    }
                    for k in ("helmet", "vest")
                },
                "inactivity": {"active": inactive, "duration_s": inactive_s},
            }

            emitted = self._emit_observation(obs, t)
            if emitted:
                observations.append(emitted)

            # 4g. draw
            self._draw_person(frame, pbox, pid, confirmed, score, inactive, obs)

        # ---- 5. cleanup gone tracks ----
        for gone in [p for p in list(self.last_obs_t) if p not in active_ids]:
            self.last_obs_t.pop(gone, None)
            self.ppe_sm.reset_track(gone)
            self.inactivity.reset(gone)
            for k in [k for k in self.zone_entry if k[0] == gone]:
                del self.zone_entry[k]

        # ---- 6. draw zones and PPE boxes ----
        self._draw_zones(frame)
        self._draw_ppe_boxes(frame, ppe_detections, used_boxes)

        # ---- 7. write ----
        if observations:
            self.writer.write(observations)

        return frame, observations

    # ------------------------------------------------------------------
    def _draw_person(self, frame, pbox, pid, confirmed, score, inactive, obs):
        px1, py1, px2, py2 = map(int, pbox)
        colour = C_RED if score > 0 else C_GREEN
        cv2.rectangle(frame, (px1, py1), (px2, py2), colour, 2)

        h = confirmed["helmet"][:1].upper()
        v = confirmed["vest"][:1].upper()
        label = f"ID:{pid}  score:{score:.0f}  H:{h}  V:{v}"
        if inactive:
            label += "  INACTIVE"
        cv2.putText(frame, label, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        if obs["zone_id"]:
            zl = f"{obs['zone_id']} ({obs['zone_type']})"
            cv2.putText(frame, zl, (px1, py2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CYAN, 1)

        if self.args.debug:
            for name, region in get_regions(pbox).items():
                rx1, ry1, rx2, ry2 = map(int, region)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), C_CYAN, 1)
                cv2.putText(frame, name, (rx1 + 2, ry1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_CYAN, 1)

    def _draw_zones(self, frame):
        for z in self.zones.zones:
            pts = np.array(z.get("polygon", []), dtype=np.int32)
            if len(pts) < 3:
                continue
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (60, 60, 180))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, C_CYAN, 2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.putText(frame, f"{z.get('zone_id')} {z.get('zone_type','')}",
                        (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)

    def _draw_ppe_boxes(self, frame, ppe_detections, used_boxes):
        for idx, (qbox, qcls, conf) in enumerate(ppe_detections):
            qx1, qy1, qx2, qy2 = map(int, qbox)
            if idx in used_boxes:
                c = C_RED if "no-" in qcls else C_GREEN
                thick = 2
            else:
                c, thick = C_GRAY, 1
            cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), c, thick)
            cv2.putText(frame, f"{qcls} {conf:.2f}", (qx1, qy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    def close(self):
        self.writer.close()


# =============================================================================
# UI — Define Zones button on the OpenCV window
# =============================================================================

_UI_STATE = {"button_rect": None, "clicked": False, "editor": None}


def _draw_button(frame):
    fh, fw = frame.shape[:2]
    bw, bh = 170, 38
    x1, y1 = fw - bw - 12, 12
    x2, y2 = x1 + bw, y1 + bh
    _UI_STATE["button_rect"] = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_WHITE, 2)
    cv2.putText(frame, "Define Zones", (x1 + 14, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2)


def _mouse_cb(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    rect = _UI_STATE["button_rect"]
    if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
        _UI_STATE["clicked"] = True


def _handle_zone_button(frame, editor):
    if not _UI_STATE["clicked"]:
        return
    _UI_STATE["clicked"] = False
    ok = cv2.imwrite(SNAPSHOT_FILE, frame)
    if not ok:
        print("[ZoneEditor] Could not write snapshot.")
        return
    print(f"[ZoneEditor] Snapshot saved -> {SNAPSHOT_FILE}")
    editor.launch()


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--camera-id", type=str, default="CAM01")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # Image mode
    if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(source)
        if img is None:
            print(f"Could not read image: {source}")
            sys.exit(1)
        pipe = Pipeline(args.camera_id, args)
        annotated, observations = pipe.process_frame(img)
        print(json.dumps(observations, indent=2))
        cv2.imshow("Layer 1", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pipe.close()
        return

    # Video / webcam mode
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        print("UC5 exceptional flow: camera unavailable. System will not start.")
        sys.exit(1)

    pipe = Pipeline(args.camera_id, args)
    editor = ZoneEditorServer(args.camera_id, ZONES_FILE, SNAPSHOT_FILE)

    win = "Layer 1 - Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb)

    print(f"[Layer 1] camera={args.camera_id}  source={source}")
    print("[Layer 1] 'q' = quit.  Click 'Define Zones' to open the browser editor.")

    consecutive_failures = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    # UC5 exceptional flow: camera unavailable
                    print(f"[UC5] Video source '{source}' unavailable. "
                          f"Interrupting inactivity timers and stopping.")
                    pipe.inactivity.reset()
                    break
                time.sleep(0.1)
                continue
            consecutive_failures = 0

            annotated, _ = pipe.process_frame(frame)
            _draw_button(annotated)
            _handle_zone_button(frame, editor)
            cv2.imshow(win, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipe.close()
        print("[Layer 1] stopped.")


if __name__ == "__main__":
    main()