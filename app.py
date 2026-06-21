"""
Local web app: drag-and-drop a MIDI file, then either use the era dropdown
or a chat box ("add in pedal", "play with more rubato") to control how the
trained model renders it as an expressive performance, played back in the
browser via the html-midi-player web component.

Run with: python app.py, then open http://localhost:5000
"""
import io
import os
import secrets
import tempfile
import traceback

from flask import Flask, jsonify, request, send_file, send_from_directory

from audio_render import render_pm_to_wav_bytes
from chat_control import DEFAULT_PARAMS, interpret_command
from infer import apply_adjustments, normalize_to_flat_midi, predict_raw
import library
from note_dataset import ERA_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# session_id -> {"file_path", "params", "raw" (cached model predictions, keyed by era)}
SESSIONS = {}


@app.errorhandler(Exception)
def handle_exception(e):
    # Without this, any unanticipated exception falls through to Flask's
    # HTML debug page, which breaks the frontend's res.json() calls with
    # "Unexpected token '<'". Print the real traceback to the console
    # (where debug=True would normally show it) and return JSON instead.
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/eras")
def eras():
    return jsonify(ERA_NAMES)


@app.route("/library/search")
def library_search():
    query = request.args.get("q", "")
    return jsonify(library.search(query))


@app.route("/library/load", methods=["POST"])
def library_load():
    data = request.get_json(force=True)
    entry = library.find(data.get("library_id"))
    if entry is None:
        return jsonify({"error": "unknown library item"}), 400

    # Render the piece in its own era by default (e.g. a Bach piece loads as
    # baroque, not the generic romantic default), matching the UI era tag.
    params = dict(DEFAULT_PARAMS)
    params["era"] = entry["era"]
    session_id = secrets.token_hex(8)
    SESSIONS[session_id] = {"file_path": entry["path"], "params": params, "raw_cache": {}}
    return jsonify({"session_id": session_id, "params": params})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    file = request.files["file"]

    os.makedirs(UPLOAD_DIR, exist_ok=True)  # in case it was removed after startup
    session_id = secrets.token_hex(8)
    ext = os.path.splitext(file.filename or "")[1].lower() or ".mid"
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
    file.save(file_path)

    SESSIONS[session_id] = {"file_path": file_path, "params": dict(DEFAULT_PARAMS), "raw_cache": {}}
    return jsonify({"session_id": session_id, "params": SESSIONS[session_id]["params"]})


@app.route("/normalized", methods=["POST"])
def normalized_route():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    session = SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "unknown session - upload a file first"}), 400
    if not os.path.exists(session["file_path"]):
        return jsonify({"error": "session expired (uploaded file is gone) - please re-upload"}), 400

    out_pm = normalize_to_flat_midi(session["file_path"])
    buf = io.BytesIO()
    out_pm.write(buf)
    buf.seek(0)
    return send_file(buf, mimetype="audio/midi", as_attachment=True, download_name="normalized.mid")


def _get_raw(session):
    era = session["params"]["era"]
    if era not in session["raw_cache"]:
        session["raw_cache"] = {era: predict_raw(session["file_path"], era, checkpoint_path=CHECKPOINT_PATH)}
    return session["raw_cache"][era]


@app.route("/render", methods=["POST"])
def render_route():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    session = SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "unknown session - upload a file first"}), 400
    if not os.path.exists(session["file_path"]):
        return jsonify({"error": "session expired (uploaded file is gone) - please re-upload"}), 400

    try:
        raw = _get_raw(session)
        out_pm = apply_adjustments(raw, session["params"])
        wav_bytes = render_pm_to_wav_bytes(out_pm)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    buf = io.BytesIO(wav_bytes)
    return send_file(buf, mimetype="audio/wav", as_attachment=True, download_name="rendered.wav")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    message = data.get("message", "")
    session = SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "unknown session - upload a file first"}), 400

    try:
        new_params, reply = interpret_command(message, session["params"])
    except Exception as e:
        return jsonify({"error": f"chat model error: {e}"}), 500

    session["params"] = new_params
    return jsonify({"reply": reply, "params": new_params})


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise SystemExit(f"no checkpoint at {CHECKPOINT_PATH} - train a model first")
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
