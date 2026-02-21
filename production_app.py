"""
Amazing Image Identifier - Render Production Version
Uses Hugging Face Inference API so no ML models are loaded into memory.
Requires HF_API_TOKEN environment variable (free HF account).
"""

import os
import threading
from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import base64
import json
import time
import logging
import hashlib
import sqlite3
from PIL import Image, ImageDraw
import colorsys
from collections import Counter
import re
import requests as req_lib
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# HF Inference API config
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '')
HAS_API = bool(HF_API_TOKEN)
hf_client = InferenceClient(token=HF_API_TOKEN) if HAS_API else None

if not HAS_API:
    logger.warning("HF_API_TOKEN not set — AI features will be disabled.")

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
CORS(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  processing_time REAL,
                  caption TEXT,
                  objects_detected TEXT,
                  colors TEXT,
                  has_text BOOLEAN,
                  file_hash TEXT UNIQUE)''')
    conn.commit()
    conn.close()

init_db()


# ── File validation ───────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_file(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except Exception:
        return False

def validate_magic_bytes(file):
    header = file.read(8)
    file.seek(0)
    return header.startswith(b'\xff\xd8\xff') or \
           header.startswith(b'\x89PNG\r\n\x1a\n')

def sanitize_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\-.]', '_', filename)
    return secure_filename(filename)

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── HF Inference API ──────────────────────────────────────────────────────────

def generate_caption(image_bytes):
    if not HAS_API:
        return "AI unavailable — HF_API_TOKEN not configured."
    try:
        result = hf_client.image_to_text(image_bytes, model="Salesforce/blip-image-captioning-base")
        logger.info(f"Caption result: {result}")
        # result is a string directly from InferenceClient
        if isinstance(result, str) and result:
            return result
        # or a list of dicts
        if isinstance(result, list) and result:
            return result[0].get('generated_text', 'No caption generated.')
        return 'No caption generated.'
    except Exception as e:
        logger.error(f"Caption error: {e}")
        return 'Could not generate caption.'

def detect_objects(image_bytes):
    if not HAS_API:
        return []
    try:
        result = hf_client.object_detection(image_bytes, model="facebook/detr-resnet-50")
        logger.info(f"Detection result: {result}")
        objects = []
        for item in result:
            score = getattr(item, 'score', None) or item.get('score', 0) if isinstance(item, dict) else item.score
            label = getattr(item, 'label', None) or item.get('label', 'unknown') if isinstance(item, dict) else item.label
            box   = getattr(item, 'box',   None) or item.get('box',   {})         if isinstance(item, dict) else item.box
            if score < 0.5:
                continue
            xmin = getattr(box, 'xmin', None) or (box.get('xmin', 0) if isinstance(box, dict) else 0)
            ymin = getattr(box, 'ymin', None) or (box.get('ymin', 0) if isinstance(box, dict) else 0)
            xmax = getattr(box, 'xmax', None) or (box.get('xmax', 0) if isinstance(box, dict) else 0)
            ymax = getattr(box, 'ymax', None) or (box.get('ymax', 0) if isinstance(box, dict) else 0)
            objects.append({
                'label':      label,
                'confidence': round(float(score), 2),
                'box':        [int(xmin), int(ymin), int(xmax), int(ymax)]
            })
        return objects[:10]
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return []


# ── Local image processing (lightweight, no torch) ────────────────────────────

def extract_dominant_colors(image_path, num_colors=5):
    try:
        img = Image.open(image_path).resize((150, 150)).convert('RGB')
        color_counts = Counter(img.getdata())
        color_names = []
        for (r, g, b), _ in color_counts.most_common(num_colors):
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            if   v < 0.2:               name = "black"
            elif s < 0.1:               name = "gray" if v < 0.9 else "white"
            elif h < 0.05 or h > 0.95: name = "red"
            elif h < 0.15:              name = "orange"
            elif h < 0.25:              name = "yellow"
            elif h < 0.45:              name = "green"
            elif h < 0.55:              name = "cyan"
            elif h < 0.7:               name = "blue"
            elif h < 0.85:              name = "purple"
            else:                       name = "pink"
            if name not in color_names:
                color_names.append(name)
        return color_names[:5]
    except Exception as e:
        logger.error(f"Color extraction error: {e}")
        return []

def perform_ocr(_):
    return {"has_text": False, "text": ""}

def draw_bounding_boxes(image_path, objects):
    try:
        img  = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        colors = {'person':'#3498db','vehicle':'#e74c3c','animal':'#2ecc71',
                  'furniture':'#f39c12','electronics':'#9b59b6','default':'#95a5a6'}
        for obj in objects:
            box   = obj['box']
            label = obj['label']
            color = next((colors[k] for k in colors if k in label.lower()), colors['default'])
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], max(0, box[1]-20)),
                      f"{label} {obj['confidence']*100:.0f}%", fill=color)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Bounding box error: {e}")
        return None


# ── History helpers ───────────────────────────────────────────────────────────

def save_to_history(filename, processing_time, caption, objects, colors, has_text, file_hash):
    try:
        conn = sqlite3.connect('images.db')
        conn.cursor().execute(
            '''INSERT OR REPLACE INTO images
               (filename, processing_time, caption, objects_detected, colors, has_text, file_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (filename, processing_time, caption,
             json.dumps(objects), json.dumps(colors), has_text, file_hash))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"History save error: {e}")

def get_history(limit=10):
    try:
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        c.execute('''SELECT filename, upload_time, processing_time, caption,
                            objects_detected, colors
                     FROM images ORDER BY upload_time DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        return [{'filename': r[0], 'upload_time': r[1], 'processing_time': r[2],
                 'caption': r[3],
                 'objects': json.loads(r[4]) if r[4] else [],
                 'colors':  json.loads(r[5]) if r[5] else []} for r in rows]
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return []

def clear_history():
    try:
        conn = sqlite3.connect('images.db')
        conn.cursor().execute('DELETE FROM images')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"History clear error: {e}")
        return False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only JPG and PNG are allowed.'}), 400
    if not validate_magic_bytes(file):
        return jsonify({'error': 'Invalid image header'}), 400

    filepath = None
    try:
        filename = sanitize_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if not validate_image_file(filepath):
            return jsonify({'error': 'File is not a valid image'}), 400

        file_hash = get_file_hash(filepath)

        # Read image bytes once — used for HF API calls AND base64 response.
        # Must happen before finally block deletes the file.
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        original_image = base64.b64encode(image_bytes).decode('utf-8')

        # Run caption + detection in parallel, passing bytes (not file path)
        caption_box = [None]
        objects_box = [[]]
        def do_caption(): caption_box[0] = generate_caption(image_bytes)
        def do_detect():  objects_box[0] = detect_objects(image_bytes)
        t1 = threading.Thread(target=do_caption)
        t2 = threading.Thread(target=do_detect)
        t1.start(); t2.start()
        t1.join();  t2.join()

        caption = caption_box[0]
        objects = objects_box[0]
        colors  = extract_dominant_colors(filepath)
        ocr     = perform_ocr(filepath)
        annotated_image = draw_bounding_boxes(filepath, objects)

        processing_time = round(time.time() - start_time, 2)
        save_to_history(filename, processing_time, caption, objects,
                        colors, ocr['has_text'], file_hash)

        return jsonify({
            'success':        True,
            'filename':       filename,
            'caption':        caption,
            'objects':        objects,
            'object_count':   len(objects),
            'colors':         colors,
            'ocr':            ocr,
            'processing_time': processing_time,
            'original_image':  f"data:image/jpeg;base64,{original_image}",
            'annotated_image': f"data:image/png;base64,{annotated_image}" if annotated_image else None
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)  # CR-49

@app.route('/download/<fmt>', methods=['POST'])
def download_results(fmt):
    try:
        data = request.json
        if fmt == 'txt':
            lines  = ["Image Analysis Results", "=====================", "",
                      f"Caption: {data.get('caption','N/A')}", "",
                      f"Objects Detected ({data.get('object_count',0)}):"]
            lines += [f"  - {o['label']}: {o['confidence']*100:.1f}%"
                      for o in data.get('objects', [])]
            lines += ["", f"Colors: {', '.join(data.get('colors',[]))}",
                      f"Processing Time: {data.get('processing_time',0)}s"]
            return send_file(io.BytesIO('\n'.join(lines).encode()),
                             mimetype='text/plain', as_attachment=True,
                             download_name='analysis.txt')
        elif fmt == 'json':
            return send_file(io.BytesIO(json.dumps(data, indent=2).encode()),
                             mimetype='application/json', as_attachment=True,
                             download_name='analysis.json')
        return jsonify({'error': 'Invalid format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    limit = request.args.get('limit', 10, type=int)
    return jsonify({'success': True, 'history': get_history(limit)})

@app.route('/history/clear', methods=['POST'])
def clear_history_route():
    ok = clear_history()
    return jsonify({'success': True} if ok else {'error': 'Failed'}), (200 if ok else 500)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'ai_enabled': HAS_API,
                    'mode': 'huggingface_inference_api'})

@app.route('/credits')
def credits():
    return render_template('credits.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 10MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
