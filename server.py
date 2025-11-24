import os, uuid, json, subprocess
from flask import Flask, request, jsonify, render_template, send_file
from analyze_separate_gui import analyze_mix_file

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200*1024*1024

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files.get('mix', None)
    if f is None:
        return jsonify({'error':'no mix file uploaded'}), 400
    filename = f.filename or ('mix_' + str(uuid.uuid4()) + '.wav')
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    try:
        result = analyze_mix_file(save_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
