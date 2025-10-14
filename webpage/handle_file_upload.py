from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import logging
import argparse

# 配置命令行参数
parser = argparse.ArgumentParser(description='Flask Server for Audio Upload')
parser.add_argument('--port', type=int, default=5500, help='Port to run the server on')
args = parser.parse_args()

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [f"http://127.0.0.1:{args.port}"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# 配置参数
BASE_DIR = '/root/ltu-main/ltu-main/src/ltu_as/webpage'
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'upload')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mpeg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', f'http://127.0.0.1:{args.port}')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'project copy.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # 先尝试从上传目录获取文件
    if os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        return send_from_directory(UPLOAD_FOLDER, filename)
    # 如果不在上传目录，则从基础目录获取
    return send_from_directory(BASE_DIR, filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
        
    logger.info('收到请求: %s', request.method)
    logger.debug('请求头: %s', dict(request.headers))
    logger.debug('请求文件: %s', request.files)
    
    # 检查请求中是否有文件
    if 'audio' not in request.files:
        logger.error('未检测到文件字段')
        return jsonify({'error': '未检测到文件'}), 400
    
    file = request.files['audio']
    logger.info('文件名: %s', file.filename)
    
    # 验证文件名
    if file.filename == '':
        logger.error('无效文件名')
        return jsonify({'error': '无效文件名'}), 400
    
    # 验证文件类型
    if not allowed_file(file.filename):
        logger.error('不支持的文件类型: %s', file.filename)
        return jsonify({'error': '不支持的文件类型'}), 400
    
    try:
        # 安全处理文件名
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info('尝试保存文件到: %s', save_path)
        
        # 确保上传目录存在
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # 保存文件
        file.save(save_path)
        logger.info('文件保存成功')
        
        return jsonify({
            'status': 'success',
            'filepath': save_path,
            'filename': filename
        }), 200
        
    except Exception as e:
        logger.exception('文件保存失败')
        return jsonify({'error': f'文件保存失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 启动前创建上传目录
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f'启动服务器在 http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port, debug=True)