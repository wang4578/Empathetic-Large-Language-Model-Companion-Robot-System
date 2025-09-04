from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import logging
import argparse
import json
import os
import torch
import shlex
import time
import subprocess
import watchdog.observers
import watchdog.events
import logging
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
import numpy as np
import datetime
import re
import skimage.measure
import whisper_at
from whisper.model import Whisper, ModelDimensions


# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 常量定义
UPLOAD_FOLDER = '/root/ltu-main/ltu-main/src/ltu_as/webpage/upload'
DEFAULT_QUESTION = 'What can be inferred from the spoken text and sounds? Why?'

# 全局变量初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# 加载模型和配置
def load_models():
    logger.info("开始加载模型...")
    
    # 加载whisper模型
    def convert_params_to_float32(model):
        for name, param in model.named_parameters():
            if "audio_encoder" in name and "ln" in name:
                if param.dtype == torch.float16:
                    logger.info(f"Converting parameter '{name}' to float32")
                    param.data = param.data.float()

    # 加载whisper-at模型用于转录
    whisper_text_model = whisper_at.load_model("/root/ltu-main/ltu-main/large-v2.pt", device=device, in_memory=True)
    logger.info("whisper-at模型加载完成")

    # 加载whisper模型用于特征提取
    def load_whisper():
        mdl_size = 'large-v1'
        checkpoint_path = '/root/ltu-main/ltu-main/pretrained_mdls/large-v1.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        dims = ModelDimensions(**checkpoint["dims"])
        whisper_feat_model = Whisper(dims)
        whisper_feat_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        whisper_feat_model.to('cuda:0')
        return whisper_feat_model
    
    whisper_feat_model = load_whisper()


    # 加载LLM模型
    base_model = "../../pretrained_mdls/vicuna_ltuas/"
    prompt_template = "alpaca_short"
    eval_mdl_path = '/root/ltu-main/ltu-main/src/ltu_as/exp/ltuas_ft_meld/checkpoint-33/pytorch_model.bin'
    #eval_mdl_path = '/root/ltu-main/ltu-main/pretrained_mdls/ltuas_long_noqa_a6.bin'
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    
    if device == 'cuda':
        model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
    else:
        model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")
    
    convert_params_to_float32(model)
    logger.info("LLaMA模型加载完成")
    
    # 配置LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    # 加载微调权重
    state_dict = torch.load(eval_mdl_path, map_location='cpu')
    miss, unexpect = model.load_state_dict(state_dict, strict=False)
    logger.info("LoRA权重加载完成")

    model.is_parallelizable = True
    model.model_parallel = True

    # 设置token ID
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    logger.info("所有模型加载完成")

    # 生成推理日志路径
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_save_path = './inference_log/'
    if not os.path.exists(log_save_path):
        os.mkdir(log_save_path)
    log_save_path = os.path.join(log_save_path, f"{cur_time}.json")
    
    return whisper_text_model, whisper_feat_model, model, tokenizer, prompter, log_save_path

# 文本处理函数
def remove_thanks_for_watching(text):
    variations = [
        "thanks for watching", "Thanks for watching", "THANKS FOR WATCHING",
        "thanks for watching.", "Thanks for watching.", "THANKS FOR WATCHING.",
        "thanks for watching!", "Thanks for watching!", "THANKS FOR WATCHING!",
        "thank you for watching", "Thank you for watching", "THANK YOU FOR WATCHING",
        "thank you for watching.", "Thank you for watching.", "THANK YOU FOR WATCHING.",
        "thank you for watching!", "Thank you for watching!", "THANK YOU FOR WATCHING!"
    ]
    variations = sorted(variations, key=len, reverse=True)
    pattern = "|".join(re.escape(var) for var in variations)
    result = re.sub(pattern, "", text)
    return result

# 音频转录和特征提取
text_cache = {}
def load_audio_trans(filename, whisper_text_model, whisper_feat_model):
    global text_cache
    if filename not in text_cache:
        result = whisper_text_model.transcribe(filename)
        logger.info(f"音频转录结果: {result}")
        text = remove_thanks_for_watching(result["text"].lstrip())
        text_cache[filename] = text
    else:
        text = text_cache[filename]
        logger.info('使用ASR缓存')
    
    _, audio_feat = whisper_feat_model.transcribe_audio(filename)
    audio_feat = audio_feat[0]
    audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
    audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
    audio_feat = audio_feat[1:]  # skip the first layer
    audio_feat = torch.FloatTensor(audio_feat)
    return audio_feat, text

# 截取回复部分
def trim_string(a):
    separator = "### Response:\n"
    trimmed_string = a.partition(separator)[-1]
    trimmed_string = trimmed_string.strip()
    return trimmed_string

# 推理函数
def predict(audio_path,  models,cur_input):
    whisper_text_model, whisper_feat_model, model, tokenizer, prompter, log_save_path = models
    
    eval_log = []
    if os.path.exists(log_save_path):
        try:
            with open(log_save_path, 'r') as f:
                eval_log = json.load(f)
        except:
            eval_log = []
    
    temp, top_p, top_k = 0.1, 0.95, 500
    
    logger.info(f'处理音频: {audio_path}')
    begin_time = time.time()

    if audio_path is not None:
        cur_audio_input, cur_input = load_audio_trans(audio_path, whisper_text_model, whisper_feat_model)
        if torch.cuda.is_available():
            cur_audio_input = cur_audio_input.unsqueeze(0).half().to(device)
    question="You are an emotional dialogue robot. Please provide the most appropriate response based on this information and the dialogue content."
    instruction = question
    prompt = prompter.generate_prompt(instruction, cur_input)
    logger.info(f'输入提示: {prompt}')
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        max_new_tokens=500,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # 生成回复
    with torch.no_grad():
        logger.info(f"输入形状: {input_ids.shape}, 音频特征形状: {cur_audio_input.shape}")
        generation_output = model.generate(
            input_ids=input_ids,
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=500,
        )
    
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output[5:-4]  # 去除特殊标记
    end_time = time.time()
    
    trimmed_output = trim_string(output)
    logger.info(f"模型回复: {trimmed_output}")
    
    # 记录结果
    cur_res = {
        'audio_id': audio_path, 
        'instruction': instruction, 
        'input': cur_input, 
        'output': trimmed_output
    }
    eval_log.append(cur_res)
    
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    
    logger.info(f'推理耗时: {end_time-begin_time} 秒')

    
    return trimmed_output

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

models = load_models()




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


@app.route('/asr', methods=['POST'])
def asr():
    whisper_text_model, whisper_feat_model, model, tokenizer, prompter, log_save_path = models
    file =  request.json
    filename=file['audio']
    print(filename)
    
    audio_save_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # 只做ASR转文本
    cur_audio_input,asr_text  = load_audio_trans(audio_save_path, whisper_text_model, whisper_feat_model)  # 模型里前两个就是whisper等
    return jsonify({'asr': asr_text, 'audio_path': audio_save_path})


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
        
@app.route('/llm',methods=['POST'])
def llm():
    data = request.json
    print(data)
    # 来自前端的音频路径或特征和asr文本
    asr_text = data['asr']
    audio_path = data['audio_path']
    
    
    llm_text = predict(audio_path, models,asr_text)
    return jsonify({'llm': llm_text})

@app.route('/tts', methods=['POST'])
def tts():
    file =  request.json
    output=file['response']
    #output='Hello, I am an emotional robot. How may I assist you?'
    # 生成语音回复
    safe_text = shlex.quote(output)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "/root/ltu-main/ltu-main/sounds_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wav_filename = f"{output_dir}{timestamp}_en_vits.wav"
    output_filename = f"/sounds_output/{timestamp}_en_vits.wav"


    conda_env = 'hbyenv'
    speaker_path = "/root/ltu-main/ltu-main/src/ltu_as/gentlevoice.wav"
    command = f'conda run -n {conda_env} python run_vc.py {safe_text} "{wav_filename}" "{speaker_path}"'
    #command = f'conda run -n {conda_env} tts --text {safe_text} --model_name "tts_models/en/ljspeech/vits" --out_path {wav_filename}'
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f"TTS语音生成成功: {wav_filename}")
    except subprocess.CalledProcessError as e:
        logger.error(f"TTS生成失败: {e}")
    return jsonify({'tts_url': output_filename})

@app.route('/sounds_output/<filename>')
def serve_audio(filename):
    return send_from_directory('/root/ltu-main/ltu-main/sounds_output', filename)

if __name__ == '__main__':
    # 启动前创建上传目录
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f'启动服务器在 http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port, debug=True)
