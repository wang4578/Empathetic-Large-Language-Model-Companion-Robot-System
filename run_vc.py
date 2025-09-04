import sys
from TTS.api import TTS

text = sys.argv[1]
output_path = sys.argv[2]
speaker_wav = sys.argv[3]

# 加载模型
tts = TTS("tts_models/en/ljspeech/vits")

# 克隆说话人并合成语音
tts.tts_with_vc_to_file(
    text,
    speaker_wav=speaker_wav,
    file_path=output_path
)
