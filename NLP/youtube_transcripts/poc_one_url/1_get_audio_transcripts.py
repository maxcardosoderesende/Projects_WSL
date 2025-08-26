import subprocess
import whisper
from loguru import logger

import warnings
# Suprimir warnings do Whisper
warnings.filterwarnings("ignore", category=UserWarning)


# 1. Baixar áudio usando yt-dlp
video_url = "https://www.youtube.com/shorts/0zvyfMW84GA"
audio_file = "audio.mp3"
logger.info("📥 Baixando áudio do YouTube...")
subprocess.run([
    "yt-dlp", "-f", "bestaudio", "--extract-audio", "--audio-format", "mp3",
    "-o", audio_file, video_url
])

# 2. Carregar modelo Whisper
logger.info("🤖 Carregando modelo Whisper...")
model = whisper.load_model("small")

# 3. Transcrever áudio
logger.info("📝 Transcrevendo áudio...")
result = model.transcribe(audio_file, language="pt")

# 4. Salvar transcrição em .txt
txt_file = "transcricao.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write(result["text"])
logger.info(f"✅ Transcrição salva em {txt_file}")
