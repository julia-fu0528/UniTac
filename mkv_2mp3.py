# import ffmpeg

# input_file = "screen.mkv"
# output_file = "screen.mp3"

# ffmpeg.input(input_file).output(output_file, format='mp3', audio_bitrate='192k').run()

from pydub import AudioSegment
from pydub.utils import which

# Set the correct paths manually
AudioSegment.converter = which("ffmpeg") or "/usr/local/bin/ffmpeg"
AudioSegment.ffprobe = which("ffprobe") or "/usr/local/bin/ffprobe"

# Convert MKV to MP3
input_file = "screen.mkv"
output_file = "screen.mp3"

audio = AudioSegment.from_file(input_file, format="mkv")
audio.export(output_file, format="mp3")

print("Conversion completed!")
