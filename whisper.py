import openai
from pydub import AudioSegment
import pandas as pd
from transformers import GPT2TokenizerFast
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# audio_file = open("audio/cs50-1.mp3", "rb")

# transcript = openai.Audio.transcribe(
#     "whisper-1", audio_file, params={"response_format": "text"}
# )

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))


lect = AudioSegment.from_file("audio/sysdesign.mp4")

one_min = 60 * 1000
lect_duration = lect.duration_seconds * 1000
segments = int(lect_duration // one_min)

for counter in range(segments):
    start = counter * one_min
    end = counter * one_min + one_min
    if end > lect_duration:
        end = lect_duration
    lect[start:end].export("audio/sysdesign-min{}.mp3".format(counter), format="mp3")
    counter += 1

segmented_transcripts = {"content": [], "tokens": []}

for i in range(segments):
    audio_file = open("audio/sysdesign-min{}.mp3".format(i), "rb")
    transcript = openai.Audio.transcribe(
        "whisper-1", audio_file, params={"response_format": "text"}
    )
    text = "Minute {}: ".format(i) + transcript["text"]
    segmented_transcripts["content"].append(text)
    segmented_transcripts["tokens"].append(count_tokens(text))

df = pd.DataFrame(data=segmented_transcripts)

df.to_csv("cs50-1.csv", index=False)
