# import os
# import sys
# import subprocess

# 1. Clone the repo if it doesn't exist
# if not os.path.exists("LuxTTS"):
#     subprocess.run(["git", "clone", "https://github.com/ysharma3501/LuxTTS.git"])

# 2. Install requirements
# subprocess.run([sys.executable, "-m", "pip", "install", "-r", "LuxTTS/requirements.txt"])

# 3. Add to path
# sys.path.append(os.path.abspath("LuxTTS"))

import re
import json
import numpy as np
import time
import gradio as gr
import torch
from zipvoice.luxvoice import LuxTTS

# Load multi-tone dictionary from JSON file
def load_multi_tone_dict(file_path):
    try:
        print(f"\033[31mLoading multi-tone dictionary from {file_path}...\033[0m")
        with open(file_path, 'r', encoding='utf-8') as file:
            gr.Info(f"Loaded multi-tone dictionary from {file_path}.")
            return json.load(file)
    except FileNotFoundError:
        gr.Warning(f"File {file_path} not found. Using empty dictionary.")
        return {}
    except json.JSONDecodeError:
        gr.Warning(f"Invalid JSON format in {file_path}. Using empty dictionary.")
        return {}

def save_multi_tone_dict(file_path, multi_tone_dict):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(multi_tone_dict, file, ensure_ascii=False, indent=4)

def replace_multi_tones(text, multi_tone_dict):
    """Helper function to replace multi-tone words with their pinyin representations."""
    for word, pinyin in multi_tone_dict.items():
        text = text.replace(word, pinyin)
    return text

def load_dict(file_path):
    if file_path:
        new_dict = load_multi_tone_dict(file_path)
        return json.dumps(new_dict, ensure_ascii=False, indent=4)

# Save Multi-Tone Dictionary
def save_dict(json_text, save_path):
    print("\033[31mSaving multi-tone dictionary...\033[0m")
    try:
        new_dict = json.loads(json_text)
        save_multi_tone_dict(save_path, new_dict)
        gr.Info(f"Dictionary saved successfully to {save_path}.")
        return f"Dictionary saved successfully to {save_path}!"
    except json.JSONDecodeError:
        gr.Warning("Invalid JSON format. Please check your input.")
        return "Invalid JSON format. Please check your input."

# Split text into segments at punctuation marks
def split_text_into_segments(text, max_length=100):
    """Helper function to split text into smaller segments at punctuation marks."""
    # Use regular expression to find punctuation marks
    segments = re.split(r'(?<=[。！？.!?])', text)
    # Combine segments to ensure each is within max_length
    combined_segments = []
    current_segment = ''
    for segment in segments:
        if len(current_segment) + len(segment) > max_length:
            combined_segments.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment
    if current_segment:
        combined_segments.append(current_segment)
    combined_segments = list(filter(lambda x: len(x) > 0, combined_segments))
    return combined_segments


def infer(
    text,
    audio_prompt,
    rms,
    ref_duration,
    t_shift,
    num_steps,
    speed,
    return_smooth,
    multi_tone_dict,
):
    if audio_prompt is None or not text:
        return None, "Please provide text and reference audio."

    start_time = time.time()

    # Replace multi-tone words
    text = replace_multi_tones(text, multi_tone_dict)

    # Encode reference (WITH duration)
    encoded_prompt = lux_tts.encode_prompt(
        audio_prompt,
        duration=ref_duration,
        rms=rms,
    )


    text_segments = split_text_into_segments(text)
    print(f"{len(text_segments)} text segments: {text_segments}")

    # Generate speech for each segment
    final_wav_segments = []
    for i, segment in enumerate(text_segments):
        print(f"Generating speech for segment {i+1}: {segment}")
        wav_segment = lux_tts.generate_speech(
            segment,
            encoded_prompt,
            num_steps=int(num_steps),
            t_shift=t_shift,
            speed=speed,
            return_smooth=return_smooth,
        )
        gr.Info(f"Generated segment {i+1}/{len(text_segments)}...")
        wav_segment = wav_segment.cpu().squeeze(0).numpy()
        wav_segment = (np.clip(wav_segment, -1.0, 1.0) * 32767).astype(np.int16)
        final_wav_segments.append(wav_segment)

    # Concatenate all segments
    final_wav = np.concatenate(final_wav_segments)

    duration = round(time.time() - start_time, 2)

    stats_msg = f"✨ Generation complete in **{duration}s**."
    return (48000, final_wav), stats_msg

# Init Model
device = "cuda" if torch.cuda.is_available() else "cpu"
lux_tts = LuxTTS("./models/LuxTTS", device=device, threads=2)
# =======================
# Gradio UI
# =======================
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ LuxTTS Voice Cloning")

    gr.Markdown(
        """
        > **Note:** This demo runs on a **2-core CPU**, so expect slower inference.  
        > **Tip:** If words get cut off, lower **Speed** or increase **Ref Duration**.
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Text to Synthesize",
                value="Hey, what's up? I'm feeling really great today!",
            )
            input_audio = gr.Audio(
                label="Reference Audio (.wav)",
                type="filepath",
            )

            with gr.Row():
                rms_val = gr.Number(
                    value=0.01,
                    label="RMS (Loudness)",
                )
                ref_duration_val = gr.Number(
                    value=5,
                    label="Reference Duration (sec)",
                    info="Lower = faster. Set ~1000 if you hear artifacts.",
                )
                t_shift_val = gr.Number(
                    value=0.9,
                    label="T-Shift",
                )

            with gr.Row():
                steps_val = gr.Slider(
                    1,
                    10,
                    value=4,
                    step=1,
                    label="Num Steps",
                )
                speed_val = gr.Slider(
                    0.5,
                    2.0,
                    value=0.8,
                    step=0.1,
                    label="Speed",
                )
                smooth_val = gr.Checkbox(
                    label="Return Smooth",
                    value=False,
                )

            btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            audio_out = gr.Audio(label="Result")
            status_text = gr.Markdown("Ready to generate...")

            with gr.Row():
                load_path_textbox = gr.Textbox(
                    label="Load Multi-Tone Dictionary",
                    value="./multi_tone_dict.json",
                )
                save_path_textbox = gr.Textbox(
                    label="Save Multi-Tone Dictionary",
                    value="./multi_tone_dict.json",
                )
            with gr.Row():
                load_btn = gr.Button("Load Dictionary")
                save_btn = gr.Button("Save Dictionary")
                # Multi-tone dictionary editor and loader   
            multi_tone_textbox = gr.Textbox(
                label="Multi-Tone Dictionary (JSON format)",
                value=load_dict(load_path_textbox.value)
            )
        

    # Button click to load dictionary from specified path
    load_btn.click(
        fn=load_dict,
        inputs=[load_path_textbox],
        outputs=[multi_tone_textbox],
    )
    # Button click to save dictionary to specified path
    save_btn.click(
        fn=save_dict,
        inputs=[multi_tone_textbox, save_path_textbox],
        outputs=[status_text],
    )

    # Generate speech with updated multi-tone dictionary
    btn.click(
        fn=lambda text, audio_prompt, rms, ref_duration, t_shift, num_steps, speed, return_smooth, multi_tone_dict: infer(
            text,
            audio_prompt,
            rms,
            ref_duration,
            t_shift,
            num_steps,
            speed,
            return_smooth,
            json.loads(multi_tone_dict),
        ),
        inputs=[
            input_text,
            input_audio,
            rms_val,
            ref_duration_val,
            t_shift_val,
            steps_val,
            speed_val,
            smooth_val,
            multi_tone_textbox,
        ],
        outputs=[audio_out, status_text],
    )

demo.launch(theme=gr.themes.Soft(), inbrowser=True)

# 基础示例 
# print("\033[31m红色文字\033[0m")  # 31为红色，\033[0m重置样式 
 
# 颜色代码参考：
# 前景色：30（黑）、31（红）、32（绿）、33（黄）、34（蓝）、35（紫）、36（青）、37（白）
# 背景色：40-47（对应前景色范围）
# 样式：1（加粗）、4（下划线）
