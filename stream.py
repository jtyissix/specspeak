import argparse
import datetime
import os
import re
import sys
import time
from threading import Thread
from typing import Optional

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

from loguru import logger
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.tokenizer import get_audio_tokenizer


def get_args():
    parser = argparse.ArgumentParser(description="VITA-Audio stream inference")
    parser.add_argument("--audio_input", type=str, default=None, help="path to input audio file")
    parser.add_argument("--model_path", default="models/vita_balance_official/")
    parser.add_argument("--audio_tokenizer_path", default="models/THUDM/")
    parser.add_argument("--flow_path", default="models/Decoder/")
    parser.add_argument("--trt_path", default="models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan")
    parser.add_argument("--output_dir", default="output/")
    args = parser.parse_args()
    logger.info(args)
    return args


# init parms
args = get_args()

model_name_or_path = args.model_path

device_map = "auto"

sys.path.append("third_party/GLM-4-Voice/")
sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

audio_tokenizer_path = args.audio_tokenizer_path
flow_path = args.flow_path

audio_tokenizer_rank = 0
audio_tokenizer_type = "glm4voice"
#audio_tokenizer_type = "sensevoice_glm4voice"

prompt_audio_path = None

torch_dtype = torch.bfloat16

audio_tokenizer = get_audio_tokenizer(
    audio_tokenizer_path,
    audio_tokenizer_type,
    flow_path=flow_path,
    rank=audio_tokenizer_rank,
)
audio_tokenizer.load_model(load_flow_trt=True,trt_path=args.trt_path)

chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""

add_generation_prompt = True

default_system_message = []

luke_system_message = [
    {
        "role": "system",
        "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
    },
]
mode = "luke"
message = ""
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    chat_template=chat_template,
)
# logger.info(f"{tokenizer=}")



model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2",
    ignore_mismatched_sizes=True,
).eval()


# logger.info("model", model)
logger.info(f"{model.config.model_type=}")
# logger.info(f"{model.hf_device_map=}")

# TTS_END_LOCK = False

model.generation_config = GenerationConfig.from_pretrained(
    model_name_or_path, trust_remote_code=True
)

model.generation_config.max_new_tokens = 8192
model.generation_config.chat_format = "chatml"
model.generation_config.max_window_size = 8192
model.generation_config.use_cache = True
# model.generation_config.use_cache = False
model.generation_config.do_sample = True
model.generation_config.temperature = 1.0
model.generation_config.top_k = 50
model.generation_config.top_p = 1.0
model.generation_config.num_beams = 1
model.generation_config.pad_token_id = tokenizer.pad_token_id


def extract_token_ids_as_int(text):
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    token_ids = pattern.findall(text)
    return [int(id) for id in token_ids]


class TextAudioIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        # self.audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
        self.audio_offset = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
        self.num_decode_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and logger.infos them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.num_decode_tokens += len(value)

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we logger.info the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        elif self.token_cache[-1] >= self.audio_offset:
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, logger.infos until the last space char (simple heuristic to avoid logger.infoing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)
        while self.text_queue.qsize() > 10:
            time.sleep(0.01)


streamer = TextAudioIteratorStreamer(tokenizer, skip_prompt=True)
audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")


if prompt_audio_path is not None:
    if audio_tokenizer.apply_to_role("system", is_discrete=True):
        # discrete codec
        prompt_audio_tokens = audio_tokenizer.encode(prompt_audio_path)
        prompt_audio_tokens = "".join(f"<|audio_{i}|>" for i in prompt_audio_tokens)
        system_message = [
            {
                "role": "system",
                "content": f"Your Voice: <|begin_of_audio|>{prompt_audio_tokens}<|end_of_audio|>\n",
            },
        ]

    else:
        # contiguous codec
        system_message = default_system_message

elif mode == "luke":
    system_message = luke_system_message

else:
    system_message = default_system_message


def run_infer_stream(audio_tensor,output_dir):
    all_audio=[]
    audio_tokenizer.audio_decoder.reset_dict()
    #logger.info("=" * 100)
    start_time = time.perf_counter()
    #logger.info(start_time)

    if audio_tensor is not None:
        messages = system_message + [
            {
                "role": "user",
                "content": message + "\n<|audio|>",
            },
        ]
    else:
        messages = system_message + [
            {
                "role": "user",
                "content": message,
            },
        ]

    if audio_tensor is not None and audio_tokenizer.apply_to_role("user", is_discrete=True):
        # discrete codec
        audio_tokens = audio_tokenizer.encode(audio_tensor)
        audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
        messages[-1]["content"] = messages[-1]["content"].replace(
            "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
        )

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        # return_tensors="pt",
    )  # .to("cuda")

    if audio_tensor is not None and audio_tokenizer.apply_to_role("user", is_contiguous=True):
        # contiguous codec
        print(f"{audio_tensor=}")
        input_ids, audios, audio_indices = add_audio_input_contiguous(
            input_ids, [audio_tensor], tokenizer, audio_tokenizer
        )
    else:
        audios = None
        audio_indices = None

    # mtp_inference_mode = [1, 10, 4, 10]
    # model.generation_config.mtp_inference_mode = mtp_inference_mode
    input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

    #logger.info(f"input {tokenizer.decode(input_ids[0], skip_special_tokens=False)}", flush=True)

    model.generation_config.do_sample = False

    generation_kwargs = dict(
        input_ids=input_ids,
        audios=audios,
        audio_indices=audio_indices,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    past_tts_speech_len = 0
    past_audio_token_len = 0

    option_steps = 10
    num_audio_chunk = 0
    for new_text in streamer:
        # logger.info(f"{new_text=}")

        generated_text += new_text
        if num_audio_chunk==0:
            print("before audio text time:",time.perf_counter() - start_time)
        
        if "<|end_of_audio|>" == new_text:
            #breakpoint()
            audio_tokens = extract_token_ids_as_int(generated_text)
            #print(f"{generated_text=}")
            print(len(audio_tokens))
            if 'boost' in model_name_or_path:
                if num_audio_chunk == 0:
                    pass
                elif len(audio_tokens) - past_audio_token_len > 16:
                    pass
                else:
                    continue
            elif 'balance' in model_name_or_path or 'normal' in model_name_or_path:
                if num_audio_chunk == 0 and len(audio_tokens)==8:
                    pass
                elif len(audio_tokens) - past_audio_token_len > 16:
                    pass
                else:
                    continue
             
            #breakpoint()
            # from torch.nn.attention import SDPBackend, sdpa_kernel
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            tts_speech = audio_tokenizer.decode(
                audio_tokens,
                source_speech_16k=prompt_audio_path,
                option_steps=option_steps,
            )
            option_steps = 10 #min(option_steps + 2, 10)

            new_tts_speech = tts_speech[past_tts_speech_len:]
            tts_np = new_tts_speech.squeeze().float().cpu().numpy()
            max_val = np.max(np.abs(tts_np))
            if max_val > 0:
                tts_np = tts_np / max_val  # 归一化到 [-1, 1]

            output_data = (tts_np * 32767).astype(np.int16)
            all_audio.append(output_data)
            # import pdb;pdb.set_trace()

            
            if num_audio_chunk == 0:
                first_audio_time = (
                    time.perf_counter() - start_time
                )  # Capture the first audio generation time
                
                logger.info(f"First audio generation time: {first_audio_time}")

            past_tts_speech_len = len(tts_speech)
            past_audio_token_len = len(audio_tokens)

            if len(audio_tokens) > 512:
                generated_text = ""
                past_tts_speech_len = 0
                past_audio_token_len = 0

            num_audio_chunk += 1
    #breakpoint()
    if not len(all_audio)==0:
        # logger.info(f"{output_data.shape=} {output_data[:20]=}")
        # logger.info(max(output_data))
        output_data=np.concatenate(all_audio)
        tensor = torch.from_numpy(output_data.astype("int16")).unsqueeze(0)  # (1, N)

        base_name = os.path.basename(audio_tensor).replace('mp3', 'wav')
        torchaudio.save(
            f"{output_dir}/{base_name}",
            tensor,
            22050,
            encoding="PCM_S",
            bits_per_sample=16,
        )


if __name__ == "__main__":
    

    audio_input = args.audio_input
    if audio_input is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(7):
            run_infer_stream(audio_input, args.output_dir)