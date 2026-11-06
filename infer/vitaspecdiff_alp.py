import argparse
import datetime
import os
import csv
import re
import sys
import time
import threading
from typing import Tuple
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Thread
from typing import Optional
import vita_audio.models
import numpy as np
import torch
import torchaudio
import json
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig
from loguru import logger
import logging
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.tokenizer import get_audio_tokenizer
target_model_name_or_path = "models/vita_balance_official/"
draft_model_name_or_path = "models/vita_0.5b_balance_final/"
device_map = "auto"
sys.path.append("../third_party/GLM-4-Voice/")
sys.path.append("../third_party/GLM-4-Voice/cosyvoice/")
sys.path.append("../third_party/GLM-4-Voice/third_party/Matcha-TTS/")
audio_tokenizer_path ="models/THUDM/"
flow_path = "Decoder/"

audio_tokenizer_rank = 0
audio_tokenizer_type = "glm4voice"
#audio_tokenizer_type = "sensevoice_glm4voice"

prompt_audio_path = None
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
torch_dtype = torch.bfloat16
#---------------------------------------------------------

def remove_audio_tokens(text):
    pattern = re.compile(r"<\|audio_\d+\|>|<\|begin_of_audio\|>|<\|end_of_audio\|>|<\|im_end\|>")
    return pattern.sub("", text)
def extract_token_ids_as_int(text):
        pattern = re.compile(r"<\|audio_(\d+)\|>")
        token_ids = pattern.findall(text)
        return [int(id) for id in token_ids]
def speculative_sample(
    
    draft_tokens: torch.Tensor,      # [K]
    draft_logits: torch.Tensor,      # [K, vocab_size]
    target_logits: torch.Tensor,     # [K+1, vocab_size]
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> Tuple[torch.Tensor, int]:
    """
    执行speculative sampling，支持temperature, top_k, top_p
    
    Args:
        draft_tokens: [K] draft生成的token
        draft_logits: [K, vocab_size] 来自draft model的logits
        target_logits: [K+1, vocab_size] 来自verify_with_draft的logits
        temperature: 采样温度
        top_k: top-k截断，0表示不使用
        top_p: nucleus sampling阈值，1.0表示不使用
        
    Returns:
        (accepted_tokens, num_accepted)
        accepted_tokens: [num_accepted] 接受的token tensor
        num_accepted: 接受的token数量
    """
    accepted = []
    K = draft_tokens.shape[0]
    
    for t in range(K):
        # 获取当前位置的logits
        q_logits = target_logits[t].clone()  # [vocab_size]
        p_logits = draft_logits[t].clone()   # [vocab_size]
        
        # 应用temperature
        if temperature != 1.0:
            q_logits = q_logits / temperature
            p_logits = p_logits / temperature
        
        # 计算概率分布
        q_probs = F.softmax(q_logits, dim=-1)
        p_probs = F.softmax(p_logits, dim=-1)
        
        draft_token = draft_tokens[t].item()
        q_prob = q_probs[draft_token].item()
        p_prob = p_probs[draft_token].item()
       
        # 计算接受概率 r = min(1, q(x)/p(x))
        if p_prob < 1e-10:
            accept_prob = 1.0 if q_prob > 1e-10 else 0.0
        else:
            accept_prob = min(1.0, q_prob / p_prob)
        print(accept_prob,q_prob,p_prob)
        #breakpoint()
        # 采样 u ~ U[0,1]
        u = torch.rand(1, device=draft_tokens.device).item()
        
        if u < accept_prob:
            # 接受draft token
            accepted.append(draft_token)
        else:
            # 拒绝，从调整后的分布重采样: (q - p)+ / ||q - p||_1
            residual = torch.clamp(q_probs - p_probs, min=0)
            residual_sum = residual.sum()
            
            if residual_sum < 1e-10:
                # fallback到target分布
                probs = q_probs
            else:
                probs = residual / residual_sum
            
            # 从调整后的分布采样
            new_token = _sample_token(probs, temperature=1.0, top_k=top_k, top_p=top_p)
            accepted.append(new_token)
            
            # 拒绝后立即退出
            accepted_tensor = torch.tensor(accepted, dtype=torch.long, device=draft_tokens.device)
            return accepted_tensor, len(accepted)
    
    # 所有K个draft token都被接受，从bonus位置采样
    bonus_logits = target_logits[K].clone()  # [vocab_size]
    bonus_token = _sample_token(
        bonus_logits, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p,
        apply_softmax=True
    )
    accepted.append(bonus_token)
    
    accepted_tensor = torch.tensor(accepted, dtype=torch.long, device=draft_tokens.device)
    return accepted_tensor, len(accepted)


def _sample_token(
    
    logits_or_probs: torch.Tensor,  # [vocab_size]
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    apply_softmax: bool = False
) -> int:
    """
    从logits或概率分布中采样一个token
    
    Args:
        logits_or_probs: logits或概率分布
        temperature: 采样温度
        top_k: top-k截断
        top_p: nucleus sampling阈值
        apply_softmax: 是否需要先应用softmax
        
    Returns:
        采样的token id
    """
    if apply_softmax:
        # 应用temperature
        if temperature != 1.0:
            logits_or_probs = logits_or_probs / temperature
        probs = F.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs
    
    # 应用top-k
    if top_k > 0 and top_k < probs.shape[0]:
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs)
        probs.scatter_(0, top_k_indices, top_k_probs)
        probs = probs / probs.sum()
    
    # 应用top-p (nucleus sampling)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        
        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumsum_probs > top_p
        # 保留第一个超过top_p的token（确保至少有一个token）
        if sorted_indices_to_remove.any():
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
        
        # 移除低概率tokens
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        
        # 重新归一化
        prob_sum = probs.sum()
        if prob_sum > 1e-10:
            probs = probs / prob_sum
        else:
            # fallback: 均匀分布
            probs = torch.ones_like(probs) / probs.shape[0]
    
    # 采样
    token = torch.multinomial(probs, 1).item()
    return token

class VitaStreaming():
    def __init__(self):
        
        self.audio_tokenizer = get_audio_tokenizer(
        audio_tokenizer_path,
        audio_tokenizer_type,
        flow_path=flow_path,
        rank=audio_tokenizer_rank,
        )
        self.audio_tokenizer.load_model(load_flow_trt=True,trt_path='models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan')
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.verify_stream = torch.cuda.Stream(device='cuda')
        self.vocoder_stream = torch.cuda.Stream(device='cuda')
        #self.dr_spg_stream= torch.cuda.Stream(device='cuda')
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            target_model_name_or_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )
        # logger.info(f"{tokenizer=}")
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_model_name_or_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )


        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            trust_remote_code=False,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True,
        ).eval()
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name_or_path,
            trust_remote_code=False,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True,
        ).eval()

        # logger.info("model", model)
        logger.info(f"{self.target_model.config.model_type=}")
        # logger.info(f"{model.hf_device_map=}")
        logger.info(f"{self.draft_model.config.model_type=}")
        # TTS_END_LOCK = False

        self.target_model.generation_config = GenerationConfig.from_pretrained(
            target_model_name_or_path, trust_remote_code=True
        )

        self.target_model.generation_config.max_new_tokens = 8192
        self.target_model.generation_config.chat_format = "chatml"
        self.target_model.generation_config.max_window_size = 8192
        self.target_model.generation_config.use_cache = True
        # model.generation_config.use_cache = False
        self.target_model.generation_config.do_sample = True
        self.target_model.generation_config.temperature = 1.0
        self.target_model.generation_config.top_k = 50
        self.target_model.generation_config.top_p = 1.0
        self.target_model.generation_config.num_beams = 1
        self.target_model._prepare_mtp_for_generation(self.target_model.generation_config.mtp_inference_mode, max_new_tokens=self.target_model.generation_config.max_new_tokens)
        self.target_model.generation_config.pad_token_id = self.target_tokenizer.pad_token_id
        self.audio_offset = self.target_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        
        self.draft_model.generation_config.max_new_tokens = 8192
        self.draft_model.generation_config.chat_format = "chatml"
        self.draft_model.generation_config.max_window_size = 8192
        self.draft_model.generation_config.use_cache = True
        # model.generation_config.use_cache = False
        self.draft_model.generation_config.do_sample = True
        self.draft_model.generation_config.temperature = 1.0
        self.draft_model.generation_config.top_k = 50
        self.draft_model.generation_config.top_p = 1.0
        self.draft_model.generation_config.num_beams = 1
        self.draft_model._prepare_mtp_for_generation(self.draft_model.generation_config.mtp_inference_mode, max_new_tokens=self.draft_model.generation_config.max_new_tokens)
        self.draft_model.generation_config.pad_token_id = self.draft_tokenizer.pad_token_id
        self.audio_offset = self.draft_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        
        
        if prompt_audio_path is not None:
            if self.audio_tokenizer.apply_to_role("system", is_discrete=True):
                # discrete codec
                prompt_audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
                prompt_audio_tokens = "".join(f"<|audio_{i}|>" for i in prompt_audio_tokens)
                self.system_message = [
                    {
                        "role": "system",
                        "content": f"Your Voice: <|begin_of_audio|>{prompt_audio_tokens}<|end_of_audio|>\n",
                    },
                ]

            else:
                # contiguous codec
                self.system_message = default_system_message

        elif mode == "luke":
            self.system_message = luke_system_message

        else:
            self.system_message = default_system_message
        self._vocoder_abort: threading.Event = threading.Event()
        self._vocoder_abort.clear()
    def _vocoder_diffusion_loop(
        self,
        audio_tokens,
        source_speech_16k,
        num_steps: int,
        
        t0=None
    ) :
        """
        Execute ``num_steps`` vocoder diffusion steps **sequentially**
        inside a single background thread.

        Before every step we poll ``self._vocoder_abort``.  If it has
        been set (draft was rejected), we:
          1. Exit the loop immediately.
          2. Reset the vocoder streaming state so the *next* chunk
             starts from a clean cache.
          3. Return ``None`` to signal "no usable audio produced".

        The ``is_last_speech_chunk`` flag is forwarded only on the very
        last step of the very last chunk, matching CosyVoice2's
        expected call convention.

        Returns
        -------
        dict or None
            Vocoder output containing ``tts_speech`` on success,
            or ``None`` if the loop was aborted.
        """
        #result: Optional[dict] = None

        for step in range(num_steps):
            # ── cooperative abort check ──────────────────────────────
            
            if self._vocoder_abort.is_set():
                # Streaming cache is now inconsistent (partial diffusion).
                # Reset so the next chunk can start cleanly.
                '''
                self.cosy_vocoder.model.flow.reset_step_cache(
                    reset_step_cache=False, device=speech_token.device
                )
                '''
                logging.info(
                    "vocoder_diffusion_loop: aborted at step %d/%d",
                    step, num_steps,
                )
                return None
            
            # is_last_speech_chunk only on the *final* step of the
            # *final* chunk — otherwise the vocoder won't flush its
            # internal state prematurely.
            #final_flag = is_last_speech_chunk and (step == num_steps - 1)
            print('time from t0 to vocoder one step:',time.perf_counter()-t0)
            result = self.audio_tokenizer.decode_one_step(
                    audio_tokens,
                    source_speech_16k=source_speech_16k)
        return result
    def n_step_vocoder_worker(self,*args, **kwargs):
        with torch.cuda.stream(self.vocoder_stream):
            return self._vocoder_diffusion_loop(*args, **kwargs)
    def verify_worker(self,*args, **kwargs):
        with torch.cuda.stream(self.verify_stream):
            return self.target_model.speculative_verify(*args, **kwargs)
    def run_infer_stream(self,audio_tensor,output_dir):
        all_audio=[]
        takeover_ready=False
        first_audio_time = None
        self.draft_model.stream_resume_state=None #self.draft_model
        self.target_model.stream_resume_state=None #self.target_model
        self.audio_tokenizer.audio_decoder.reset_dict()
        self.audio_tokenizer.audio_decoder.flow.reset_step_cache(True,device='cuda')
        #logger.info("=" * 100)
        start_time = time.perf_counter()
        #logger.info(start_time)

        if audio_tensor is not None:
            messages = self.system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = self.system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_tensor is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_tensor)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self.draft_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            # return_tensors="pt",
        )  # .to("cuda")

        if audio_tensor is not None and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
            # contiguous codec
            print(f"{audio_tensor=}")
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, [audio_tensor], self.draft_tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        # mtp_inference_mode = [1, 10, 4, 10]
        # model.generation_config.mtp_inference_mode = mtp_inference_mode
        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

        #logger.info(f"input {tokenizer.decode(input_ids[0], skip_special_tokens=False)}", flush=True)

        self.draft_model.generation_config.do_sample = False
        self.target_model.generation_config.do_sample = False
        def get_num_accepted(draft_tokens,draft_logit,verify_logit):
            draft_text_logit=torch.cat(draft_logit)
            draft_text_logit=torch.cat([draft_text_logit[:1,:],draft_text_logit[5:,:]],dim=0)
            target_verify_logit=torch.cat([verify_logit[:,:1,:],verify_logit[:,5:,:]],dim=1).squeeze(0)
            draft_text_tokens=torch.cat(draft_tokens)
            draft_text_tokens=torch.cat([draft_text_tokens[:1],draft_text_tokens[5:]])
            #check the shape
            accepted_tensor, num_accepted = speculative_sample(
                        draft_tokens=draft_text_tokens,      # [K]
                        draft_logits=draft_text_logit,      # [K, vocab_size]
                        target_logits=target_verify_logit,     # [K+1, vocab_size]
                    )
            
            return accepted_tensor, num_accepted
        full_text=""
        generated_text = ""
        past_tts_speech_len = 0
        past_audio_token_len = 0
        steps, past_kv = 0, None
        option_steps = 10
        num_audio_chunk = 0
        draft_toks=[]
        draft_logits=[]
        get_verify_logit_task=None
        get_num_accepted_task=None
        self.audio_tokenizer.audio_decoder.flow.reset_step_cache(True,device='cuda')
        for tok,logit in self.draft_model.stream_generate(input_ids, max_new_tokens=16,
                                      steps_done=0,
                                      do_sample=False,
                                      eos_token_id=[151645, 151643],
                                      return_past_key_values=False):
            # logger.info(f"{new_text=}")
            
            draft_toks.append(tok)
            draft_logits.append(logit)
            new_text=self.draft_tokenizer.decode(draft_toks[-1])
            generated_text+=new_text
            full_text+=new_text
            if num_audio_chunk==0:
                print("before audio text time:",time.perf_counter() - start_time)
            if len(draft_toks)==6:
                get_verify_logit_task = self.executor.submit(self.verify_worker,
                    input_ids=input_ids,
                    draft_tokens=torch.cat(draft_toks,dim=-1).unsqueeze(0)
                )
            if get_verify_logit_task is not None and get_verify_logit_task.done() and get_num_accepted_task is None:
                get_num_accepted_task = self.executor.submit(get_num_accepted,draft_toks[:6],draft_logits[:6],get_verify_logit_task.result())
            if "<|end_of_audio|>" == new_text:
                #breakpoint()
                audio_tokens = extract_token_ids_as_int(generated_text)
                #print(f"{generated_text=}")
                print(len(audio_tokens))
                
                if num_audio_chunk == 0 and len(audio_tokens)==8:
                    pass
                elif len(audio_tokens) - past_audio_token_len > 16:
                    pass
                else:
                    continue
                vocoder_prediffuse_task_1=self.executor.submit(self.n_step_vocoder_worker,
                        audio_tokens,
                        prompt_audio_path,
                        num_steps=10,
                        t0=start_time
                    )
                
        if get_num_accepted_task is None:
            get_num_accepted_task = self.executor.submit(get_num_accepted,draft_toks[:6],draft_logits[:6],get_verify_logit_task.result())
        accepted_tensor,num_accepted=get_num_accepted_task.result()
        if num_accepted==3:
            tts_speech=vocoder_prediffuse_task_1.result()
            new_tts_speech = tts_speech[past_tts_speech_len:]
            tts_np = new_tts_speech.squeeze().float().cpu().numpy()
            max_val = np.max(np.abs(tts_np))
            if max_val > 0:
                tts_np = tts_np / max_val  # 归一化到 [-1, 1]

            output_data = (tts_np * 32767).astype(np.int16)
            all_audio.append(output_data)
            if num_audio_chunk == 0:
                    first_audio_time = (
                        time.perf_counter() - start_time
                    )
            logger.info(f"first audio chunk time: {first_audio_time}")
            past_tts_speech_len = len(tts_speech)
            past_audio_token_len = len(audio_tokens)
            num_audio_chunk += 1
            takeover_ready=True
        elif num_accepted==2:
            draft_toks,_=self.draft_model.draft_correct_and_generate(
                input_ids=input_ids,
                first_text_token=draft_toks[0].unsqueeze(0),
                first_audio_tokens=torch.cat(draft_toks[1:5],dim=-1).unsqueeze(0),
                corrected_text_token=accepted_tensor[1].unsqueeze(0).unsqueeze(0),
                do_sample=False,
            )
            self._vocoder_abort.set()
            vocoder_prediffuse_task_1.result() # wait for vocoder to finish
            this_step=10-min(4,self.audio_tokenizer.audio_decoder.flow.now_step)
            self.audio_tokenizer.audio_decoder.flow.reset_step_cache(False,device='cuda')
            self.audio_tokenizer.audio_decoder.flow.set_now_steps(10-this_step)
            self._vocoder_abort.clear()
            #breakpoint()
            generated_text=self.draft_tokenizer.decode(torch.cat(draft_toks))
            full_text=remove_audio_tokens(generated_text)
            audio_tokens = extract_token_ids_as_int(generated_text)
            vocoder_prediffuse_task_2=self.executor.submit(self.n_step_vocoder_worker,
                        audio_tokens,
                        prompt_audio_path,
                        num_steps=this_step,
                        t0=start_time
                    )
            tts_speech=vocoder_prediffuse_task_2.result()
            new_tts_speech = tts_speech[past_tts_speech_len:]
            tts_np = new_tts_speech.squeeze().float().cpu().numpy()
            max_val = np.max(np.abs(tts_np))
            if max_val > 0:
                tts_np = tts_np / max_val  # 归一化到 [-1, 1]

            output_data = (tts_np * 32767).astype(np.int16)
            all_audio.append(output_data)
            if num_audio_chunk == 0:
                    first_audio_time = (
                        time.perf_counter() - start_time
                    )
            logger.info(f"first audio chunk time: {first_audio_time}")
            past_tts_speech_len = len(tts_speech)
            past_audio_token_len = len(audio_tokens)
            num_audio_chunk += 1
            takeover_ready=True
        else:
            self._vocoder_abort.set()
            
            generate_5,_=self.target_model.target_correct_and_generate(
                input_ids=input_ids,
                first_text_token=accepted_tensor[0].unsqueeze(0).unsqueeze(0),
            )
            draft_toks,_=self.draft_model.draft_correct_and_generate(
                input_ids=input_ids,
                first_text_token=accepted_tensor[0].unsqueeze(0).unsqueeze(0),
                first_audio_tokens=generate_5[:,:4],
                corrected_text_token=generate_5[:,4:],
                do_sample=False,
            )
            self._vocoder_abort.set()
            vocoder_prediffuse_task_1.result() # wait for vocoder to finish
            this_step=10-min(2,self.audio_tokenizer.audio_decoder.flow.now_step)
            self.audio_tokenizer.audio_decoder.flow.reset_step_cache(False,device='cuda')
            self.audio_tokenizer.audio_decoder.flow.set_now_steps(10-this_step)
            self._vocoder_abort.clear()
            #breakpoint()
            generated_text=self.draft_tokenizer.decode(torch.cat(draft_toks))
            full_text=remove_audio_tokens(generated_text)
            audio_tokens = extract_token_ids_as_int(generated_text)
            vocoder_prediffuse_task_2=self.executor.submit(self.n_step_vocoder_worker,
                        audio_tokens,
                        prompt_audio_path,
                        num_steps=this_step,
                        t0=start_time
                    )
            tts_speech=vocoder_prediffuse_task_2.result()
            new_tts_speech = tts_speech[past_tts_speech_len:]
            tts_np = new_tts_speech.squeeze().float().cpu().numpy()
            max_val = np.max(np.abs(tts_np))
            if max_val > 0:
                tts_np = tts_np / max_val  # 归一化到 [-1, 1]

            output_data = (tts_np * 32767).astype(np.int16)
            all_audio.append(output_data)
            if num_audio_chunk == 0:
                    first_audio_time = (
                        time.perf_counter() - start_time
                    )
            logger.info(f"first audio chunk time: {first_audio_time}")
            past_tts_speech_len = len(tts_speech)
            past_audio_token_len = len(audio_tokens)
            num_audio_chunk += 1
            takeover_ready=True
        if takeover_ready:
            seam_step=True
            target_toks=[]
            #breakpoint()
            for tok,logit in self.target_model.draft_prefill_and_stream_generate(
                input_ids=input_ids,
                first_text_token=draft_toks[0].unsqueeze(0),
                first_audio_tokens=torch.cat(draft_toks[1:5],dim=-1).unsqueeze(0),
                partial_text_tokens=draft_toks[5].unsqueeze(0),
                second_audio_tokens=torch.cat(draft_toks[8:],dim=-1).unsqueeze(0),
                max_new_tokens=8192,
                do_sample=False,
                eos_token_id=[151645, 151643],
                return_past_key_values=False,
            ):
                if seam_step:
                    target_toks.append(tok)
                    #breakpoint()
                    generated_text=self.target_tokenizer.decode(torch.cat(target_toks)[0])
                    full_text=remove_audio_tokens(generated_text)
                    seam_step=False
                else:
                    target_toks.append(tok)
                    new_text=self.draft_tokenizer.decode(target_toks[-1])
                    generated_text+=new_text
                    full_text+=new_text
                    if "<|end_of_audio|>" == new_text:
                        self.audio_tokenizer.audio_decoder.flow.reset_step_cache(True,device='cuda')
                        audio_tokens = extract_token_ids_as_int(generated_text)
                        if num_audio_chunk == 0 and len(audio_tokens)==8:
                            pass
                        elif len(audio_tokens) - past_audio_token_len > 16:
                            pass
                        else:
                            continue
                        tts_speech=self._vocoder_diffusion_loop(audio_tokens,
                            source_speech_16k=prompt_audio_path,
                            num_steps=10,t0=start_time)
                        new_tts_speech = tts_speech[past_tts_speech_len:]
                        tts_np = new_tts_speech.squeeze().float().cpu().numpy()
                        max_val = np.max(np.abs(tts_np))
                        if max_val > 0:
                            tts_np = tts_np / max_val  # 归一化到 [-1, 1]

                        output_data = (tts_np * 32767).astype(np.int16)
                        all_audio.append(output_data)
                        past_tts_speech_len = len(tts_speech)
                        past_audio_token_len = len(audio_tokens)

                        if len(audio_tokens) > 512:
                            generated_text = ""
                            past_tts_speech_len = 0
                            past_audio_token_len = 0
                
                        num_audio_chunk += 1
        print(f"first audio chunk time: {first_audio_time},generated text: {remove_audio_tokens(generated_text)}")
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
        return {"text":remove_audio_tokens(full_text), 'audio': f"{output_dir}/{base_name}"},first_audio_time   
if __name__ == "__main__":
    import os, json, math

    # ====== paths ======
    audio_folder = "dataset/aplca/eval_datas/alpaca_eval/audios"
    time_json_path = ""
    # 只保存每个音频第一次生成的文本（jsonl，一行一个json），不存在会自动创建
    text_jsonl_path = ""
    audio_save_path= ""
    data_csv_path = "aplca/eval_datas/alpaca_eval/alpaca_eval.csv"
    os.makedirs(os.path.dirname(text_jsonl_path), exist_ok=True)
    vita=VitaStreaming()
    '''
    with open(time_json_path, "r") as f:
        num_dict = json.load(f)
    '''
    

    # ====== build audio list (optional: check existence) ======
    audio_path_list = []
    with open(data_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        header = next(reader)  # 跳过表头

        for row in reader:
            wav_name = row[4]  # 假设第5列是 wav 文件名
            wav_path = f"{audio_folder}/{wav_name}"
        
            audio_path_list.append(wav_path)

    
    print(f"Total {len(audio_path_list)} audios to test")

    

    # ====== evaluation loop ======
    for i in range( len(audio_path_list)):
        time_list = []

        wav_path = audio_path_list[i]
        print(f"[{i+1}/{len(audio_path_list)}] {wav_path}")

        
            
            
        response, first_audio_time = vita.run_infer_stream(wav_path,audio_save_path)
        #time_list.append(first_audio_time)

        
        
        with open(text_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "index": i,
                "wav_path": wav_path,
                "text": response.get("text", ""),
                "audio_output": response.get("audio", None)
            }, ensure_ascii=False) + "\n")
            f.flush()
        
        
        
        total_time_list.append(first_audio_time)
        
        
        # 每个音频都更新一次时间json（保持你原逻辑）
        
        num_dict["our_time"] = total_time_list
        with open(time_json_path, "w") as f:
            json.dump(num_dict, f)
        
    # ====== summary ======
    
    print(len(total_time_list))
    print(total_time_list)
    print("average first audio time:", sum(total_time_list) / len(total_time_list))
    print("fastest first audio time:", min(total_time_list))

    # p90：用 ceil(0.9*n)-1，避免 int(0.9*n) 在小样本时偏到最大值
    x_sorted = sorted(total_time_list)
    p90_idx = max(0, math.ceil(0.9 * len(x_sorted)) - 1)
    print("p90 first audio time:", x_sorted[p90_idx])
    