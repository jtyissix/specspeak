
import os
import csv
import json
import re
import sys
import time
import logging
import threading
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import vita_audio.models
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
sys.path.append("../third_party/GLM-4-Voice/")
sys.path.append("../third_party/GLM-4-Voice/cosyvoice/")
sys.path.append("../third_party/GLM-4-Voice/third_party/Matcha-TTS/")

#from vita_audio.tokenizer import AudioTokenizer
from vita_audio.tokenizer import get_audio_tokenizer
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous

logger = logging.getLogger(__name__)


target_model_name_or_path = "models/vita_balance_official/"
draft_model_name_or_path = "models/vita_0.5b_balance_final/"
device_map = "auto"
audio_tokenizer_path ="models/THUDM/"
flow_path = "models/Decoder/"

audio_tokenizer_rank = 0
audio_tokenizer_type = "glm4voice"
add_generation_prompt = True
prompt_audio_path = None
default_system_message = []
luke_system_message = [
    {
        "role": "system",
        "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
    },
]


chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""
mode = "luke"
message = ""
torch_dtype = torch.bfloat16

# ── 工具函数（复用自 vitaspecdiff）──────────────────────────────────────

def remove_audio_tokens(text):
    pattern = re.compile(r"<\|audio_\d+\|>|<\|begin_of_audio\|>|<\|end_of_audio\|>|<\|im_end\|>")
    return pattern.sub("", text)

def extract_token_ids_as_int(text):
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    return [int(id) for id in pattern.findall(text)]

def _sample_token(logits_or_probs, temperature=1.0, top_k=0, top_p=1.0,
                   apply_softmax=False):
    if apply_softmax:
        if temperature != 1.0:
            logits_or_probs = logits_or_probs / temperature
        probs = F.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs
    if top_k > 0 and top_k < probs.shape[0]:
        topk_probs, topk_idx = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs)
        mask.scatter_(0, topk_idx, topk_probs)
        probs = mask / mask.sum()
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        remove = cumsum - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        probs = torch.zeros_like(probs)
        probs.scatter_(0, sorted_idx, sorted_probs)
    return torch.multinomial(probs, num_samples=1).item()

def speculative_sample(draft_tokens, draft_logits, target_logits,
                        temperature=1.0, top_k=0, top_p=1.0):
    """标准 speculative sampling。复用自 vitaspecdiff。"""
    accepted = []
    K = draft_tokens.shape[0]
    for t in range(K):
        q_logits = target_logits[t].clone()
        p_logits = draft_logits[t].clone()
        if temperature != 1.0:
            q_logits = q_logits / temperature
            p_logits = p_logits / temperature
        q_probs = F.softmax(q_logits, dim=-1)
        p_probs = F.softmax(p_logits, dim=-1)
        draft_token = draft_tokens[t].item()
        q_prob = q_probs[draft_token].item()
        p_prob = p_probs[draft_token].item()
        if p_prob < 1e-10:
            accept_prob = 1.0 if q_prob > 1e-10 else 0.0
        else:
            accept_prob = min(1.0, q_prob / p_prob)
        print("accepted prob: ",accept_prob,"target prob: ",q_prob,"draft prob: ",p_prob)
        u = torch.rand(1, device=draft_tokens.device).item()
        if u < accept_prob:
            accepted.append(draft_token)
        else:
            residual = torch.clamp(q_probs - p_probs, min=0)
            residual_sum = residual.sum()
            probs = residual / residual_sum if residual_sum > 1e-10 else q_probs
            new_token = _sample_token(probs, temperature=1.0, top_k=top_k, top_p=top_p)
            accepted.append(new_token)
            return (torch.tensor(accepted, dtype=torch.long,
                                 device=draft_tokens.device), len(accepted))
    # 全部接受 → bonus
    bonus_logits = target_logits[K].clone()
    bonus_token = _sample_token(bonus_logits, temperature=temperature,
                                top_k=top_k, top_p=top_p, apply_softmax=True)
    accepted.append(bonus_token)
    return (torch.tensor(accepted, dtype=torch.long,
                         device=draft_tokens.device), len(accepted))


# ═════════════════════════════════════════════════════════════════════════
#                         主类 VitaPureSpecDecode
# ═════════════════════════════════════════════════════════════════════════
class VitaPureSpecDecode:

    # 调度表 M mmmm MMM mmmmmmmm
    # text token 在 8-token draft 中的位置
    TEXT_POSITIONS = [0, 5, 6, 7]

    def __init__(
        self,
        target_model_name_or_path=None,
        draft_model_name_or_path=None,
        device_map="cuda:0",
    ):
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            target_model_name_or_path, trust_remote_code=True,
            chat_template=chat_template)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_model_name_or_path, trust_remote_code=True,
            chat_template=chat_template)
        self.audio_tokenizer = get_audio_tokenizer(
        audio_tokenizer_path,
        audio_tokenizer_type,
        flow_path=flow_path,
        rank=audio_tokenizer_rank,
        )
        self.audio_tokenizer.load_model(load_flow_trt=True,trt_path='models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan')
        # ── Target model ─────────────────────────────────────────────────
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path, trust_remote_code=False,
            device_map=device_map, torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True).eval()
        self.target_model.generation_config = GenerationConfig.from_pretrained(
            target_model_name_or_path, trust_remote_code=True)
        self.target_model.generation_config.max_new_tokens = 8192
        self.target_model.generation_config.chat_format = "chatml"
        self.target_model.generation_config.max_window_size = 8192
        self.target_model.generation_config.use_cache = True
        self.target_model.generation_config.do_sample = False
        self.target_model.generation_config.temperature = 1.0
        self.target_model.generation_config.top_k = 50
        self.target_model.generation_config.top_p = 1.0
        self.target_model.generation_config.num_beams = 1
        self.target_model._prepare_mtp_for_generation(
            self.target_model.generation_config.mtp_inference_mode,
            max_new_tokens=self.target_model.generation_config.max_new_tokens)
        self.target_model.generation_config.pad_token_id = self.target_tokenizer.pad_token_id

        # ── Draft model ──────────────────────────────────────────────────
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name_or_path, trust_remote_code=False,
            device_map=device_map, torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True).eval()
        self.draft_model.generation_config.max_new_tokens = 8192
        self.draft_model.generation_config.chat_format = "chatml"
        self.draft_model.generation_config.max_window_size = 8192
        self.draft_model.generation_config.use_cache = True
        self.draft_model.generation_config.do_sample = False
        self.draft_model.generation_config.temperature = 1.0
        self.draft_model.generation_config.top_k = 50
        self.draft_model.generation_config.top_p = 1.0
        self.draft_model.generation_config.num_beams = 1
        self.draft_model._prepare_mtp_for_generation(
            self.draft_model.generation_config.mtp_inference_mode,
            max_new_tokens=self.draft_model.generation_config.max_new_tokens)
        self.draft_model.generation_config.pad_token_id = self.draft_tokenizer.pad_token_id
        self.audio_offset = self.draft_tokenizer.convert_tokens_to_ids("<|audio_0|>")

        # ── System message ───────────────────────────────────────────────
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

        self._vocoder_abort = threading.Event()
        self._vocoder_abort.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Vocoder（复用自 vitaspecdiff）
    # ─────────────────────────────────────────────────────────────────────
    def _vocoder_diffusion_loop(self, audio_tokens, source_speech_16k,
                                 num_steps, t0=None):
        for step in range(num_steps):
            if self._vocoder_abort.is_set():
                logging.info("vocoder aborted at step %d/%d", step, num_steps)
                return None
            if t0: print(f"vocoder step time: {time.perf_counter()-t0:.4f}")
            result = self.audio_tokenizer.decode_one_step(
                audio_tokens, source_speech_16k=source_speech_16k)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # 验证 text token（完全复用 vitaspecdiff 的切片逻辑）
    # ─────────────────────────────────────────────────────────────────────
    def _get_text_accepted(self, input_ids, draft_toks, draft_logits):
        """
        验证 draft_toks[0..7] 中全部 4 个 text token。

        与 vitaspecdiff 的 get_num_accepted 完全对齐：
          - draft_text_logit  = cat([logit[:1], logit[5:]])  → [4, V]
          - target_verify     = cat([vl[:,:1,:], vl[:,5:,:]]) → [5, V]
          - draft_text_tokens = cat([tok[:1], tok[5:]])       → [4]
          - speculative_sample([4], [4,V], [5,V])

        Returns: (accepted_tensor, num_accepted)
          num_accepted==5 → bonus token, 全部 4 个 text 确认
          num_accepted<5  → 前 num_accepted-1 个原样接受, 第 num_accepted 个纠正
        """
        # ── 构建 [1, 8] draft 序列 ──────────────────────────────────────
        draft_all = torch.stack(
            [t.view(-1)[0] for t in draft_toks]).unsqueeze(0)  # [1, 8]

        # ── Target verify (backbone only, 返回 [1, 9, V]) ──────────────
        verify_logits = self.target_model.speculative_verify(
            input_ids, draft_all)  # [1, 9, V]

        # ── 提取 text-only 切片 ────────────────────────────────────────
        # draft logits: [8, V] → text positions [0, 5, 6, 7] → [4, V]
        all_draft_logits = torch.cat(draft_logits)  # [8, V]
        draft_text_logits = torch.cat(
            [all_draft_logits[:1, :], all_draft_logits[5:, :]], dim=0)  # [4, V]

        # target: positions [0, 5, 6, 7, 8] → [5, V]
        target_text_logits = torch.cat(
            [verify_logits[:, :1, :], verify_logits[:, 5:, :]], dim=1
        ).squeeze(0)  # [5, V]

        # draft text tokens: [4]
        all_tokens = torch.cat(draft_toks)  # [8]
        draft_text_tokens = torch.cat(
            [all_tokens[:1], all_tokens[5:]])  # [4]

        # ── speculative_sample ──────────────────────────────────────────
        accepted_tensor, num_accepted = speculative_sample(
            draft_text_tokens, draft_text_logits, target_text_logits)
        return accepted_tensor, num_accepted

    # ─────────────────────────────────────────────────────────────────────
    # 主推理
    # ─────────────────────────────────────────────────────────────────────
    def run_infer_stream(self, audio_tensor, output_dir):
        all_audio = []
        first_audio_time = None
        self.draft_model.stream_resume_state = None
        self.target_model.stream_resume_state = None
        self.audio_tokenizer.audio_decoder.reset_dict()
        self.audio_tokenizer.audio_decoder.flow.reset_step_cache(True, device="cuda")
        start_time = time.perf_counter()

        # ── 构建 input_ids ───────────────────────────────────────────────
        if audio_tensor is not None:
            messages = self.system_message + [
                {"role": "user", "content": message + "\n<|audio|>"}]
        else:
            messages = self.system_message + [
                {"role": "user", "content": message}]

        if (audio_tensor is not None
                and self.audio_tokenizer.apply_to_role("user", is_discrete=True)):
            atoks = self.audio_tokenizer.encode(audio_tensor)
            atoks_str = "".join(f"<|audio_{i}|>" for i in atoks)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{atoks_str}<|end_of_audio|>")

        input_ids = self.draft_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt)

        if (audio_tensor is not None
                and self.audio_tokenizer.apply_to_role("user", is_contiguous=True)):
            print(f"{audio_tensor=}")
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, [audio_tensor],
                self.draft_tokenizer, self.audio_tokenizer)
        else:
            audios, audio_indices = None, None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
        self.draft_model.generation_config.do_sample = False
        self.target_model.generation_config.do_sample = False

        # ════════════════════════════════════════════════════════════════
        # 阶段一: Draft 生成 8 token + 投机验证循环
        #
        # 目标序列: t0(M) a0(m) a1(m) a2(m) a3(m) t5(M) t6(M) t7(M)
        # 只验证 text token (位置 0,5,6,7)，循环直到全部确认
        #
        # 每次验证失败后，根据纠正点构造 confirmed_prefix，
        # 调用 draft_generate_for_sd 重新 prefill + 生成剩余 token。
        # draft_generate_for_sd 内部每次完全重置 MTP 状态机、新建
        # DynamicCache，不存在脏缓存问题。
        # ════════════════════════════════════════════════════════════════
        #print("[Phase 1] Draft generating first 8 tokens ...")

        # 初始: prefix 为空，从头生成 8 个 token
        confirmed_prefix = torch.zeros(1, 0, dtype=torch.long, device="cuda")
        draft_toks, draft_logits = self.draft_model.draft_generate_for_sd(
            input_ids, confirmed_prefix, n_to_generate=8, do_sample=False)

        confirmed_text_count = 0
        max_rounds = 10

        for round_idx in range(1, max_rounds + 1):
            if confirmed_text_count >= 4:
                break

            

            # ── 验证全部 4 个 text token ─────────────────────────────────
            accepted, num_accepted = self._get_text_accepted(
                input_ids, draft_toks, draft_logits)

            print(f"    num_accepted={num_accepted}")

            # num_accepted 语义（K=4 个 text token）:
            #   5 → 全部 4 个接受 + bonus token
            #   4 → 前 3 个接受 + 第 4 个纠正
            #   3 → 前 2 个接受 + 第 3 个纠正
            #   2 → 前 1 个接受 + 第 2 个纠正
            #   1 → 第 1 个就纠正
            #breakpoint()
            if num_accepted >= 5:
                # 全部接受（bonus token 不使用）
                for i in range(4):
                    draft_toks[self.TEXT_POSITIONS[i]] = accepted[i].view(1)
                confirmed_text_count = 4
                
                break

            # ── 部分接受 + 纠正 ──────────────────────────────────────────
            # accepted[0 .. num_accepted-1]:
            #   前 num_accepted-1 个 = 原样接受
            #   第 num_accepted 个   = 纠正 token（从 target 残差分布采样）
            for i in range(num_accepted):
                draft_toks[self.TEXT_POSITIONS[i]] = \
                    accepted[i].view(1)
            confirmed_text_count = num_accepted

            if confirmed_text_count >= 4:
                
                break

            # ── 构造 confirmed_prefix，重新 prefill + 生成剩余 ──────────
            # 最后一个确认 text token 在 8-token 序列中的绝对位置
            last_pos = self.TEXT_POSITIONS[confirmed_text_count - 1]

            # confirmed_prefix = draft_toks[0 .. last_pos]
            # 包含已确认/纠正的 text 和中间的 audio token
            prefix_tensor = torch.cat(
                [t.view(1, 1) for t in draft_toks[:last_pos + 1]],
                dim=1)  # [1, last_pos+1]

            n_remaining = 8 - (last_pos + 1)
            

            new_toks, new_logits = self.draft_model.draft_generate_for_sd(
                input_ids, prefix_tensor, n_to_generate=n_remaining,
                do_sample=False)

            # 更新 draft_toks & draft_logits 的尾部
            for i, pos in enumerate(range(last_pos + 1, 8)):
                draft_toks[pos] = new_toks[i]
                draft_logits[pos] = new_logits[i]

        assert confirmed_text_count >= 4, \
            f"Failed to confirm 4 text tokens in {max_rounds} rounds"

        

        # ════════════════════════════════════════════════════════════════
        # 阶段二: Draft 生成 a8-a15 (MTP × 8)
        #
        # prefix = 全部 8 个已确认 token，从调度表步骤 8 开始全是 m
        # draft_generate_for_sd 会 prefill [prompt | 8 tokens]
        # 然后 num_decode_tokens=8 → 调度表位置 0-7 (M mmmm MMM) 在 prefill 消费
        # 续接从位置 8 开始: mmmmmmmm
        # ════════════════════════════════════════════════════════════════
        

        all8_prefix = torch.cat(
            [t.view(1, 1) for t in draft_toks], dim=1)  # [1, 8]

        audio_new_toks, audio_new_logits = self.draft_model.draft_generate_for_sd(
            input_ids, all8_prefix, n_to_generate=8, do_sample=False)

        # 合并: 前 8 + 后 8 = 16 个 token
        all_draft_toks = draft_toks + audio_new_toks

        print(f"[Phase 2 Done] 16 tokens ready, "
              f"{time.perf_counter()-start_time:.4f}s")

        # ════════════════════════════════════════════════════════════════
        # 阶段三: Vocoder 生成首段音频
        # ════════════════════════════════════════════════════════════════
        generated_text = self.draft_tokenizer.decode(
            torch.cat([t.view(-1) for t in all_draft_toks]))
        audio_tokens = extract_token_ids_as_int(generated_text)
        print(f"[Phase 3] Vocoder: {len(audio_tokens)} audio tokens")

        self.audio_tokenizer.audio_decoder.flow.reset_step_cache(True, device="cuda")
        tts_speech = self.audio_tokenizer.decode(
                        audio_tokens,source_speech_16k=prompt_audio_path,
                    option_steps=10,)

        past_tts_speech_len = 0
        past_audio_token_len = 0
        num_audio_chunk = 0

        if tts_speech is not None:
            new_tts = tts_speech[past_tts_speech_len:]
            tts_np = new_tts.squeeze().float().cpu().numpy()
            mx = np.max(np.abs(tts_np))
            if mx > 0: tts_np = tts_np / mx
            all_audio.append((tts_np * 32767).astype(np.int16))
            first_audio_time = time.perf_counter() - start_time
            logger.info(f"First audio time: {first_audio_time}")
            
            past_tts_speech_len = len(tts_speech)
            past_audio_token_len = len(audio_tokens)
            num_audio_chunk += 1

        # ════════════════════════════════════════════════════════════════
        # 阶段四: Target model 接管（复用 draft_prefill_and_stream_generate）
        #
        # partial_text_tokens = [t5, t6, t7] (K=3, 全部已验证)
        # draft_prefill_and_stream_generate 内部:
        #   Phase1: prefill [prompt | t0 | a0-a3 | t5 t6 t7]
        #   Phase3: MTP 注入 a8-a15
        #   Phase4: 正常续接生成
        # ════════════════════════════════════════════════════════════════
        

        partial_text = torch.cat(
            [all_draft_toks[5].view(1, 1),
             all_draft_toks[6].view(1, 1),
             all_draft_toks[7].view(1, 1)], dim=1)  # [1, 3]

        seam_step = True
        target_toks = []
        full_text=""
        for tok, logit in self.target_model.draft_prefill_and_stream_generate(
            input_ids=input_ids,
            first_text_token=all_draft_toks[0].view(1, 1),
            first_audio_tokens=torch.cat(
                [t.view(1) for t in all_draft_toks[1:5]]).unsqueeze(0),
            partial_text_tokens=partial_text,
            second_audio_tokens=torch.cat(
                [t.view(1) for t in all_draft_toks[8:16]]).unsqueeze(0),
            max_new_tokens=8192,
            do_sample=False,
            eos_token_id=[151645, 151643],
            return_past_key_values=False,
        ):
            if seam_step:
                target_toks.append(tok)
                generated_text = self.target_tokenizer.decode(
                    torch.cat(target_toks)[0]
                    if torch.cat(target_toks).dim() > 1
                    else torch.cat(target_toks))
                full_text = remove_audio_tokens(generated_text)
                seam_step = False
            else:
                target_toks.append(tok)
                new_text = self.draft_tokenizer.decode(target_toks[-1])
                generated_text += new_text
                full_text += new_text
                if "<|end_of_audio|>" == new_text:
                    self.audio_tokenizer.audio_decoder.flow.reset_step_cache(
                        True, device="cuda")
                    audio_tokens = extract_token_ids_as_int(generated_text)

                    if num_audio_chunk == 0 and len(audio_tokens) == 8:
                        pass
                    elif len(audio_tokens) - past_audio_token_len > 16:
                        pass
                    else:
                        continue

                    tts_speech = self.audio_tokenizer.decode(
                        audio_tokens,source_speech_16k=prompt_audio_path,
                    option_steps=10,)
                    if tts_speech is not None:
                        new_tts = tts_speech[past_tts_speech_len:]
                        tts_np = new_tts.squeeze().float().cpu().numpy()
                        mx = np.max(np.abs(tts_np))
                        if mx > 0: tts_np = tts_np / mx
                        all_audio.append((tts_np * 32767).astype(np.int16))

                        if num_audio_chunk == 0:
                            first_audio_time = time.perf_counter() - start_time
                            logger.info(f"First audio time: {first_audio_time}")

                        past_tts_speech_len = len(tts_speech)
                        past_audio_token_len = len(audio_tokens)
                        if len(audio_tokens) > 512:
                            generated_text = ""
                            past_tts_speech_len = 0
                            past_audio_token_len = 0
                        num_audio_chunk += 1

        # ── 保存音频 ─────────────────────────────────────────────────────
        full_text = remove_audio_tokens(full_text)
        print(f"generated text: {full_text}")
        base_name = "output.wav"
        if len(all_audio) > 0:
            output_data = np.concatenate(all_audio)
            tensor = torch.from_numpy(output_data.astype("int16")).unsqueeze(0)
            base_name = os.path.basename(audio_tensor).replace("mp3", "wav")
            torchaudio.save(f"{output_dir}/{base_name}", tensor, 22050,
                            encoding="PCM_S", bits_per_sample=16)
        print("first_audio_time: ", first_audio_time)
        return {"text": full_text, "audio": f"{output_dir}/{base_name}"}, first_audio_time


if __name__ == "__main__":
    import os, json, math

    # ====== paths ======
    audio_folder = "/aplca/eval_datas/alpaca_eval/audios"
    time_json_path = ""
    # 只保存每个音频第一次生成的文本（jsonl，一行一个json），不存在会自动创建
    text_jsonl_path = ""
    audio_save_path= ""
    data_csv_path = "aplca/eval_datas/alpaca_eval/alpaca_eval.csv"
    os.makedirs(os.path.dirname(text_jsonl_path), exist_ok=True)
    vita=VitaPureSpecDecode(draft_model_name_or_path=draft_model_name_or_path,
                            target_model_name_or_path=target_model_name_or_path,device_map=device_map)
    
    with open(time_json_path, "r") as f:
        num_dict = json.load(f)
    
    

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

    
    total_time_list = []
    # ====== evaluation loop ======
    for i in range( len(audio_path_list)):
        time_list = []

        wav_path = audio_path_list[i]
        print(f"[{i+1}/{len(audio_path_list)}] {wav_path}")

        
            
            
        response, first_audio_time = vita.run_infer_stream(wav_path,audio_save_path)
        

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
        
        num_dict["sd_time"] = total_time_list
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
    
    