import torch
import torchaudio
import numpy as np
import re
from hyperpyyaml import load_hyperpyyaml
import uuid
from collections import defaultdict


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class AudioDecoder:
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, load_trt=False,trt_path=None,device="cuda"):
        self.device = device
        
        with open(config_path, 'r') as f:
            self.scratch_configs = load_hyperpyyaml(f)

        # Load models
        self.flow = self.scratch_configs['flow']
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device))
        self.hift = self.scratch_configs['hift']
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        if load_trt:
            del self.flow.decoder.estimator
            import tensorrt as trt
            
            #import pycuda.driver as cuda
            torch.cuda.set_device(0)
            with open(trt_path, 'rb') as f:
                self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
            self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()
            if 'fp16' in trt_path:
                self.flow.decoder.fp16 = True
                #self.flow.to(dtype=torch.float16)
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
    def reset_dict(self):
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)

    def validate_trt_batch_size(self, speech_batch_size):
        """Validate the CFG-expanded batch against the loaded TRT profile."""
        if speech_batch_size <= 0:
            raise ValueError(
                f"speech_batch_size must be positive, got {speech_batch_size}"
            )
        if not hasattr(self.flow.decoder, "estimator_engine"):
            return

        cfg_multiplier = 2 if self.flow.decoder.inference_cfg_rate > 0 else 1
        expected_batch = cfg_multiplier * speech_batch_size
        engine = self.flow.decoder.estimator_engine
        declared_shape = tuple(engine.get_tensor_shape("x"))

        if declared_shape[0] >= 0:
            min_batch = max_batch = declared_shape[0]
        else:
            try:
                min_shape, _, max_shape = engine.get_tensor_profile_shape("x", 0)
            except (AttributeError, TypeError):
                min_shape, _, max_shape = engine.get_profile_shape(0, "x")
            min_batch = min_shape[0]
            max_batch = max_shape[0]

        if not min_batch <= expected_batch <= max_batch:
            raise ValueError(
                "TensorRT estimator batch mismatch: speech batch "
                f"{speech_batch_size} with CFG requires estimator batch "
                f"{expected_batch}, but engine profile supports "
                f"{min_batch}..{max_batch}. Keep the configured path and "
                "replace its .plan with one built for 2 * BATCH_SIZE."
            )

    def _validate_batch_inputs(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
    ):
        batch_size = token.shape[0]
        if not (
            token_len.shape == (batch_size,)
            and prompt_token.shape[0] == batch_size
            and prompt_token_len.shape == (batch_size,)
            and prompt_feat.shape[0] == batch_size
            and prompt_feat_len.shape == (batch_size,)
            and embedding.shape[0] == batch_size
        ):
            raise ValueError("All decoder inputs must use the same batch size")
        self.validate_trt_batch_size(batch_size)
        return batch_size

    def token2wav_batch(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        option_steps=10,
    ):
        """Decode one fixed-size speech batch through Flow and HiFT."""
        batch_size = self._validate_batch_inputs(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
        )
        tts_mel = self.flow.inference(
            token=token.to(self.device),
            token_len=token_len.to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=prompt_token_len.to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=prompt_feat_len.to(self.device),
            embedding=embedding.to(self.device),
            option_steps=option_steps,
        )
        empty_source_cache = torch.zeros(
            batch_size,
            1,
            0,
            device=self.device,
            dtype=tts_mel.dtype,
        )
        tts_speech, _ = self.hift.inference(
            mel=tts_mel,
            cache_source=empty_source_cache,
        )
        return tts_speech, tts_mel

    def token2wav_one_step_batch(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        option_steps=10,
        validate_inputs=True,
    ):
        """Advance a speech batch by one diffusion step."""
        if validate_inputs:
            batch_size = self._validate_batch_inputs(
                token,
                token_len,
                prompt_token,
                prompt_token_len,
                prompt_feat,
                prompt_feat_len,
                embedding,
            )
        else:
            batch_size = token.shape[0]
        tts_mel = self.flow.inference_one_step(
            token=token.to(self.device),
            token_len=token_len.to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=prompt_token_len.to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=prompt_feat_len.to(self.device),
            embedding=embedding.to(self.device),
        )
        if tts_mel is None:
            return None
        empty_source_cache = torch.zeros(
            batch_size,
            1,
            0,
            device=self.device,
            dtype=tts_mel.dtype,
        )
        tts_speech, _ = self.hift.inference(
            mel=tts_mel,
            cache_source=empty_source_cache,
        )
        return tts_speech, tts_mel

    def trim_batch_speech(self, tts_speech, token_len):
        """Remove waveform padding introduced by a variable-length token batch."""
        feat_len = (
            token_len
            / self.flow.input_frame_rate
            * 22050
            / 256
        ).int()
        max_feat_len = max(1, int(feat_len.max().item()))
        full_speech_len = tts_speech.shape[-1]
        speech_lengths = (
            feat_len.to(torch.long) * full_speech_len // max_feat_len
        )
        return [
            tts_speech[index, :int(speech_lengths[index].item())]
            .squeeze()
            .cpu()
            for index in range(token_len.shape[0])
        ]
    def token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192), finalize=False, option_steps=10):
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device),
                                      option_steps=option_steps)

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)

        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            # if self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            tts_speech = tts_speech[:, :-self.source_cache_len]

        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
            # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel
    
    def token2wav_one_step(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192), finalize=False, option_steps=10):
        tts_mel = self.flow.inference_one_step(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device),
                                      )
        if tts_mel is None:
            return None
        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)

        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            # if self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            tts_speech = tts_speech[:, :-self.source_cache_len]

        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
            # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel
    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def stream_inference(self, token):
        token.to(self.device)
        this_uuid = str(uuid.uuid1())

        # Prepare other necessary input tensors
        llm_embedding = torch.zeros(1, 192).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)

        tts_speechs = []
        tts_mels = []

        block_size = self.flow.encoder.block_size
        prev_mel = None

        for idx in range(0, token.size(1), block_size):
            # if idx>block_size: break
            tts_token = token[:, idx:idx + block_size]

            print(tts_token.size())

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=flow_prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device), finalize=is_finalize)

            prev_mel = tts_mel
            prev_speech = tts_speech
            print(tts_mel.size())

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech.cpu()

