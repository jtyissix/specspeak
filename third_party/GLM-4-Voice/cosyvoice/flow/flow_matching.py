# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM

class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.fp16=False

    def _prepare_cfg_inputs(self, x, mask, mu, t, spks, cond):
        """Expand a speech batch into conditioned/unconditioned CFG halves."""
        batch_size = x.size(0)
        if t.numel() == 1:
            t = t.expand(batch_size)
        else:
            t = t.reshape(-1)
            if t.numel() != batch_size:
                raise ValueError(
                    f"Expected one diffusion timestep or {batch_size} timesteps, "
                    f"got {t.numel()}"
                )

        x_in = torch.empty(
            (2 * batch_size, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        mask_in = torch.empty(
            (2 * batch_size, *mask.shape[1:]),
            device=mask.device,
            dtype=mask.dtype,
        )
        mu_in = torch.zeros(
            (2 * batch_size, *mu.shape[1:]),
            device=mu.device,
            dtype=mu.dtype,
        )
        t_in = torch.empty(
            2 * batch_size,
            device=t.device,
            dtype=t.dtype,
        )
        spks_in = torch.zeros(
            (2 * batch_size, *spks.shape[1:]),
            device=spks.device,
            dtype=spks.dtype,
        )
        cond_in = torch.zeros(
            (2 * batch_size, *cond.shape[1:]),
            device=cond.device,
            dtype=cond.dtype,
        )

        x_in[:batch_size] = x
        x_in[batch_size:] = x
        mask_in[:batch_size] = mask
        mask_in[batch_size:] = mask
        mu_in[:batch_size] = mu
        t_in[:batch_size] = t
        t_in[batch_size:] = t
        spks_in[:batch_size] = spks
        cond_in[:batch_size] = cond
        return x_in, mask_in, mu_in, t_in, spks_in, cond_in
    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        torch.manual_seed(42)

        z = torch.randn_like(mu) * temperature
        if self.fp16 is True:
            z = z.half()
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        #breakpoint()
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                x_in, mask_in, mu_in, t_in, spks_in, cond_in = (
                    self._prepare_cfg_inputs(x, mask, mu, t, spks, cond)
                )
            else:
                x_in, mask_in, mu_in, t_in, spks_in, cond_in = x, mask, mu, t, spks, cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in
            )
            if self.inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()
    def forward_one_step(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2)):
        #our key framework: cross-token flow cache
        pass
    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)
        
        else:
            trt_dtype = torch.float16 if self.fp16 else torch.float32
        
            x    = x.to(trt_dtype).contiguous()
            mask = mask.to(trt_dtype).contiguous()
            mu   = mu.to(trt_dtype).contiguous()
            t    = t.to(trt_dtype).contiguous()
            spks = spks.to(trt_dtype).contiguous()
            cond = cond.to(trt_dtype).contiguous()
            
            output = torch.zeros_like(x)
            output = output.contiguous()
            trt_inputs = {
                'x': x,
                'mask': mask,
                'mu': mu,
                't': t,
                'spks': spks,
                'cond': cond,
            }
            for name, tensor in trt_inputs.items():
                shape = tuple(tensor.shape)
                if self.estimator.set_input_shape(name, shape) is False:
                    raise ValueError(
                        f"TensorRT profile rejected input {name} shape {shape}"
                    )
                self.estimator.set_tensor_address(name, tensor.data_ptr())
            self.estimator.set_tensor_address('estimator_out', output.data_ptr())
            # run trt engine
            stream = torch.cuda.current_stream(x.device).cuda_stream
            '''
            self.estimator.execute_v2([x.contiguous().data_ptr(),
                                       mask.contiguous().data_ptr(),
                                       mu.contiguous().data_ptr(),
                                       t.contiguous().data_ptr(),
                                       spks.contiguous().data_ptr(),
                                       cond.contiguous().data_ptr(),
                                       x.data_ptr()])
            '''
            #breakpoint()
            if not self.estimator.execute_async_v3(stream):
                raise RuntimeError(
                    f"TensorRT estimator execution failed for shape {tuple(x.shape)}"
                )
            #torch.cuda.current_stream(x.device).synchronize()
            return output
            #return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y
    def forward_one_step(self, now_cache,mu, mask,t_span, now_steps, temperature=1.0, spks=None, cond=None):
        #our key framework: cross-token flow cache
        t, _, dt = t_span[now_steps], t_span[-1], t_span[now_steps+1] - t_span[now_steps]
        #print(dt)
        t = t.unsqueeze(dim=0)
        x=now_cache
        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        

        # Classifier-Free Guidance inference introduced in VoiceBox
        if self.inference_cfg_rate > 0:
            x_in, mask_in, mu_in, t_in, spks_in, cond_in = (
                self._prepare_cfg_inputs(x, mask, mu, t, spks, cond)
            )
        else:
            x_in, mask_in, mu_in, t_in, spks_in, cond_in = x, mask, mu, t, spks, cond
        dphi_dt = self.forward_estimator(
            x_in, mask_in,
            mu_in, t_in,
            spks_in,
            cond_in
        )
        if self.inference_cfg_rate > 0:
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
        x = x + dt * dphi_dt
        #t = t + dt
        

        return x,None
