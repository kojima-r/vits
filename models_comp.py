import math
import torch

import commons

from models import SynthesizerTrn, TextEncoder

class TextEncoderComp(TextEncoder):
  def forward(self, x, x_lengths):
    if type(x) is list:
      x = self.idlist_to_embed(x, device=x_lengths.device) * math.sqrt(self.hidden_channels) # [b, t, h]
    else:
      x = self.emb(x) * math.sqrt(self.hidden_channels)
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask
  def idlist_to_embed(self, x, device):
    embeds=torch.tensor([]).to(device)
    for id in x:
      if type(id) is int:
        new_embed = self.emb( torch.tensor( [id] ).to(device) )[0]
      elif type(id) is dict:
        old_embed=self.emb( torch.tensor( list(id.keys()) ).to(device) )
        new_embed = torch.zeros(old_embed[0].shape).to(device)
        for i,k in enumerate(id.keys()):
          new_embed += old_embed[i] * id[k]
      embeds=torch.cat([embeds,new_embed[None]])
    return embeds[None]

class SynthesizerTrnComp(SynthesizerTrn):
  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    **kwargs):
    
    super().__init__(n_vocab,spec_channels,segment_size,inter_channels,hidden_channels,filter_channels,n_heads,n_layers,kernel_size,p_dropout,resblock,resblock_kernel_sizes,resblock_dilation_sizes,upsample_rates,upsample_initial_channel,upsample_kernel_sizes,n_speakers,gin_channels,use_sdp,**kwargs)
    
    self.enc_p = TextEncoderComp(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
  def inferComp(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None, w_ceil=None, g=None):
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      if g is None:
        g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    if w_ceil is None:
      if self.use_sdp:
        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
      else:
        logw = self.dp(x, x_mask, g=g)
      w = torch.exp(logw) * x_mask * length_scale
      w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)