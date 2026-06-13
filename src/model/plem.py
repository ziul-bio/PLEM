import torch
from torch import nn
import math
import torch.nn.functional as F
from dataclasses import dataclass



# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Upcast to float32 for numerical stability, then cast back
        x_float = x.float()
        norm = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * norm * self.weight.float()).type_as(x)



# ---------------------------------------------------------------------------
# Rotary positional encoding
# ---------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for attention mechanisms.
    I addpated this to work with longer sequences than pre set contex length of 2048 tokens."""
    def __init__(self, dim_emb: int, base: int = 10000):
        super().__init__()
        self.dim_emb = dim_emb
        self.base = base
        # Cache is rebuilt on first forward if the model moves to a different device.
        cached_max_len = 2048
        self._build_cache(cached_max_len, device='cpu')

    def _build_cache(self, seq_len: int, device: torch.device):
        indices = torch.arange(seq_len, dtype=torch.float32, device=device)
        scale   = 1.0 / (self.base ** (
            torch.arange(0, self.dim_emb, 2, dtype=torch.float32, device=device) / self.dim_emb
        ))
        pos = torch.outer(indices, scale)
        pos = torch.cat((pos, pos), dim=-1)          # (seq_len, dim_emb)
        # register_buffer keeps the cache device-synced with the rest of the model
        # persistent=False: excluded cache from state_dict — recomputed on load.
        self.register_buffer("position_cos", torch.cos(pos)[None, None], persistent=False)
        self.register_buffer("position_sin", torch.sin(pos)[None, None], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, seq_len, dim_emb)
        seq_len = x.size(2)

        # Rebuild if sequence is longer than cached, or if model moved to a new device
        if seq_len > self.position_cos.size(2) or self.position_cos.device != x.device:
            self._build_cache(max(seq_len, self.position_cos.size(2)), device=x.device)

        cos = self.position_cos[:, :, :seq_len, :].to(x.dtype)
        sin = self.position_sin[:, :, :seq_len, :].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# MultiHeadAttention
#   layernorm_qkv: Sequential(LayerNorm, Linear)  — norm is fused here
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["hidden_size"] % config["n_head"] == 0

        self.n_head    = config["n_head"]
        self.hidden_size = config["hidden_size"]
        self.head_dim  = self.hidden_size // self.n_head
        #self.dropout   = config["dropout"]
        self.bias      = config["bias"]

        # Combined QKV projection
        # self.qkv_ln = nn.LayerNorm(self.hidden_size, bias=self.bias)
        # self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=self.bias)
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, self.hidden_size * 3, bias=self.bias)
        )
        
        # QK Norm 
        self.q_ln = nn.LayerNorm(self.head_dim, bias=self.bias)
        self.k_ln = nn.LayerNorm(self.head_dim, bias=self.bias)
       
        self.RoPE = RotaryPositionalEncoding(self.head_dim)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        #self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Project and split into Q, K, V
        qkv = self.layernorm_qkv(x)
        #qkv = self.qkv_proj(x) # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape from (B, T, C) to (B, T, n_head, head_dim) and then transpose dim 1 and 2 (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Apply QK Norm
        q = self.q_ln(q)
        k = self.k_ln(k)

        # 4. Apply RoPE (Now the shapes match perfectly: B, nh, T, hs)
        q = self.RoPE(q)
        k = self.RoPE(k)

        # 5. Scaled Dot Product Attention
        # This first implementation creates "islands" of attention for the padding tokens, because when pad vs pad is 0, it returns true.
        # we can see that the padding tokens at the end of the sentence will actually attend to each other, when using seq of different lenght.
        # attention_mask = attn_mask.unsqueeze(-1) == attn_mask.unsqueeze(-2)
        # grid_mask = attention_mask.unsqueeze(1)

        # This second implementation avoids that, the bptton is also false.
        # Logic: (B, 1, L, 1) AND (B, 1, 1, L) -> (B, 1, L, L)
        attention_mask = attn_mask.bool()
        grid_mask = attention_mask.unsqueeze(1).unsqueeze(2) & attention_mask.unsqueeze(1).unsqueeze(3)

        # print('shape from MHA')
        # print(grid_mask)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=grid_mask)

        # 6. Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 7. Final output projection
        out = self.out_proj(out)
        #out = self.resid_dropout(out)
        return out


# ---------------------------------------------------------------------------
# Per-Layer Embeddings
# ---------------------------------------------------------------------------
class PerLayerEmbeddings(nn.Module):
    """This class creates for each token a small dedicated vector for every layer by combining two signals:
    1. token-identity component (from an embedding lookup) and a 
    2. context-aware component (from a learned projection of the main embeddings).
    """
    def __init__(self, config):
        super().__init__()

        self.ple_embed_lookup_table   = nn.Embedding(config['vocab_size'], config['n_layers'] * config['ple_dim'])
        self.ple_embed_identity_scale = config['ple_dim'] ** 0.5
        self.ple_embed_context        = nn.Linear(config['hidden_size'], config['n_layers'] * config['ple_dim'], bias=False)
        self.ple_embed_context_norm   = RMSNorm(config['ple_dim'])
        self.context_scale            = config['hidden_size'] ** -0.5
        self.combine_scale            = 2.0 ** -0.5
        self.n_layers               = config['n_layers']
        self.ple_embed_dim            = config['ple_dim']

    def forward(self, tokens, inputs_embeds):
        b, t = tokens.shape
        n, d = self.n_layers, self.ple_embed_dim

        identity = self.ple_embed_lookup_table(tokens) * self.ple_embed_identity_scale
        identity = identity.view(b, t, n, d)

        context = self.ple_embed_context(inputs_embeds) * self.context_scale
        context = context.view(b, t, n, d)
        context = self.ple_embed_context_norm(context)

        combined = (context + identity) * self.combine_scale
        return combined.permute(2, 0, 1, 3)   # (n_layers, B, T, ple_dim)


class PLEModulation(nn.Module):
    """This class uses the layer_specific_ple to modulate the hidden states
    via a lightweight residual block after attention and feed-forward.
    """
    def __init__(self, config):
        super().__init__()
        self.proj_down    = nn.Linear(config['hidden_size'], config['ple_dim'], bias=False)
        self.act_fn       = nn.SiLU()
        self.proj_up      = nn.Linear(config['ple_dim'], config['hidden_size'], bias=False)
        self.norm         = RMSNorm(config['hidden_size'])
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, layer_specific_ple):
        residual = x
        x = self.act_fn(self.proj_down(x))
        x = x * layer_specific_ple
        x = self.proj_up(x)
        x = self.norm(x)
        x = residual + x
        return x * self.layer_scalar


# ---------------------------------------------------------------------------
# SwiGLU  — gating only; the up-projection Linear lives in the FFN Sequential
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Splits the last dim in half: out = first_half * silu(second_half)."""
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)

# class SwiGLU(nn.Module):
#     """SwiGLU(x) = (xW + b) ⊗ swish(xZ + c)
#     Projects from dim_in up to dim_out*2, gates, returns dim_out.
#     """
#     def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
#         super().__init__()
#         self.dim_out = dim_out
#         # Single matrix projects to 2*dim_out: one half is value, one half is gate
#         self.linear = nn.Linear(dim_in, 2 * dim_out, bias=bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear(x)
#         # The output is split into two halves: First half: value. Second half: gate (swish input)
#         value, gate = x[..., :self.dim_out], x[..., self.dim_out:]
#         return value * F.silu(gate)


# ---------------------------------------------------------------------------
# FFN
#   LayerNorm -> Linear (up, x(expansion*2) for SwiGLU) -> SwiGLU -> Linear (down)
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """ The * 2 / 3 factor is the SwiGLU correction.
    Since SwiGLU halves the hidden dim, I need to scale up by 2/3 relative 
    to a standard FFN to land at the same effective hidden size as a plain expansion_ratio x d_model FFN.
    The // 256 * 256 rounds up to the nearest multiple of 256 for tensor core efficiency.
    
    So for d_model=960, expansion_ratio=4:
    960 * 4 * 2/3 = 2560 -> already a multiple of 256, so ffn_hidden = 2560
    Linear up: 960 -> 5120 (i.e. 2560 * 2)
    SwiGLU halves: 5120 -> 2560
    Linear down: 2560 -> 960"""
    def __init__(self, config):
        super().__init__()
        expansion_ratio = config.get("ffn_expansion_ratio", 4)
        bias            = config["bias"]
        embed_dim       = config["hidden_size"]
        ffn_hidden      = int((expansion_ratio * embed_dim * 2/3 + 255) // 256 * 256)

        self.ln      = nn.LayerNorm(embed_dim)
        self.up_proj = nn.Linear(embed_dim, ffn_hidden * 2, bias=config['bias'])
        self.swiglu   = SwiGLU()
        self.down_proj = nn.Linear(ffn_hidden, embed_dim, bias=config['bias'])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.up_proj(x)      # up-project                 (hidden_size -> ffn_dim)
        x = self.swiglu(x)       # SwiGLU gate
        x = self.down_proj(x)    # down-project              (ffn_dim -> hidden_size)
        return x

# class FFN(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         expansion_factor = 4
#         ffn_dim = config['hidden_size'] * expansion_factor
        
#         self.ln = nn.LayerNorm(config['hidden_size'])
    
#         # SwiGLU handles the up-projection (hidden_size → ffn_dim) and gating internally
#         self.swiglu   = SwiGLU(config['hidden_size'], ffn_dim, bias=config['bias'])
#         # Down-projection back to hidden_size
#         self.out_proj = nn.Linear(ffn_dim, config['hidden_size'], bias=config['bias'])
#         #self.dropout  = nn.Dropout(config['dropout'])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.ln(x)
#         x = self.swiglu(x)      # up-project + SwiGLU gate  (hidden_size → ffn_dim)
#         x = self.out_proj(x)    # down-project              (ffn_dim → hidden_size)
#         #x = self.dropout(x)
#         return x



# ---------------------------------------------------------------------------
# TransformerBlock  — norm is absorbed into attn/ffn sub-modules
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.ln_1    = RMSNorm(config['hidden_size'])
        self.attn    = MultiHeadAttention(config)
        #self.ln_2    = RMSNorm(config['hidden_size'])
        self.ffn     = FFN(config)
        self.scaling_factor = math.sqrt(config["n_layers"] / 36) if config.get("scale_residue", True) else 1.0
        #self.ln_3    = RMSNorm(config['hidden_size'])
        self.ple_module = PLEModulation(config) 

    def forward(self, x, layer_specific_ple, attn_mask):
        #x = self.ln_1(x)
        x = x + self.attn(x, attn_mask)
        #x = self.ln_2(x)
        x = x + self.ffn(x) / self.scaling_factor
        #x = self.ln_3(x) # Gemma4 does not have this prenorm
        x = self.ple_module(x, layer_specific_ple)
        return x


# ---------------------------------------------------------------------------
# TransformerStack
# ---------------------------------------------------------------------------
class TransformerStack(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.norm = nn.LayerNorm(config["hidden_size"])

    def forward(self, x: torch.Tensor, ple_all: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        hiddens = []
        for i, block in enumerate(self.blocks):
            x = block(x, ple_all[i], attn_mask)
            hiddens.append(x)
        return self.norm(x), hiddens



# ---------------------------------------------------------------------------
# Language Model head 
# ---------------------------------------------------------------------------

class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.proj_final = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act(x)
        x = self.proj_final(self.ln(x))
        return x



# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelOutput:
    logits: torch.Tensor                          # (B, T, vocab_size)
    embeddings: torch.Tensor | None = None        # (B, T, d_model)  or None
    mean_embeddings: torch.Tensor | None = None   # (B, d_model)  or None
    hidden_states: torch.Tensor | None = None     # (n_layers, B, T, d_model) or None


# ---------------------------------------------------------------------------
# BERT with PLE — model
# ---------------------------------------------------------------------------
class FLARE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed          = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.ple_embed      = PerLayerEmbeddings(config) 
        self.transformer    = TransformerStack(config)
        self.lm_head        = LMHead(config["hidden_size"], config["vocab_size"])

    def forward(
            self, 
            tokens: torch.Tensor,
            attention_mask = torch.Tensor,           
            return_embeddings: bool = False,
            return_mean_embeddings: bool = False,
            return_hidden_states: bool = False,

    ) -> ModelOutput:
        
        # 1. Embed tokens
        inputs_embeds = self.embed(tokens)
        
        # 2. PLE 
        ple_all = self.ple_embed(tokens, inputs_embeds)   # (n_layers, B, T, ple_dim)

        # 3. Transformer stack
        x, hiddens = self.transformer(inputs_embeds, ple_all, attention_mask)
        
        # 4. Logits
        logits = self.lm_head(x)

        # 5. Mean embeddings (over non-special tokens)
        mean_representations = None
        if return_mean_embeddings:
            seq_lengths = attention_mask.sum(dim=1) - 2    # exclude CLS and EOS
            reps = []
            for i in range(x.size(0)):
                reps.append(x[i, 1 : seq_lengths[i] + 1, :].mean(dim=0))
            mean_representations = torch.stack(reps)       # (B, embed_dim)

        return ModelOutput(
            logits=logits,
            embeddings= x if return_embeddings else None,
            mean_embeddings = mean_representations if return_mean_embeddings else None,
            hidden_states=torch.stack(hiddens, dim=0) if return_hidden_states else None,
        )
    




if __name__ == "__main__":
    
    config = {
    "vocab_size"    : 30,
    "hidden_size"   : 640,
    "n_layers"      : 20,
    "n_head"        : 10,
    "ple_dim"       : 128,
    "bias"          : False,
}


    model = FLARE(config)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    tokens = torch.tensor([[0, 20, 15, 11, 4, 11, 4, 2], [0, 17, 9, 9, 2, 1, 1, 1]])
    attn_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]])
    out    = model(tokens, attn_mask, return_embeddings=True, return_hidden_states=False)
    
    # print(out.logits.shape)                         
    # print(out.embeddings.shape)                       









