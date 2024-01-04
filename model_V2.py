import torch.nn as nn 
import torch 
import torch.nn.functional as F
import math 

class Config():
    def __init__(self, vocab_size : int, emb_size : int, head_nb : int, block_nb : int, block_size : int, dropout : float = 0.0):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.head_nb = head_nb
        self.block_nb = block_nb
        self.block_size = block_size
        self.head_size = emb_size // head_nb
        self.attn_dropout = dropout
        self.multi_attn_dropout = dropout
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
        def __init__(self, config : Config) :
            super().__init__()
            self.n_head = config.head_nb
            self.head_size = config.head_size
            self.emb_size = config.emb_size
            self.c_attn = nn.Linear(self.emb_size, 3*self.emb_size, bias = False)
            self.c_proj = nn.Linear(self.emb_size, self.emb_size, bias = False)
            self.attn_dropout = nn.Dropout(config.attn_dropout)
            self.resid_dropout = nn.Dropout(config.attn_dropout)
            self.dropout = config.dropout
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.c_attn(x).split(self.emb_size, dim=2)

            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            y = self.resid_dropout(self.c_proj(y))            
            return y

class SingleAttentionHead(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.emb_size = config.emb_size
        self.head_size = config.head_size
        self.block_size = config.block_size
        self.q_linear = nn.Linear(self.emb_size, self.head_size, bias = False)
        self.v_linear = nn.Linear(self.emb_size, self.head_size, bias = False)
        self.k_linear = nn.Linear(self.emb_size, self.head_size, bias = False)


        self.dropout = nn.Dropout(config.attn_dropout)

        # Change
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(self.block_size,self.block_size))
        )
    
    def forward(self, x):
        B,T,C = x.shape

       
        
        k = self.k_linear(x)
        q = self.q_linear(x) # B, T, C

        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5) # B, T, T

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.v_linear(x)
        out = wei @ v # B, T, C


        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.head_nb = config.head_nb
        self.emb_size = config.emb_size
        self.block_size = config.block_size
        self.heads = nn.ModuleList([SingleAttentionHead(config) for _ in range(self.head_nb)])
        self.proj = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(config.multi_attn_dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, config : Config):
        super().__init__()

        #self.MHA = MultiHeadAttention(config)
        self.attn = CausalSelfAttention(config)
        self.feed_forward = nn.Sequential(
                nn.Linear(config.emb_size, 4*config.emb_size),
                nn.GELU(),
                nn.Linear(4*config.emb_size, config.emb_size),
                nn.Dropout(config.dropout)
        )

        self.ln1 = nn.LayerNorm(config.emb_size)
        self.ln2 = nn.LayerNorm(config.emb_size)

    def forward(self, x):
        out = self.ln1(x)
        out = self.attn(out)
        #out = self.MHA(out)
        out = out + x 
        out = self.ln2(out)
        out = self.feed_forward(out)
        out = out + x
        return out
    
# class FeedForward(nn.Sequential):
#     def __init__(self, config : Config):
#         super().__init__(
#         nn.Linear(config.emb_size, 4*emb_size, bias = False),
#         nn.ReLU(),
#         nn.Linear(4*emb_size, emb_size, bias = False),
#         nn.Dropout(0.1)
#         )
        
class LLM(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.emb_size = config.emb_size
        self.head_nb = config.head_nb
        self.block_nb = config.block_nb
        self.block_size = config.block_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_emb = nn.Embedding(self.block_size, config.emb_size)
        #self.blocks = nn.ModuleList([TransformerBlock(self.emb_size, self.head_nb, self.block_size) for _ in range(self.block_nb)])
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(self.block_nb)),
            nn.LayerNorm(config.emb_size)
        )

        self.dropout = nn.Dropout(config.dropout)
        # self.ln = nn.LayerNorm(self.emb_size)
  
        self.lm_head = nn.Linear(self.emb_size, config.vocab_size)

    def forward(self, x):
        B,T = x.shape
        
        tok_emb =  self.tok_emb(x)
        
        pos_emb = self.pos_emb(torch.arange(T, device = x.device))

        out = tok_emb + pos_emb

        out = self.dropout(out)
        # for block in self.blocks:
        #     out = block(out)
        out = self.blocks(out)

        #out = self.ln(out)
        logits = self.lm_head(out)

        return logits
    
    def generate(self, x, max_len = 100, temperature = 1.0):
        self.eval()
        B,T = x.shape
        with torch.no_grad():
            for _ in range(max_len):
                
                x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
       
                logits = self(x_cond)
            
                logits = logits[:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)

                next_tok = torch.multinomial(probs, num_samples=1)
                
                x = torch.cat([x, next_tok], dim=-1)
        return x
    
    