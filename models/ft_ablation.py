import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from .sdm import On_attention_gaussian_mask
# helpers
 
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), ** kwargs)
 
class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )            
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., mask=None,
                 is_cls=False, is_vt=True, is_ss=True, is_sdm=True, is_SA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)

        self.is_ss = is_ss
        self.is_sdm = is_sdm
        self.is_SA = is_SA

        if is_SA:
            self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
            init_weights(self.to_qkv)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if self.is_ss:
            if is_cls:
                self.dynamicqk = nn.Parameter(torch.zeros(1, 1, num_patches+1, num_patches+1), requires_grad=True)
            else:
                self.dynamicqk = nn.Parameter(torch.zeros(1, 1, num_patches, num_patches), requires_grad=True)
        if self.is_sdm:
            self.decay_mask = mask


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        if self.is_SA:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            attn = self.attend(dots)
        elif self.is_ss:
            v = x
            v = rearrange(v, 'b n (h d) -> b h n d', h = h)
            attn = self.dynamicqk
        
        # import pdb; pdb.set_trace()
        if self.is_sdm:
            attn = attn * self.decay_mask

        out = einsum('b h i j, b h j d -> b h i d', attn, v) 
                
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

        # v = x
        # v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        # out = (self.dynamicqk * self.decay_mask) @ v
            
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # return self.to_out(out)
    
    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches  
            flops += self.dim * self.inner_dim * 3  


class FlashFormer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., mask=None,
                 is_cls=False, is_vt=True, is_ss=True, is_sdm=True, is_SA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}
        self.norm = nn.LayerNorm(dim)
        self.is_vt = is_vt

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, mask=mask,is_cls=is_cls, is_ss=is_ss, is_sdm=is_sdm, is_SA=is_SA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout)),
            ]))   

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):

        b, n, d = x.shape

        for i, (attn, ff) in enumerate(self.layers):   
            if self.is_vt:
                vertical_token = x[:, 0, :].unsqueeze(dim=1)
                x = x[:, 1:, :]

            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x    
            self.scale[str(i)] = attn.fn.scale

            if self.is_vt:
                vertical_token = self.norm(x.mean(dim=1).unsqueeze(dim=1) + vertical_token)
                x = torch.cat((vertical_token, x), dim=1)
        return x  


        return x

class Flash_Ablation(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.,
                 is_cls=False, is_vt=True, is_pe=False, is_SA=False, is_ss=True, is_sdm=True):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes

        if is_cls:
            self.mask = nn.Parameter(On_attention_gaussian_mask(self.num_patches+1), requires_grad=False)
        else:
            self.mask = nn.Parameter(On_attention_gaussian_mask(self.num_patches), requires_grad=False)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(self.patch_dim, self.dim)
        )

        self.is_cls = is_cls
        self.is_vt = is_vt
        self.is_pe = is_pe
        self.is_SA = is_SA
        self.is_ss = is_ss
        self.is_sdm = is_sdm


        if is_vt:
            self.vertical_token = nn.Parameter(torch.zeros(1, 1, dim))

        if is_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        if is_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, self.dim))
            
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = FlashFormer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, 
                                       stochastic_depth, mask=self.mask, is_cls=is_cls, is_vt=is_vt,is_ss=is_ss, is_sdm=is_sdm, is_SA=is_SA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
            
        b, n, _ = x.shape

        if self.is_cls:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.is_vt:
            vertical_token = repeat(self.vertical_token, '() n d -> b n d', b = b)
            x = torch.cat((vertical_token, x), dim=1)

        if self.is_pe:
            x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        if self.is_cls:
            return self.mlp_head(x[:, 0])
        
        else:
            return self.mlp_head(x.mean(dim=1))
        


