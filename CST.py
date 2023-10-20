import torch
from torch import nn
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from collections import OrderedDict


# helpers

def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def default(val, d):
    return val if exists(val) else d

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., T=0, ):
        super().__init__()
        self.T = 6

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head, )

        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()
        b = (bt - 1) // self.T
        x = x.view(l, b, self.T, d)

        msg_token = self.message_fc(x[0, :, :, :])
        msg_token = msg_token.view(b, self.T, 1, d)

        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b, d)
        msg_token = msg_token + self.drop_path(
            self.message_attn(self.message_ln(msg_token), self.message_ln(msg_token), self.message_ln(msg_token),
                              need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1, 2, 0, 3)

        x = torch.cat([x, msg_token], dim=0)

        x = x.view(l + 1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l, :, :]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class C0_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossTransformer(nn.Module):
    def __init__(self, dim, cross_depth, cross_head, cross_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(cross_depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, C0_Attention(dim, heads = cross_head, dim_head = cross_dim, dropout = dropout)),
                PreNorm(dim, C0_Attention(dim, heads = cross_head, dim_head = cross_dim, dropout = dropout))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))
        ### lg_cls: [32, 1, 2048]; sm_cls: [32, 1, 2048];  lg_patch_tokens: [32, 54, 2048];
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CST_module(nn.Module):
    def __init__(
            self,
            *,
            image_size_h,
            image_size_w,
            patch_size_h,
            patch_size_w,
            frames,
            frame_patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=2048,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.
    ):
        super().__init__()
        # image_height, image_width = pair(image_size)
        image_height = image_size_h
        image_width = image_size_w

        # patch_height, patch_width = pair(patch_size)
        patch_height = patch_size_h
        patch_width = patch_size_w

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)

        # num_patches = (image_height // patch_height) * (image_width // patch_width)

        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.Crosstransformer = CrossTransformer(dim, depth, heads, dim_head, dropout).to('cuda')

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        ###——————————————————————————cross-frame attention fusion————————————————————————————————————————————
        self.T = frames

        self.message_fc = nn.Linear(dim, dim)
        self.message_ln = nn.LayerNorm(dim)
        self.message_attn = nn.MultiheadAttention(dim, dim, )

        self.attn = nn.MultiheadAttention(dim, 128, )
        self.ln_1 = nn.LayerNorm(dim)
        droppath = 0
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))
        self.ln_2 = nn.LayerNorm(dim)


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, video):

        #####------------------------------------------------------------------------------------
        #####------------------------cross-frame tube Transformer Module (CTTM)------------------------------------------------------------
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape
        b_temp = b
        n_split = n
        self.T = f

        x = x + self.pos_embedding[:, :, :(n), :]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b=b, f=f)  # [16, 6, 1, 2048];        ||||### [16, 4, 1, 2048]
            x = torch.cat((spatial_cls_tokens, x), dim=2)

        x = self.dropout(x)

        x = rearrange(x, 'b f n d -> (b f) n d')
        x = self.spatial_transformer(x)


        x_spat = x
        x_spat = x_spat.permute(1, 0, 2)

        ######-----------------------------Multi-frame Transformer Fusion Module-------------------------------------------------------------------------------------
        #### SENet Attention in model_main.py
        ###——————————————————————————Inter-frame Interaction attention  Module———————————————————————————————————————————-------------------—
        x_1 = x.permute(1, 0, 2)
        l, bt, d_c = x_1.size()
        b_c = bt // self.T

        x_cross = x_1.view(l, b_c, self.T, d_c)

        msg_token = self.message_fc(x_cross[0, :, :, :])
        msg_token = msg_token.view(b_c, self.T, 1, d_c)

        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b_c, d_c)
        msg_token = msg_token + self.drop_path(
            self.message_attn(self.message_ln(msg_token), self.message_ln(msg_token), self.message_ln(msg_token),
                              need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b_c, d_c).permute(1, 2, 0, 3)

        x_msg = torch.cat([x_cross, msg_token], dim=0)
        x_msg = x_msg.view(l + 1, -1, d_c)


        #### Itra-frame Fusion attention Module
        x_att = x_msg + self.attn(self.ln_1(x_msg), self.ln_1(x_msg), self.ln_1(x_msg))[0] ###

        x_att_patch = x_att[:l, :, :] ####

        x_fusion = x_att_patch + self.drop_path(self.mlp(self.ln_2(x_att_patch)))
        x_fusion = x_fusion.permute(1, 0, 2)
        ##############--------------------------------------------------------------------------------------------------

        #####--------------------------------Temproal Fusion Module---
        x = rearrange(x_fusion, '(b f) n d -> b f n d', b=b_temp)

        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b=b_temp)

            x = torch.cat((temporal_cls_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        x = self.to_latent(x)
        return x, self.mlp_head(x)