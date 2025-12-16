from functools import partial
from .layers import *
from ..transformer.SubLayers import MultiHeadAttention
import math

## transformer layer(res, self-attn, cross_attn, res)
class hybrid_transformer_v3_0(nn.Module):
    def __init__(
        self,
        init_dim: int  = 64,
        dim_list: list = [128, 256, 512, 1024, 512, 256, 128],
        channels: int = 1,
        resnet_block_groups: int = 8,
        learned_sin_dim: int = 12,
        num_classes: int = 10,
        image_size: tuple =(1, 8, 8),
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.dropout_rate = dropout
        self.dim = 512
        input_channels = output_channels = channels
        in_out_dim_list = list(zip(dim_list[:-1], dim_list[1:]))

        # time and class embeddings
        emb_dim = 256
        sin_pos_emb = LearnedSinusoidalPosEmb(learned_sin_dim)
        self.time_emb = nn.Sequential(
            sin_pos_emb,
            nn.Linear(learned_sin_dim + 1, emb_dim//4),
            nn.GELU(),
            nn.Linear(emb_dim // 4, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes+1, emb_dim)

        # blocks declaration
        GN_Pos_Resblock = partial(ResnetBlock, groups=resnet_block_groups, dropout=dropout)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding=1)
        self.init_Resblock = GN_Pos_Resblock(init_dim, dim_list[0], time_emb_dim=None)
        self.context_proj = LearnableContextPlus(in_dim=emb_dim, out_dim=512, num_tokens=8)
        # module_stacks
        self.module_stacks = nn.ModuleList([])
        for (dim_in, dim_out) in in_out_dim_list:
            self.module_stacks.append(
                nn.ModuleList([
                    GN_Pos_Resblock(dim_in, dim_in, time_emb_dim=emb_dim),
                    nn.LayerNorm(dim_in),
                    MultiHeadAttention(n_head=8, d_model=dim_in, d_k=64, d_v=64, dropout=dropout),
                    MultiHeadAttention(n_head=8, d_model=dim_in, d_k=64, d_v=64, dropout=dropout),
                    GN_Pos_Resblock(dim_in, dim_out, time_emb_dim=emb_dim),]
                )
            )

        self.final_Resblock = GN_Pos_Resblock(dim_list[-1], init_dim, time_emb_dim=None)
        self.final_conv = nn.Conv2d(init_dim, output_channels, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor = None, classes: torch.Tensor = None):
        time_emb = self.time_emb(time)
        class_emb = self.class_emb(classes)
        context = self.context_proj(time_emb + class_emb)
        x = self.init_conv(x)
        x = self.init_Resblock(x)
        for [block1, norm, self_attn, cross_attn, block2,] in self.module_stacks:
            # 1st group normalization position embedding res-block
            x = block1(x, time_emb + class_emb)
            x = rearrange(x, 'b c h w -> b (h w) c') #image->token vector like
            # self-attention
            x = norm(x)
            x, _= self_attn(q=x, k=x, v=x)

            # cross-attention
            x, _ = cross_attn(q=x, k=context, v=context)
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.image_size[1], w=self.image_size[2])
            # 2nd group normalization position embedding res-block
            x = block2(x, time_emb + class_emb)

        x = self.final_Resblock(x)
        x = self.final_conv(x)

        return x
