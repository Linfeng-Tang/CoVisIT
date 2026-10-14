


import copy
import torch.nn.functional as F
import torch
import torch.nn as nn



from models.fdam_pre import FDAMPreAlign
from models.unet_utils import (
    BasicTransformerBlock, Downsample, ResBlock, SpatialTransformer,
    SpatialTransformerV2, Timestep, TimestepEmbedSequential,
    TransformerBlockV2, Upsample, conv_nd, linear, normalization,
    timestep_embedding, zero_module, GatedCrossAttention2d)


from models.fdam_ms_align import FullFDAMAlign
from models.cross_modal_refine import MultiScaleCrossModalEncoder

def exists(x):
    return x is not None

def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


class DiffusionUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.init_params(cfg)
        self.construct_network()

    def init_params(self, cfg):
        self.in_channels = cfg.IN_CHANNELS  
        self.model_channels = cfg.MODEL_CHANNELS  
        self.out_channels = cfg.OUT_CHANNELS 
        self.num_res_blocks = cfg.NUM_RES_BLOCKS 
        self.attention_resolutions = cfg.ATTENTION_RESOLUTIONS

        self.num_heads = cfg.get('NUM_HEADS', -1)
        self.num_head_channels = cfg.get('NUM_HEADS_CHANNELS', -1)
        self.dropout = cfg.get('DROPOUT', 0)
        self.channel_mult = tuple(cfg.get('CHANNEL_MULT', [1, 2, 4, 4])) 
        
        self.conv_resample = cfg.get('CONV_RESAMPLE', True) 
        self.dims = cfg.get('DIMS', 2)  

        self.use_checkpoint = cfg.get('USE_CHECKPOINT', False)
        self.use_scale_shift_norm = cfg.get('USE_SCALE_SHIFT_NORM', False)
        self.resblock_updown = cfg.get('RESBLOCK_UPDOWN', False) 
        self.use_new_attention_order = cfg.get('USE_NEW_ATTENTION_ORDER', True) 
        self.use_spatial_transformer = cfg.get('USE_SPATIAL_TRANSFORMER', True) 
        self.transformer_depth = cfg.get('TRANSFORMER_DEPTH', 1)
        self.use_linear_in_transformer = cfg.get('USE_LINEAR_IN_TRANSFORMER',
                                                 False)
        self.disable_self_attentions = cfg.get('DISABLE_SELF_ATTENTIONS', None)
        self.disable_middle_self_attn = cfg.get('DISABLE_MIDDLE_SELF_ATTN',
                                                False)
        self.adm_in_channels = cfg.get('ADM_IN_CHANNELS', None) 
        self.ignore_keys = cfg.get('IGNORE_KEYS', [])

        
        
        
        
        
        
        self.use_fdam_pre = cfg.get('USE_FDAM_PRE', False)
        self.fdam_feat_channels = cfg.get('FDAM_FEAT_CHANNELS', self.out_channels)
        self.fdam_kernel_sizes = tuple(cfg.get('FDAM_KERNEL_SIZES', [3, 5, 7]))
        self.fdam_num_fab = int(cfg.get('FDAM_NUM_FAB', 2))
        self.fdam_weight_softmax = cfg.get('FDAM_WEIGHT_SOFTMAX', True)
        self.fdam_use_groupnorm = cfg.get('FDAM_USE_GROUPNORM', True)
        self.fdam_dkg_hidden = cfg.get('FDAM_DKG_HIDDEN_CHANNELS', None)

        
        
        
        self.use_ms_refine_align = cfg.get('USE_MS_REFINE_ALIGN', False)
        self.ms_refine_num_fab = int(cfg.get('MS_REFINE_NUM_FAB', 1))
        self.ms_refine_weight_softmax = bool(cfg.get('MS_REFINE_WEIGHT_SOFTMAX', True))
        self.ms_refine_use_groupnorm = bool(cfg.get('MS_REFINE_USE_GROUPNORM', True))
        self.ms_refine_dkg_hidden = cfg.get('MS_REFINE_DKG_HIDDEN_CHANNELS', None)
        self.ms_refine_share_ms_encoder = bool(cfg.get('MS_REFINE_SHARE_MS_ENCODER', False))
        self.ms_refine_zero_init_affine = bool(cfg.get('MS_REFINE_ZERO_INIT_AFFINE', True))
        
        self.ms_refine_scales = (2, 4)
        self.ms_refine_kernel_sizes_map = {
            2: (3, 5, 7),
            4: (3, 5),
        }
        self.ms_refine_groups_map = {
            2: 7,
            4: 14,
        }

        assert (self.num_heads > 0 or self.num_head_channels > 0) and \
               (self.num_heads == -1 or self.num_head_channels == -1)

        if isinstance(self.num_res_blocks, int):
            self.num_res_blocks = len(
                self.channel_mult) * [self.num_res_blocks]
        elif len(self.num_res_blocks) != len(self.channel_mult):
            raise ValueError(
                'provide num_res_blocks either as an int (globally constant) or '
                'as a list/tuple (per-level) with the same length as channel_mult'
            ) 

    def construct_network(self):
        
        in_channels = self.in_channels
        model_channels = self.model_channels
        out_channels = self.out_channels
        attention_resolutions = self.attention_resolutions
        channel_mult = self.channel_mult

        num_heads = self.num_heads
        num_head_channels = self.num_head_channels
        dims = self.dims
        dropout = self.dropout
        use_checkpoint = self.use_checkpoint
        use_scale_shift_norm = self.use_scale_shift_norm
        disable_self_attentions = self.disable_self_attentions
        disable_middle_self_attn = self.disable_middle_self_attn
        transformer_depth = self.transformer_depth

        use_linear_in_transformer = self.use_linear_in_transformer
        resblock_updown = self.resblock_updown
        conv_resample = self.conv_resample
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        
        
        if self.use_fdam_pre:
            self.fdam_pre = FDAMPreAlign(
                feat_channels=self.fdam_feat_channels,
                kernel_sizes=self.fdam_kernel_sizes,
                num_fab=self.fdam_num_fab,
                dkg_hidden_channels=self.fdam_dkg_hidden,
                weight_softmax=self.fdam_weight_softmax,
                use_groupnorm=self.fdam_use_groupnorm,
            )
        else:
            self.fdam_pre = None


        
        
        
        
        self.guide_channels = in_channels // 3
        ch_ds1 = model_channels * channel_mult[0]  
        ch_ds2 = model_channels * channel_mult[1]  
        ch_ds4 = model_channels * channel_mult[2]  
        ch_ds8 = model_channels * channel_mult[3]  
        self.cross_modal_ms_encoder = MultiScaleCrossModalEncoder(
            in_channels=self.guide_channels,
            ch_ds1=ch_ds1,
            ch_ds2=ch_ds2,
            ch_ds4=ch_ds4,
            dims=dims,
            use_checkpoint=use_checkpoint,
            zero_init_affine=self.ms_refine_zero_init_affine,
        )

        
        if self.use_ms_refine_align:
            self.ms_refine_aligners = nn.ModuleDict()
            scale_to_ch = {2: ch_ds2, 4: ch_ds4, 8: ch_ds8}
            for s in self.ms_refine_scales:
                if s not in scale_to_ch:
                    continue
                self.ms_refine_aligners[str(s)] = FullFDAMAlign(
                    channels=scale_to_ch[s],
                    kernel_sizes=self.ms_refine_kernel_sizes_map[s],
                    num_fab=self.ms_refine_num_fab,
                    dkg_hidden_channels=self.ms_refine_dkg_hidden,
                    weight_softmax=self.ms_refine_weight_softmax,
                    use_groupnorm=self.ms_refine_use_groupnorm,
                    use_checkpoint=use_checkpoint,
                    num_groups=self.ms_refine_groups_map[s],
                )
        else:
            self.ms_refine_aligners = None
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))  
        ]) 
        self._feature_size = model_channels  
        input_block_chans = [model_channels] 
        input_down_flag = [False] 
        ch = model_channels 
        ds = 1  
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(  
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False  

                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer, 
                            use_checkpoint=use_checkpoint))
                    
                    layers.append(GatedCrossAttention2d(ch, num_heads, dim_head, ds=ds, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                input_down_flag.append(False)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch) 
                input_down_flag.append(True) 
                ds *= 2
                self._feature_size += ch
        self._input_block_chans = copy.deepcopy(input_block_chans) 
        self._input_down_flag = input_down_flag 

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch,
                               num_heads,
                               dim_head,
                               depth=transformer_depth,
                               disable_self_attn=disable_middle_self_attn,
                               use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self._middle_block_chans = [ch]

        self._output_block_chans = []
        self.output_blocks = nn.ModuleList([])
        self.lsc_identity = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]: 
            for i in range(self.num_res_blocks[level] + 1): 
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                    
                    
                    layers.append(
                        GatedCrossAttention2d(
                            ch, num_heads, dim_head,
                            ds=ds,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.lsc_identity.append(nn.Identity())
                self._feature_size += ch
                self._output_block_chans.append(ch)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )



    def _forward_origin(self, x, emb,  context=None, cross_context_map=None, **kwargs):
        hs = []
        h = x

        

        
        for module in self.input_blocks:
            h = module(h, emb, context, cross_context_map=cross_context_map)
            hs.append(h)
        
        
        h = self.middle_block(h, emb, context, cross_context_map=cross_context_map)

        
        for m_id, module in enumerate(self.output_blocks):
            skip_h = hs.pop()
            h = torch.cat([h, self.lsc_identity[m_id](skip_h)], dim=1)
            
            target_size = hs[-1].shape[-2:] if len(hs) > 0 else None
            h = module(h, emb, context, target_size, cross_context_map=cross_context_map) 
        out = self.out(h)
        return out
    
    def forward(self, x, t=None, cond=None, guide=None, **kwargs): 
        
        
        
        

        
        
        
        if self.fdam_pre is not None:
            guide = self.fdam_pre(rgb_feat=cond, ir_feat=guide)


        
        
        
        def _to_tokens(f):
            b, c, h, w = f.shape
            return f.view(b, c, h*w).permute(0, 2, 1).contiguous()

        rgb_feats, guide_feats = self.cross_modal_ms_encoder(cond, guide)

        
        
        if self.use_ms_refine_align and (self.ms_refine_aligners is not None):
            for s in self.ms_refine_scales:
                k = str(s)
                if k not in self.ms_refine_aligners:
                    continue
                if s not in guide_feats or s not in rgb_feats:
                    continue
                guide_feats[s] = self.ms_refine_aligners[k](rgb_feats[s], guide_feats[s])

        cross_context_map = {
            2: _to_tokens(guide_feats[2]),
            4: _to_tokens(guide_feats[4]),
        }

        
        x_in = torch.cat([guide, cond, x], dim=1)

        
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        
        out= self._forward_origin(x_in, emb, cross_context_map=cross_context_map, **kwargs)
        return out

