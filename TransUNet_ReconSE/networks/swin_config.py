import ml_collections

def get_swin_config():
    config = ml_collections.ConfigDict()

    # Model Basics
    config.img_size = 320
    config.patch_size = 4
    config.in_chans = 3
    config.embed_dim = 96
    config.depths = [2, 2, 6, 2]
    config.num_heads = [3, 6, 12, 24]
    config.window_size = 5
    config.mlp_ratio = 4.0
    config.qkv_bias = True
    config.qk_scale = None

    # Regularization
    config.drop_rate = 0.0
    config.attn_drop_rate = 0.0
    config.drop_path_rate = 0.1
    config.norm_layer = "layernorm"
    config.ape = False
    config.patch_norm = True
    config.use_checkpoint = False

    # Decoder
    config.decoder_channels = (256, 128, 64, 16)
    #config.skip_channels = [192, 384, 768, 1536]  # Swin encoder outputs (doubles each stage)
    config.n_skip = 0
    config.n_classes = 1  # for MRI reconstruction (1-channel output)

    return config
