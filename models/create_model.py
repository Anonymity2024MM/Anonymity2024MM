from .vit import ViT
from .swin import SwinTransformer
from .cait import CaiT
from .t2t import T2TViT
from .mbvit import MobileViT
from .cct import CCT
from .cvt import CvT
from .deepvit import DeepViT
from .nystromformer import NystromFormer
from .rvt import RvT
from .regionvit import RegionViT
from .crossvit import CrossViT
from .xcit import XCiT
from .resnet import ResNet20, ResNet56, ResNet110

from .ft import Flash
from .ft_ablation import Flash_Ablation

def create_model(img_size, n_classes, args):
    if img_size == 32:
        patch_size = 4
    elif img_size == 64:
        patch_size = 8
    elif img_size == 256:
        patch_size = 32

    if args.model == 'vit':
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd)

    
    elif args.model == 'ft-ablation':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=False, is_vt=True, is_pe=False, is_SA=False, is_ss=True, is_sdm=True)


    elif args.model == 'ft-ablation-cls-pe-sa':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=True, is_vt=False, is_pe=True, is_SA=True, is_ss=False, is_sdm=False)


    elif args.model == 'ft-ablation-vt-pe-sa':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=False, is_vt=True, is_pe=True, is_SA=True, is_ss=False, is_sdm=False)

    elif args.model == 'ft-ablation-cls-ss-sdm':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=True, is_vt=False, is_pe=False, is_SA=False, is_ss=True, is_sdm=True)

    elif args.model == 'ft-ablation-vt-ss-sdm':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=False, is_vt=True, is_pe=False, is_SA=False, is_ss=True, is_sdm=True)

    elif args.model == 'ft-ablation-vt-ss-pe':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=False, is_vt=True, is_pe=True, is_SA=False, is_ss=True, is_sdm=False)

    elif args.model == 'ft-ablation-vt-sdm':
        model = Flash_Ablation(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd,
                    is_cls=False, is_vt=True, is_pe=True, is_SA=True, is_ss=False, is_sdm=False)


    elif args.model == 'vit-05m':
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=4, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd)
    
    elif args.model == 'vit-1m':
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=136, 
                    mlp_dim_ratio=2, depth=7, heads=8, dim_head=136//8,
                    stochastic_depth=args.sd)
        
    elif args.model == 'vit-2m':
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=200, 
                    mlp_dim_ratio=2, depth=6, heads=8, dim_head=200//8,
                    stochastic_depth=args.sd)
        
    elif args.model == 'vit-4m':
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=180, 
                    mlp_dim_ratio=2, depth=14, heads=12, dim_head=180//12,
                    stochastic_depth=args.sd)
        
        
    elif args.model == 'vit-8m':
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=232, 
                    mlp_dim_ratio=3, depth=14, heads=12, dim_head=232//12,
                    stochastic_depth=args.sd)

    elif args.model == 'ft-b':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd)
        

    elif args.model == 'ft-sota':
        patch_size = 4 if img_size == 32 else 8
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=80, 
                    mlp_dim_ratio=1, depth=8, heads=4, dim_head=80//4,
                    stochastic_depth=args.sd)
        
    elif args.model == 'ft-05m':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=120, 
                    mlp_dim_ratio=2, depth=7, heads=4, dim_head=120//4,
                    stochastic_depth=args.sd)
        
        
    elif args.model == 'ft-1m':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=144, 
                    mlp_dim_ratio=2, depth=9, heads=8, dim_head=144//8,
                    stochastic_depth=args.sd)

    elif args.model == 'ft-2m':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=204, 
                    mlp_dim_ratio=2, depth=10, heads=12, dim_head=204//12,
                    stochastic_depth=args.sd)

        
    elif args.model == 'ft-4m':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=264, 
                    mlp_dim_ratio=2, depth=12, heads=12, dim_head=264//12,
                    stochastic_depth=args.sd)
        
    elif args.model == 'ft-8m':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=360, 
                    mlp_dim_ratio=2, depth=12, heads=12, dim_head=360//12,
                    stochastic_depth=args.sd)
    
    elif args.model == 'ft-l':
        model = Flash(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=306, 
                    mlp_dim_ratio=2, depth=14, heads=6, dim_head=306//6,
                    stochastic_depth=args.sd)



    elif args.model == 'cct-05m':
        model = CCT(
            img_size = (img_size, img_size),
            num_classes=n_classes,
            embedding_dim = 80,
            n_conv_layers = 1,
            kernel_size = 7,
            stride = patch_size,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 8,
            num_heads = 4,
            mlp_ratio = 2.,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        )
    
    elif args.model == 'cct-1m':
        model = CCT(
            img_size = (img_size, img_size),
            num_classes=n_classes,
            embedding_dim = 160,
            n_conv_layers = 1,
            kernel_size = 7,
            stride = patch_size,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 5,
            num_heads = 4,
            mlp_ratio = 2.,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        )

    elif args.model == 'cct-2m':
        model = CCT(
            img_size = (img_size, img_size),
            num_classes=n_classes,
            embedding_dim = 200,
            n_conv_layers = 1,
            kernel_size = 7,
            stride = patch_size,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 6,
            num_heads = 4,
            mlp_ratio = 2.,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        )

    elif args.model == 'cct-4m':
        model = CCT(
            img_size = (img_size, img_size),
            num_classes=n_classes,
            embedding_dim = 204,
            n_conv_layers = 1,
            kernel_size = 7,
            stride = patch_size,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 12,
            num_heads = 12,
            mlp_ratio = 2.,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        )

    elif args.model == 'cct-8m':
        model = CCT(
            img_size = (img_size, img_size),
            num_classes=n_classes,
            embedding_dim = 400,
            n_conv_layers = 1,
            kernel_size = 7,
            stride = patch_size,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 6,
            num_heads = 4,
            mlp_ratio = 2.,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        )
        

    elif args.model == 'mbv2-05m':
        dims = [18, 36, 50]
        channels = [8, 16, 24, 24, 32, 32, 56, 56, 64, 64, 384]
        model = MobileViT((img_size, img_size), dims, channels, patch_size=(1, 1), num_classes=n_classes)

    elif args.model == 'mbv2-1m':
        dims = [54, 60, 72]
        channels = [10, 20, 36, 36, 54, 54, 72, 72, 94, 94, 384]
        model = MobileViT((img_size, img_size), dims, channels, patch_size=(1, 1), num_classes=n_classes)

    elif args.model == 'mbv2-2m':
        dims = [72, 120, 134]
        channels = [18, 36, 54, 54, 72, 72, 72, 72, 108, 108, 384]
        model = MobileViT((img_size, img_size), dims, channels, patch_size=(1, 1), num_classes=n_classes)

    elif args.model == 'mbv2-4m':
        dims = [144, 180, 224]
        channels = [24, 48, 54, 54, 72, 72, 108, 108, 120, 120, 384]
        model = MobileViT((img_size, img_size), dims, channels, patch_size=(1, 1), num_classes=n_classes)

    elif args.model == 'mbv2-8m':
        dims = [224, 302, 324]
        channels = [48, 53, 72, 72, 80, 80, 120, 120, 156, 156, 384]
        model = MobileViT((img_size, img_size), dims, channels, patch_size=(1, 1), num_classes=n_classes)

        
    elif args.model =='swin-s':
        depths = [1, 3, 2]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        embed_dim=84
        patch_size = 2 if img_size == 32 else 4
            
        model = SwinTransformer(img_size=img_size, embed_dim=embed_dim, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                )
        
    elif args.model =='swin-t':
        depths = [1, 2, 1]
        num_heads = [2, 4, 8]
        mlp_ratio = 2
        window_size = 4
        embed_dim=64
        patch_size = 2 if img_size == 32 else 4
            
        model = SwinTransformer(img_size=img_size, embed_dim=embed_dim, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                )
    
        
    elif args.model == 'cait-t':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(image_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=72, depth=16, cls_depth=1, mlp_dim=144, heads=4, dim_head=32, 
                     )

    elif args.model == 'cait-s':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(image_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=96, depth=18, cls_depth=2, mlp_dim=192, heads=4, dim_head=64, 
                     )
    
    elif args.model =='t2t-t':
        model = T2TViT(image_size=img_size, num_classes=n_classes, dim=128, depth = 10, heads = 4, mlp_dim = 128, pool = 'cls', channels = 3, dim_head = 32, dropout = 0., emb_dropout = 0., t2t_layers = ((11, 8),)
                        )
        
    elif args.model =='t2t-s':
        model = T2TViT(image_size=img_size, num_classes=n_classes, dim=192, depth = 10, heads = 4, mlp_dim = 192, pool = 'cls', channels = 3, dim_head = 48, dropout = 0., emb_dropout = 0., t2t_layers = ((11, 8),)
                        )

    elif args.model =='t2t-b':
        model = T2TViT(image_size=img_size, num_classes=n_classes, dim=256, depth = 8, heads = 4, mlp_dim = 256, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., t2t_layers = ((7, 4), (3, 2), (2, 1))
                        )
        

    elif args.model =='cvt':
        model = CvT(num_classes=n_classes,
                    s1_emb_dim = 32,
                    s1_emb_kernel = 7,
                    s1_emb_stride = 4,
                    s1_proj_kernel = 3,
                    s1_kv_proj_stride = 2,
                    s1_heads = 1,
                    s1_depth = 1,
                    s1_mlp_mult = 4,
                    s2_emb_dim = 96,
                    s2_emb_kernel = 3,
                    s2_emb_stride = 2,
                    s2_proj_kernel = 3,
                    s2_kv_proj_stride = 2,
                    s2_heads = 3,
                    s2_depth = 2,
                    s2_mlp_mult = 4,
                    s3_emb_dim = 192,
                    s3_emb_kernel = 3,
                    s3_emb_stride = 2,
                    s3_proj_kernel = 3,
                    s3_kv_proj_stride = 2,
                    s3_heads = 6,
                    s3_depth = 10,
                    s3_mlp_mult = 4,
                    dropout = 0.,
                    channels = 3
                )

    elif args.model =='deepvit-t':
        model = DeepViT(image_size=img_size, 
                        patch_size=patch_size, 
                        num_classes=n_classes, 
                        dim=80, 
                        depth=10, 
                        heads=4, 
                        mlp_dim=80, 
                        pool = 'cls', 
                        channels = 3, 
                        dim_head = 64, 
                        dropout = 0., 
                        emb_dropout = 0.
                        )
    
    elif args.model =='deepvit-s':
        model = DeepViT(image_size=img_size, 
                        patch_size=patch_size, 
                        num_classes=n_classes, 
                        dim=136, 
                        depth=14, 
                        heads=4, 
                        mlp_dim=136, 
                        pool = 'cls', 
                        channels = 3, 
                        dim_head = 64, 
                        dropout = 0., 
                        emb_dropout = 0.
                        )


    elif args.model =='nystromformer':
        model = NystromFormer(image_size=img_size, 
                            patch_size=patch_size, 
                            num_classes=n_classes, 
                            dim=160, 
                            depth=4, 
                            heads=8, 
                            num_landmarks=16
                            )
        
    elif args.model =='nystromformer-05m':
        model = NystromFormer(image_size=img_size, 
                            patch_size=patch_size, 
                            num_classes=n_classes, 
                            dim=48, 
                            depth=4, 
                            heads=8, 
                            num_landmarks=16
                            )

    elif args.model =='nystromformer-8m':
        model = NystromFormer(image_size=img_size, 
                            patch_size=patch_size, 
                            num_classes=n_classes, 
                            dim=400, 
                            depth=4, 
                            heads=8, 
                            num_landmarks=16
                            )
    
    elif args.model =='rvt-t':
        model = RvT(image_size=img_size, 
                    patch_size=patch_size, 
                    num_classes=n_classes, 
                    dim=64, 
                    depth=6, 
                    heads=8, 
                    mlp_dim=64
                    )
        
    elif args.model =='rvt-s':
        model = RvT(image_size=img_size, 
                    patch_size=patch_size, 
                    num_classes=n_classes, 
                    dim=128, 
                    depth=7, 
                    heads=8, 
                    mlp_dim=128
                    )

    
    elif args.model =='regionvit-t':
        model = RegionViT(
                dim = (48, 72, 108, 112),
                depth = (1, 1, 2, 1),
                window_size = img_size//8,
                num_classes = n_classes,
                tokenize_local_3_conv = False,
                local_patch_size = patch_size//2,
                use_peg = False,
                attn_dropout = 0.,
                ff_dropout = 0.,
                channels = 3,
            )
        

    elif args.model =='regionvit-s':
        model = RegionViT(
                dim = (48, 96, 128, 256),
                depth = (1, 2, 3, 2),
                window_size = img_size//8,
                num_classes = n_classes,
                tokenize_local_3_conv = False,
                local_patch_size = patch_size//2,
                use_peg = False,
                attn_dropout = 0.,
                ff_dropout = 0.,
                channels = 3,
            )
        
    elif args.model =='regionvit-b':
        model = RegionViT(
                dim = (64, 128, 256, 512),
                depth = (2, 2, 8, 2),
                window_size = img_size//8,
                num_classes = n_classes,
                tokenize_local_3_conv = False,
                local_patch_size = patch_size//2,
                use_peg = False,
                attn_dropout = 0.,
                ff_dropout = 0.,
                channels = 3,
            )
        
    elif args.model =='crossvit-t':
        model = CrossViT(
            image_size=img_size,
            num_classes=n_classes,
            sm_dim=102,
            lg_dim=102,
            sm_patch_size = patch_size,
            sm_enc_depth = 1,
            sm_enc_heads = 8,
            sm_enc_mlp_dim = 102,
            sm_enc_dim_head = 16,
            lg_patch_size = patch_size,
            lg_enc_depth = 4,
            lg_enc_heads = 8,
            lg_enc_mlp_dim = 102,
            lg_enc_dim_head = 16,
            cross_attn_depth = 1,
            cross_attn_heads = 8,
            cross_attn_dim_head = 16,
            depth = 2,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    elif args.model =='crossvit-s':
        model = CrossViT(
            image_size=img_size,
            num_classes=n_classes,
            sm_dim=128,
            lg_dim=128,
            sm_patch_size = patch_size,
            sm_enc_depth = 1,
            sm_enc_heads = 8,
            sm_enc_mlp_dim = 128,
            sm_enc_dim_head = 16,
            lg_patch_size = patch_size,
            lg_enc_depth = 4,
            lg_enc_heads = 8,
            lg_enc_mlp_dim = 128,
            lg_enc_dim_head = 16,
            cross_attn_depth = 2,
            cross_attn_heads = 8,
            cross_attn_dim_head = 16,
            depth = 3,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    elif args.model =='xcit-t':
        model = XCiT(
        image_size=img_size,
        patch_size=patch_size,
        num_classes=n_classes,
        dim=120,
        depth=8,
        cls_depth=2,
        heads=4,
        mlp_dim=120,
        dim_head = 30,
        dropout = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
        )

    elif args.model =='xcit-s':
        model = XCiT(
        image_size=img_size,
        patch_size=patch_size,
        num_classes=n_classes,
        dim=192,
        depth=10,
        cls_depth=2,
        heads=4,
        mlp_dim=192,
        dim_head = 48,
        dropout = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
        )


    elif args.model == 'resnet20':
        model = ResNet20(img_size=img_size, num_classes=n_classes)

    elif args.model == 'resnet56':
        model = ResNet56(img_size=img_size, num_classes=n_classes)

    elif args.model == 'resnet110':
        model = ResNet110(img_size=img_size, num_classes=n_classes)
    
    return model