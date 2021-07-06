import argparse
from utils import bool_flag

def parse_arguments():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
            'Instead, it is %s.' % v)
    
    parser = argparse.ArgumentParser(description="Implementation of SwAV")
    parser.register('type', 'bool', str2bool)

    #########################
    #### data parameters ####
    #########################
    parser.add_argument('--data_path', type=str, default='/path/to/kinetics',
                        help='path to dataset repository')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='number of frames to sample per clip')
    parser.add_argument('--target_fps', type=int, default=30,
                        help='video fps')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='rate to sample frames')
    parser.add_argument('--num_train_clips', type=int, default=1,
                        help='number of clips to sample per videos')
    parser.add_argument('--train_crop_size', type=int, default=112,
                        help="train crop size")
    parser.add_argument('--test_crop_size', type=int, default=112,
                        help="test crop size")
    parser.add_argument('--colorjitter', type='bool', default='True',
                        help='use color jitter')
    parser.add_argument('--use_grayscale', type='bool', default='True',
                        help='use grayscale augmentation')
    parser.add_argument('--use_gaussian', type='bool', default='True',
                        help='use gaussian augmentation')
    parser.add_argument('--num_sec_aud', type=int, default=1,
                        help='number of seconds of audio')
    parser.add_argument('--num_sec', type=int, default=1,
                        help='number of seconds for video, should equal num_frames/fps and num_sec_aud')
    parser.add_argument('--aud_sample_rate', type=int, default=24000,
                        help='audio sample rate')
    parser.add_argument('--audio_augtype', type=str, default='none',
                        choices=['none', 'mild', 'medium', 'heavy'], 
                        help='audio augmentation strength with Spec Augment')
    parser.add_argument('--aud_spec_type', type=int, default=2,
                        help="audio spec type")
    parser.add_argument('--use_volume_jittering', type='bool', default='False',
                        help='use volume jittering')
    parser.add_argument('--use_audio_temp_jittering', type='bool', default='False',
                        help='use audio temporal jittering')
    parser.add_argument('--z_normalize', type='bool', default='False',
                        help='z-normalize the audio')
    parser.add_argument('--dual_data', type='bool', default='False',
                        help='sample two clips per video')

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size per gpu, i.e. how many unique instances per gpu')
    parser.add_argument('--base_lr', default=4.8, type=float, 
                        help='base learning rate')
    parser.add_argument('--temp', default=0.1, type=float, 
                        help='within-modal NCE temp')
    parser.add_argument('--final_lr', type=float, default=0, 
                        help='final learning rate')
    parser.add_argument('--wd', default=1e-6, type=float, 
                        help='weight decay')
    parser.add_argument('--warmup_epochs', default=10, type=int, 
                        help='number of warmup epochs')
    parser.add_argument('--start_warmup', default=0, type=float,
                        help='initial warmup learning rate')
    parser.add_argument('--lr_milestones', default='20,30,40', type=str, 
                        help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float, 
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--use_lars', default='False', type='bool',
                        help="use LARS optimizer")
    parser.add_argument('--use_warmup_scheduler', default='True', type='bool',
                        help="use warmup scheduler")
    parser.add_argument('--use_lr_scheduler' , default='False', type='bool',
                        help='use cosine LR scheduler')
    parser.add_argument('--cross_modal_nce', type='bool', default='True',
                        help='use cross-modal NCE loss')
    parser.add_argument('--sample_in_dataset', type='bool', default='True',
                        help='sample speed in dataset')
    parser.add_argument('--load_from_memory', type='bool', default='True',
                        help='whether to load validation batch from memory')
    parser.add_argument('--multi_crop', type='bool', default='False',
                        help='do multi-crop comparisons')
    parser.add_argument('--use_random_resize_crop', type='bool', default='True',
                        help='use random resized crop instead of short stide jitter')
    parser.add_argument('--constant_scale', type='bool', default='False',
                        help='use constant scale for all random resized crops')
    parser.add_argument('--cross_modal_alpha', type=float, default=0.5,
                        help='weighting of cross-modal loss')
    parser.add_argument('--num_large_crops', type=int, default=1,
                        help='Number of Large Crops')
    parser.add_argument('--num_small_crops', type=int, default=0,
                        help='Number of small Crops')
    parser.add_argument('--num_large_tcrops', type=int, default=0,
                        help='Number of Large temporal Crops ')
    parser.add_argument('--num_small_tcrops', type=int, default=0,
                        help='Number of small temporal Crops')
    parser.add_argument('--only_cross_modal', type='bool', default='False',
                        help='do loss only crossmodally')
    parser.add_argument('--also_cross_modal', type='bool', default='False',
                        help='do loss ALSO crossmodally, use with multi-crop==True')

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument('--dist_url', default='env://', type=str, 
                        help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--world_size', default=-1, type=int, 
                        help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument('--rank', default=0, type=int, 
                        help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument('--local_rank', default=0, type=int,
                        help='this argument is not used and should be ignored')
    parser.add_argument('--bash', action='store_true', 
                        help='slrum bash mode')

    ############################
    #### transformer pooling ###
    ############################
    parser.add_argument('--num_layer', default=2, type=int, 
                        help='num of transformer layers')
    parser.add_argument('--dp', default=0.0, type=float, 
                        help='dropout rate in transformer')
    parser.add_argument('--num_head', default=4, type=int, 
                        help='num head in transformer')
    parser.add_argument('--positional_emb', type='bool', default='False', 
                        help='use positional emb in transformer')
    parser.add_argument('--qkv_mha', type='bool', default='False', 
                        help='complete qkv in MHA')
    parser.add_argument('--transformer_time_dim', default=4, type=int, 
                        help='temporal input for transformer')
    

    #########################
    #### model parameters ###
    #########################
    parser.add_argument('--vid_base_arch', default='r2plus1d_18', type=str, 
                        help='video architecture', 
                        choices=['r2plus1d_18', 'r2plus1d_34', 'r3d_50'])
    parser.add_argument('--aud_base_arch', default='resnet9', type=str, 
                        help="audio architecture", 
                        choices=['resnet9', 'resnet18'])
    parser.add_argument('--use_mlp', type='bool', default='True',
                        help='use MLP head')
    parser.add_argument('--mlp_dim', default=256, type=int,
                        help='final layer dimension in projection head')

    #########################
    #### other parameters ###
    #########################
    parser.add_argument('--workers', default=10, type=int,
                        help='number of data loading workers')
    parser.add_argument('--checkpoint_freq', type=int, default=20,
                        help='Save the model periodically')
    parser.add_argument('--use_fp16', type='bool', default='False',
                        help='whether to train with mixed precision or not')
    parser.add_argument('--sync_bn', type=str, default='pytorch', 
                        help='synchronize bn')
    parser.add_argument('--dump_path', type=str, default='.',
                        help='experiment dump path for checkpoints and log')
    parser.add_argument('--resume', type=str, default=".",
                        help='experiment dump path for checkpoints and log')
    parser.add_argument('--seed', type=int, default=31, 
                        help='seed')
    parser.add_argument('--test_only', type='bool', default='False', 
                        help='only test model')
    parser.add_argument('--eval_freq', type=int, default=25,
                        help='Save the model periodically')
    return parser
