import shutil
import os


class Config:
    info = 'test'
    batch_size = 32
    num_works = 16
    max_epoch = 30
    device = 'cuda'
    dataset_dir = 'Datasets/celeba'
    eval_checkpoint_epoch = 1
    lr = 1e-3
    min_lr = 1e-6
    momentum = 0.9
    betas = (0.9, 0.999)
    weight_decay = 0.05

    output_dir_root = os.path.join('outputs')
    output_dir = os.path.join(output_dir_root, info + '-' + 'lr{}-bs{}'.format(lr, batch_size))


def path_init(cfg):
    if not os.path.exists(os.path.join(cfg.output_dir_root)):
        os.makedirs(os.path.join(cfg.output_dir_root))
    if not os.path.exists(os.path.join(cfg.output_dir)):
        os.makedirs(os.path.join(cfg.output_dir))
    if not os.path.exists(os.path.join(cfg.output_dir, 'tf_logs')):
        os.makedirs(os.path.join(cfg.output_dir, 'tf_logs'))
    if os.path.exists(os.path.join(cfg.output_dir, 'code')):
        shutil.rmtree(os.path.join(cfg.output_dir, 'code'))


def cfg_merge_with_args(cfg, args):
    opt = vars(args)  # args NameSpace 转 dict
    if 'lr' in opt.keys():
        if args.lr is not None:
            cfg.lr = args.lr
    if 'output_dir' in opt.keys():
        if args.output_dir is not None:
            cfg.output_dir = args.output_dir
    return cfg
