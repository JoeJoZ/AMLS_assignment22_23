import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import Config, path_init
from inference import do_evaluation
from utils.logger import setup_logger
from utils.checkpoint import CheckPointer
from utils.gpu import select_device
from utils.global_var import GlobalVar
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader
from A1.FaceDataset import FaceDataset
from A1.model import modelA1


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def main():
    # init config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed()
    cfg = Config
    path_init(cfg=cfg)
    GlobalVar().set_var('cfg', cfg)
    logger = setup_logger(cfg.info, cfg.output_dir)
    logger.info('========> this program info')
    GlobalVar().set_var('logger', logger)
    summary_writer = SummaryWriter(
        log_dir=os.path.join(cfg.output_dir, 'tf_logs'))
    if cfg.device == 'cuda':
        device = select_device(id=0, force_cpu=False)
    else:
        device = select_device(id=0, force_cpu=True)
    GlobalVar().set_var('device', device)

    # build dateset
    dataset = FaceDataset(cfg, split='train')
    data_loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_works, shuffle=True)

    # model
    model = modelA1()
    model.to(device)
    logger.info(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    n_iter_per_epoch = len(data_loader.dataset) // cfg.batch_size
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=int(int(cfg.max_epoch) * n_iter_per_epoch),
        lr_min=cfg.min_lr,
        warmup_lr_init=cfg.min_lr,
        warmup_prefix=True,
        warmup_t=0,
        cycle_limit=1,
        t_in_epochs=False,
    )

    # checkpoint
    arguments = {"epoch": 0, "batch_size": cfg.batch_size,
                 "global_step": 0}  # 超参初始化

    # load finetune model
    check_pointer = CheckPointer(
        model, optimizer, scheduler, cfg.output_dir, logger)  
    extra_checkpoint_data = check_pointer.load()  
    if extra_checkpoint_data: 
        arguments.update(extra_checkpoint_data) 

    # loss function
    loss_evaluator = nn.CrossEntropyLoss()
    
    # train
    model.train()
    global_step = arguments["global_step"]
    global_epoch = arguments["epoch"]
    for epoch in range(global_epoch, cfg.max_epoch):
        global_epoch = epoch
        for iteration, sample_batched in enumerate(data_loader):
            images, labels = sample_batched['image'], sample_batched['label']
            images = images.to(device)
            labels = labels.long().to(device)

            predicts_batched = model(images)

            loss = loss_evaluator(predicts_batched, labels)

            loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step_update(global_step)
            global_step += 1

            # save log
            if iteration % 10 == 0:
                now_lr = optimizer.param_groups[0]['lr']
                logger.info('epoch:%d/%d\titr:%d/%d\tloss:%.6g\tlr:%.6g' %
                            (epoch, cfg.max_epoch, iteration,
                             len(data_loader.dataset) // cfg.batch_size, loss,
                             now_lr))
                summary_writer.add_scalar(
                    'loss', loss, global_step=global_step)
                summary_writer.add_scalar(
                    'lr', now_lr, global_step=global_step)

        # val
        if epoch > 0 and epoch % cfg.eval_checkpoint_epoch == 0 and not epoch == cfg.max_epoch - 1:
            result_dict = do_evaluation(model, internal_split='train')
            summary_writer.add_scalars(
                'acc', {'Train': result_dict["acc"]}, global_step=global_step)
            summary_writer.add_scalars(
                'auc', {'Train': result_dict["auc"]}, global_step=global_step)

            result_dict = do_evaluation(model, internal_split='test')
            summary_writer.add_scalars(
                'acc', {'Test': result_dict["acc"]}, global_step=global_step)
            summary_writer.add_scalars(
                'auc', {'Test': result_dict["auc"]}, global_step=global_step)
            model.train()
            arguments["epoch"] = epoch + 1
            arguments["global_step"] = global_step + 1
            check_pointer.save(
                "model_{:03d}-{:06d}-Acc{:.4f}".format(epoch, global_step,
                                                       result_dict["acc"]),
                **arguments)

    # save final model
    logger.info('Train finish. Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    result_dict = do_evaluation(model, internal_split='train')
    summary_writer.add_scalars(
        'acc', {'Train': result_dict["acc"]}, global_step=global_step)
    summary_writer.add_scalars(
        'auc', {'Train': result_dict["auc"]}, global_step=global_step)

    result_dict = do_evaluation(model, internal_split='test')
    summary_writer.add_scalars(
        'acc', {'Test': result_dict["acc"]}, global_step=global_step)
    summary_writer.add_scalars(
        'auc', {'Test': result_dict["auc"]}, global_step=global_step)
    arguments["epoch"] = global_epoch + 1
    arguments["global_step"] = global_step + 1
    check_pointer.save(
        "model_final_{:03d}-{:06d}-Acc{:.4f}".format(global_epoch, global_step,
                                                     result_dict["acc"]),
        **arguments)


if __name__ == '__main__':
    main()
    GlobalVar().get_var('logger').info("========> all finished!")
