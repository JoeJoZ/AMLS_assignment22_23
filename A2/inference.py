import os
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from A2.config import Config, path_init, cfg_merge_with_args
import argparse
from utils.logger import setup_logger
from utils.checkpoint import CheckPointer
from utils.global_var import GlobalVar
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from A2.FaceDataset import FaceDataset
from A2.model import modelA2


@torch.no_grad()
def do_evaluation(internal_model, internal_split):
    internal_model.eval()
    logger = GlobalVar().get_var("logger")
    cfg = GlobalVar().get_var("cfg")
    internal_device = GlobalVar().get_var("device")
    dataset = FaceDataset(cfg, split=internal_split)
    data_loader_val = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_works, shuffle=True)
    logger.info('===> ' + 'Evaluating {} dataset: n{}'.format(internal_split,
                                                              len(data_loader_val.dataset)))
    total_results = np.empty(shape=[0])
    total_probs = np.empty(shape=[0])
    total_labels = np.empty(shape=[0])
    
    # test
    for sample_batched in tqdm(data_loader_val):  
        images, labels = sample_batched['image'], sample_batched['label']
        with torch.no_grad():
            outputs = internal_model(images.to(internal_device))
        results = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[:, 1]
        labels = labels.numpy().astype(np.uint8)

        total_results = np.concatenate((total_results, results), axis=0)
        total_probs = np.concatenate((total_probs, probs), axis=0)
        total_labels = np.concatenate((total_labels, labels), axis=0)
    
    # Calculate the performance
    correct_test = (total_results == total_labels).sum().item()
    acc = float(correct_test) / float(len(total_labels))
    auc = roc_auc_score(y_true=total_labels, y_score=total_probs)
    eval_result_dict = {'acc': acc, 'auc': auc}

    logger.info("acc: {:.4f}\tauc: {:.4f}".format(
        eval_result_dict["acc"], eval_result_dict["auc"]))

    return eval_result_dict


if __name__ == "__main__":
    # init config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("inference main eval...")

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--output_dir", type=str,
                        default='outputs/12-23-A2-atten1-lr0.001-bs32',
                        help='Specify a image dir to save predicted images.')

    args = parser.parse_args()
    cfg = Config
    cfg = cfg_merge_with_args(cfg, args)
    path_init(cfg)
    GlobalVar().set_var('cfg', cfg)
    logger = setup_logger(cfg.info, cfg.output_dir)
    logger.info('========> inference main eval...\n========> this program info')
    GlobalVar().set_var('logger', logger)
    device = torch.device(cfg.device)
    GlobalVar().set_var('device', device)
    model = modelA2()
    if torch.cuda.is_available():
        model.to(device)
        logger.info('use gpu')
    else:
        logger.info('use cpu')

    check_pointer = CheckPointer(model, None, None, cfg.output_dir, logger)
    check_pointer.load()
    acc, auc = do_evaluation(model, internal_split='test')
