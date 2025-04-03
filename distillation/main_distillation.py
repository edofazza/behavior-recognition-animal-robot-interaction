import os
os.environ['HF_HOME'] = './.cache'

import torch
import random
import numpy as np
from models.distillation_loss import (MultilabelDistillationLoss, MultilabelDistillationLossWithTemperatureAnnealing,
                                      DistillationWithGaussianNoiseAndDynamicTemperature,
                                      MultilabelDistillationLossWithDynamicTemperature,
                                      DistillationWithGaussianNoiseAndTemperatureAnnealing,
                                      DistillationWithContrastiveNTXent,
                                      DistillationWithContrastiveNTXentAndTemperatureAnnealing,
                                      DistillationWithContrastiveNTXentAndDynamicTemperature)
from torchmetrics.classification import MultilabelAveragePrecision
from models.msqnet_variation1 import MSQNetVariation1
from models.msqnet_variation1contrastive import MSQNetVariation1Contrastive
from ensemble.ensemble import Ensemble
#from models.videomamba_ak import VideoMambaAK
from models.timesformer import TimeSformer
from models.distillation_execution import DistillationExecutor


def main(args):
    if(args.seed>=0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("[INFO] Setting SEED: " + str(args.seed))
    else:
        print("[INFO] Setting SEED: None")

    if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type:", str(device), flush=True)

    config = dict()
    config['path_dataset'] = '.'  # MODIFIED, substituted get_config
    dataset = 'AnimalKingdom'
    path_data = os.path.join(config['path_dataset'], dataset)
    print("[INFO] Dataset path:", path_data, flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    train_loader = manager.get_train_loader(train_transform)
    print("[INFO] Train size:", str(len(train_loader.dataset)), flush=True)

    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform)
    print("[INFO] Test size:", str(len(val_loader.dataset)), flush=True)

    # criterion or loss
    #criterion = MultilabelDistillationLoss(args.temperature, args.alpha)
    #criterion = MultilabelDistillationLossWithTemperatureAnnealing(args.temperature, args.alpha)
    criterion = MultilabelDistillationLossWithDynamicTemperature(args.temperature, args.alpha)
    #criterion = DistillationWithGaussianNoiseAndDynamicTemperature(args.temperature, args.alpha)
    #criterion = DistillationWithGaussianNoiseAndTemperatureAnnealing(args.temperature, args.alpha)
    #criterion = DistillationWithContrastiveNTXent(args.temperature, args.alpha)
    #criterion = DistillationWithContrastiveNTXentAndTemperatureAnnealing(args.temperature, args.alpha)
    #criterion = DistillationWithContrastiveNTXentAndDynamicTemperature()
    eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
    eval_metric_string = 'Multilabel Average Precision'

    # MODEL DEFINITIONS
    model1 = MSQNetVariation1(class_embed=torch.rand((140, 512)),  # BEST 1 200 epoches
                              num_frames=16,
                              recurrent='conv',
                              fusion='normal',
                              residual=True,
                              relu=False,
                              summary_residual=False,
                              backbone_residual=True,
                              linear2_residual=False,
                              image_residual=True).to(device)
    model1.load_state_dict(torch.load('models/variation1_conv_residual_backboneresidual_imageresidual_.pth'))

    model2 = MSQNetVariation1Contrastive(class_embed=torch.rand((140, 512)),  # BEST 3 cosine
                                         num_frames=16,
                                         recurrent='conv',
                                         fusion='normal',
                                         residual=True,
                                         relu=False,
                                         summary_residual=False,
                                         backbone_residual=True,
                                         linear2_residual=True,
                                         image_residual=False).to(device)
    model2.load_state_dict(torch.load('models/variation1_conv_residual_backboneresidual_linear2residual_.pth'))

    model3 = MSQNetVariation1(class_embed=torch.rand((140, 512)),  # ARTEMIS cosine
                              num_frames=16,
                              recurrent='bilstm',
                              fusion='normal',
                              residual=True,
                              relu=False,
                              summary_residual=True,
                              backbone_residual=True,
                              linear2_residual=True,
                              image_residual=True).to(device)
    model3.load_state_dict(
        torch.load('models/variation1_bilstm_residual_sumresidual_backboneresidual_linear2residual_imageresidual_.pth'))

    models = [model1, model2, model3]
    teacher = Ensemble(models=models, data_loader=None, device=device, num_labels=num_classes)
    #student = VideoMambaAK(16).to(device)
    student = TimeSformer(16).to(device)

    executor = DistillationExecutor(teacher, student, train_loader, val_loader, criterion, eval_metric,
                                    class_list, args.test_every, device)
    executor.train(args.epoch_start, args.epochs)
    eval = executor.test()

    executor.save(f'distillation/{args.name}')
    print("[INFO] " + eval_metric_string + ": {:.2f}".format(eval * 100), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default=300, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="animalkingdom", help="Dataset: animalkingdom")
    parser.add_argument("--model", default="convit", help="Model: convit, query2label")
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=8, type=int, help="Size of the mini-batch")
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of torchvision workers used to load data (default: 8)")
    parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--train", default=False, type=bool, help="train or test")

    parser.add_argument("--name", default='model', type=str, help="Name of model")
    parser.add_argument("--temperature", default=1.0, type=float, help="Distillation temperature")
    parser.add_argument("--alpha", default=0.5, type=float, help="Distillation alpha")

    args = parser.parse_args()

    main(args)
