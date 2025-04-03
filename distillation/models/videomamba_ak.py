import time
import math
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import logging

from .videomamba import videomamba_middle, videomamba_small, videomamba_tiny


class VideoMambaAK(nn.Module):
    def __init__(self, num_frames):
        super(VideoMambaAK, self).__init__()
        self.num_classes = 140
        # BRANCH 1
        self.backbone = videomamba_middle(num_frames=num_frames, pretrained=True)     # M 576 S 384 T 192

        # ACTION RECOGNITION
        self.group_linear = GroupWiseLinear(self.num_classes, 576, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone.forward_features(images.reshape(b, c, t, h, w))
        out = self.group_linear(F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_classes).transpose(1, 2))
        return out


class VideoMambaAKExecutor(object):
    def __init__(self, train_loader, test_loader, criterion, eval_metric, class_list, test_every, distributed,
                 gpu_id) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        logging.set_verbosity_error()
        model = VideoMambaAK(num_frames).to(gpu_id)
        if distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}])
        #self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt

    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss_this = self.criterion(output, label)
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in self.train_loader:
            data, label = (data.to(self.gpu_id, non_blocking=True), label.to(self.gpu_id, non_blocking=True))
            loss_this = self._train_batch(data, label)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        self.scheduler.step()
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg), flush=True)

    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)

    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = (data.to(self.gpu_id), label.long().to(self.gpu_id))
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def test_train(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.train_loader:
            data, label = (data.to(self.gpu_id), label.long().to(self.gpu_id))
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def save(self, file_path="./checkpoint"):
        torch.save(self.model.state_dict(), file_path + '.pth')
        #torch.save(self.optimizer.state_dict(), file_path + '_optimizer.pth')

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        #self.optimizer.load_state_dict(torch.load(file_path + '_optimizer.pth'))



class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


