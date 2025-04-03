import time
import math
import torch
from torch import nn, Tensor
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features1, features2, labels):
        distances = nn.functional.pairwise_distance(features1, features2)
        loss = torch.mean((1 - labels) * torch.pow(distances, 2) +
                          (labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
        return loss


# Cosine Similarity Loss function
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, features1, features2):
        similarity = self.cos_sim(features1, features2)
        loss = torch.mean(1 - similarity)  # Cosine similarity loss
        return loss


class CCALoss(nn.Module):
    def __init__(self, output_dim=512, regularization=1e-4):
        super(CCALoss, self).__init__()
        self.output_dim = output_dim  # Dimensionality of the projection space
        self.regularization = regularization  # Regularization term to stabilize the matrix inversion

    def forward(self, H1, H2):
        """
        Compute the CCA loss between two sets of features (H1 and H2)

        H1: torch.Tensor of shape (batch_size, seq_len, feature_dim1)
        H2: torch.Tensor of shape (batch_size, seq_len, feature_dim2)

        Returns:
            CCA loss: Negative correlation between the projections of H1 and H2
        """
        # Option 1: Average pooling over the temporal dimension (seq_len)
        H1_pooled = H1.mean(dim=1)  # H1 becomes (batch_size, feature_dim1)
        H2_pooled = H2.mean(dim=1)  # H2 becomes (batch_size, feature_dim2)

        # Centering the data (remove the mean)
        H1_pooled = H1_pooled - H1_pooled.mean(dim=0, keepdim=True)
        H2_pooled = H2_pooled - H2_pooled.mean(dim=0, keepdim=True)

        # Compute the covariance matrices
        H1_H1_T = torch.mm(H1_pooled.T, H1_pooled) / (H1_pooled.size(0) - 1)
        H2_H2_T = torch.mm(H2_pooled.T, H2_pooled) / (H2_pooled.size(0) - 1)
        H1_H2_T = torch.mm(H1_pooled.T, H2_pooled) / (H1_pooled.size(0) - 1)

        # Regularization to prevent singular matrix (add small value to diagonal)
        r1 = self.regularization * torch.eye(H1_H1_T.size(0)).to(H1.device)
        r2 = self.regularization * torch.eye(H2_H2_T.size(0)).to(H2.device)

        # Covariance matrices with regularization
        H1_H1_T += r1
        H2_H2_T += r2

        # Perform SVD on the cross-covariance matrix (H1^T H2)
        U, S, V = torch.svd(torch.mm(torch.inverse(H1_H1_T), H1_H2_T).mm(torch.inverse(H2_H2_T)))

        # Canonical correlation is given by the singular values
        correlation = S.sum()

        # Negative correlation as the loss (maximize correlation => minimize negative correlation)
        loss = -correlation
        return loss


class MSQNetVariation1Contrastive(nn.Module):
    def __init__(self, class_embed, num_frames, recurrent: str | None = None, fusion: str = 'normal',
                 residual: bool = False, relu: bool = False, summary_residual: bool = False,
                 backbone_residual: bool = False, linear2_residual: bool = False, image_residual: bool = False,):
        super(MSQNetVariation1Contrastive, self).__init__()
        assert fusion in {'crossattention', 'transformer', 'selfattention', 'normal'}
        self.num_classes, self.embed_dim = class_embed.shape
        self.fusion = fusion
        self.relu = relu

        # BRANCH 1
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                         num_frames=num_frames, ignore_mismatched_sizes=True, local_files_only=True)
        self.linear1 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)

        self.recurrent = recurrent
        if self.recurrent == 'bilstm':
            self.bilstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim//2, num_layers=1, bidirectional=True)
        elif self.recurrent == 'gru':
            self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim//2, num_layers=1, bidirectional=True)
        elif self.recurrent == 'conv':
            #self.conv = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, stride=1, padding=1)
            self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        # RESIDUALS
        self.summary_residual = summary_residual

        self.backbone_residual = backbone_residual
        if self.backbone_residual:
            self.linear_backbone_residual = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=self.embed_dim)
            self.avg_pool_backbone_residual = nn.AdaptiveAvgPool1d(16)

        self.linear2_residual = linear2_residual

        self.image_residual = image_residual
        if self.image_residual:
            self.linear_image_residual = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=self.embed_dim)

        self.residual = residual
        if self.residual:
            self.linear_residual = nn.Linear(in_features=16, out_features=self.num_classes)

        # BRANCH 2
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)
        self.linear2 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim,
                                 out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(class_embed)
        # FUSION
        if self.fusion == 'crossattention':
            self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1, batch_first=True)
            self.linear_fusion = nn.Linear(in_features=16, out_features=self.num_classes)
        if self.fusion == 'selfattention':
            self.cross_attention_summary = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1, batch_first=True)
            self.cross_attention_video = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1, batch_first=True)
            self.linear_fusion = nn.Linear(in_features=self.embed_dim * 2, out_features=self.embed_dim)
        elif self.fusion == 'transformer':
            self.summary_attention = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
            self.video_attention = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
            self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1, batch_first=True)
        elif self.fusion == 'normal':
            self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)

        # ACTION RECOGNITION
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, images, summaries):
        b, t, c, h, w = images.size()
        #print('IMAGES: ', images.shape)
        x = self.backbone(images)[0]
        #print('BACKBONE: ', x.shape)
        if self.backbone_residual:
            backbone_res = self.linear_backbone_residual(x)
            backbone_res = self.avg_pool_backbone_residual(backbone_res.transpose(1, 2)).transpose(1, 2)
        #print('SUMMARIES: ', summaries.shape)
        summaries = summaries.unsqueeze(1)
        if self.recurrent == 'bilstm':
            summaries = self.bilstm(summaries)[0]
        elif self.recurrent == 'gru':
            summaries = self.gru(summaries)[0]
        elif self.recurrent == 'conv':
            summaries = self.conv(summaries)
        summaries = summaries.squeeze(1)

        x = self.linear1(
            torch.cat((
                F.adaptive_avg_pool1d(x.transpose(1, 2), t).transpose(1, 2),
                summaries.unsqueeze(1).expand(-1, 16, -1)
            ), dim=-1)
        )

        if self.backbone_residual:
            x += backbone_res

        if self.summary_residual:
            #print('SUMMARIES: ', summaries.unsqueeze(1).expand(-1, 16, -1).shape)
            x += summaries.unsqueeze(1).expand(-1, 16, -1)

        if self.relu:
            x = nn.ReLU()(x)
        x = self.pos_encod(x)
        if self.residual:
            copy = torch.clone(x)
            copy = self.linear_residual(copy.transpose(1, 2)).transpose(1, 2)

        video_features = self.image_model(images.reshape(b * t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)

        if self.image_residual:
            video_features_copy = self.linear_image_residual(video_features.repeat(1, self.num_classes, 1))

        query_embed = self.linear2(
            torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)),
                         2))
        if self.relu:
            query_embed = nn.ReLU()(query_embed)

        if self.image_residual:
            query_embed += video_features_copy

        if self.fusion == 'crossattention':
            hs, _ = self.cross_attention(x, query_embed, query_embed)
            hs = self.linear_fusion(hs.transpose(1, 2)).transpose(1, 2)
            #hs = nn.ReLU(hs)
        elif self.fusion == 'selfattention':
            query_embed, _ = self.cross_attention_summary(query_embed, query_embed, query_embed)
            x, _ = self.cross_attention_video(x, x, x)
            hs = self.linear_fusion(torch.cat((x, summaries), dim=-1))
        elif self.fusion == 'transformer':
            query_embed = self.summary_attention(query_embed)
            x = self.video_attention(x)
            hs, _ = self.cross_attention(x, query_embed, query_embed)
        else:   # normal
            hs = self.transformer(x, query_embed)  # b, t, d

        if self.residual:
            hs += copy

        if self.linear2_residual:
            hs += query_embed

        out = self.group_linear(hs)
        return out, copy, query_embed


class MSQNetVariation1ContrastiveExecutor(object):
    def __init__(self, train_loader, test_loader, criterion, eval_metric, class_list, test_every, distributed,
                 gpu_id, recurrent: None | str, fusion: str, residual: bool, relu: bool, summary_residual: bool,
                 backbone_residual: bool, linear2_residual: bool, image_residual: bool, contrastive: str) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        # CONTRASTIVE
        self.contrastive_label = False
        self.cca = False
        print(f'[INFO] Using {contrastive} loss')
        if contrastive == 'contrastive':
            self.contrastive_loss_fn = ContrastiveLoss(margin=1.0).to(gpu_id)
            self.contrastive_label = True
        elif contrastive == 'cosine':
            self.contrastive_loss_fn = CosineSimilarityLoss(margin=0.0).to(gpu_id)
        elif contrastive == 'cca':
            self.contrastive_loss_fn = CCALoss()
            self.cca = True
        else:
            print('[ERR] Contrastive loss function is not defined')

        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        logging.set_verbosity_error()
        class_embed = self._get_text_features(class_list)
        model = MSQNetVariation1Contrastive(class_embed, num_frames, recurrent, fusion, residual, relu,
                                 summary_residual, backbone_residual, linear2_residual, image_residual).to(gpu_id)
        if distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.image_model.parameters():
            p.requires_grad = False
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}]) # 00001
        #self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt

    def _get_text_features(self, cl_names):
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)
        act_prompt = self._get_prompt(cl_names)
        texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
        text_class = text_model(**texts).pooler_output.detach()
        return text_class

    def _train_batch(self, data, summaries, label):
        self.optimizer.zero_grad()
        output, feature1, feature2 = self.model(data, summaries)
        loss_this = self.criterion(output, label)
        if self.contrastive_label:
            contrastive_loss = self.contrastive_loss_fn(feature1, feature2, label)
        else:
            contrastive_loss = self.contrastive_loss_fn(feature1, feature2)
        if self.cca:
            loss = loss_this + 0.0001 * contrastive_loss
        else:
            loss = loss_this + 0.1 * contrastive_loss
        loss.backward()
        self.optimizer.step()
        return loss_this.item(), contrastive_loss.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        contrastive_list = list()
        start_time = time.time()
        for data, summaries, label in self.train_loader:
            data, summaries, label = (data.to(self.gpu_id, non_blocking=True),
                                      summaries.to(self.gpu_id, non_blocking=True),
                                      label.to(self.gpu_id, non_blocking=True))
            loss_this, contrastive_loss = self._train_batch(data, summaries, label)
            loss_meter.update(loss_this, data.shape[0])
            contrastive_list.append(contrastive_loss)
        elapsed_time = time.time() - start_time
        self.scheduler.step()
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg)
                  + " contrastive loss: " + "{:.4f}".format(np.mean(contrastive_list)),
                  flush=True)

    def train(self, start_epoch, end_epoch):
        print('FROZEN BACKBONE', flush=True)
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for name, param in self.model.backbone.named_parameters():
            print(name, param.requires_grad)
        for epoch in range(start_epoch, end_epoch // 2):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print('UNFROZEN BACKBONE', flush=True)
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        for name, param in self.model.backbone.named_parameters():
            print(name, param.requires_grad)
        for epoch in range(start_epoch, int(end_epoch * 2)):
        #for epoch in range(start_epoch, int(end_epoch * 1.5)):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)

    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, summaries, label in self.test_loader:
            data, summaries, label = (data.to(self.gpu_id),
                                      summaries.to(self.gpu_id),
                                      label.long().to(self.gpu_id))
            with torch.no_grad():
                output, _, _ = self.model(data, summaries)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def test_train(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, summaries, label in self.train_loader:
            data, summaries, label = (data.to(self.gpu_id),
                                      summaries.to(self.gpu_id),
                                      label.long().to(self.gpu_id))
            with torch.no_grad():
                output, _, _ = self.model(data, summaries)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def save(self, file_path="./checkpoint"):
        torch.save(self.model.state_dict(), file_path + '.pth')
        torch.save(self.optimizer.state_dict(), file_path + '_optimizer.pth')

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.optimizer.load_state_dict(torch.load(file_path + '_optimizer.pth'))


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out


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



