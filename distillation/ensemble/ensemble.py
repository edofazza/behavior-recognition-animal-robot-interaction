import torch
from torch import nn
from utils.utils import AverageMeter
from torchmetrics.classification import MultilabelAveragePrecision


class Ensemble(object):
    def __init__(self, models, data_loader, device, num_labels):
        super(Ensemble, self).__init__()
        self.models = models
        self.data_loader = data_loader
        self.eval_metrics = MultilabelAveragePrecision(num_labels=num_labels, average='micro')
        self.device = device

    def ensemble(self, data, summaries, weights: list):
        assert len(weights) == len(self.models), 'Number of weights must equal number of models'
        for model in self.models:
            model.eval()
        weights = [w / sum(weights) for w in weights]
        with torch.no_grad():
            tmp_results = self.models[0](data, summaries)
            if type(tmp_results) is tuple:
                tmp_results = tmp_results[0]
            weighted_predictions = tmp_results * weights[0]
            for i in range(1, len(self.models)):
                tmp_results = self.models[i](data, summaries)
                if type(tmp_results) is tuple:
                    tmp_results = tmp_results[0]
                weighted_predictions += tmp_results * weights[i]
        return weighted_predictions

    def test(self, weights: list):
        for model in self.models:
            model.eval()
        eval_meter = AverageMeter()
        for data, summaries, label in self.data_loader:
            data, summaries, label = (data.to(self.device),
                                      summaries.to(self.device),
                                      label.long().to(self.device))
            #with torch.no_grad():
            output = self.ensemble(data, summaries, weights)

            eval_this = self.eval_metrics(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

