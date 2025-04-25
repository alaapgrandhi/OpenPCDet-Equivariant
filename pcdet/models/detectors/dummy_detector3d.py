import torch.nn as nn
import torch


class DummyDetector(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        pass

    def forward(self, batch_dict):
        center = torch.mean(batch_dict['points'][:, 1:4], dim=0)
        std = torch.std(batch_dict['points'][:, 1:4], dim=0)
        angle = torch.tensor([0.0]).to(device=std.device)
        bbox = torch.concat([center, std, angle]).reshape(1, -1)
        pred_dicts = [{'pred_boxes': bbox, 'pred_scores': torch.tensor([1.0]).to(device=std.device)}]
        return pred_dicts, 1.0
