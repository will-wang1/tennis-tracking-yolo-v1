"""E2E-Spot's tennis model, cut down to what inference needs.

The original E2EModel (train_e2e.py in github.com/jhong93/spot) pulls in
the whole training stack; this keeps its network - RegNet-Y 800MF with
Gate-Shift Modules, then a bidirectional GRU - under the same attribute
names, so the published checkpoint (tennis_rny008gsm_gru_rgb) loads as is.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .shift import make_temporal_shift

CLASSES = ("far_court_bounce", "far_court_swing", "far_court_serve",
           "near_court_bounce", "near_court_swing", "near_court_serve")  # data/tennis/class.txt
CLIP_LEN = 100
MAX_GRU_HIDDEN_DIM = 768


class FCPrediction(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(batch_size, clip_len, -1)


class GRUPrediction(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(feat_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class E2ESpotTennis(nn.Module):
    """Frames (B, T, 3, H, W), ImageNet-normalised, T <= CLIP_LEN -> logits
    (B, T, 1 + len(CLASSES)); class 0 is background."""

    def __init__(self, clip_len: int = CLIP_LEN):
        super().__init__()
        features = timm.create_model("regnety_008", pretrained=False)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()
        make_temporal_shift(features, clip_len, is_gsm=True)
        self._require_clip_len = clip_len
        self._features = features
        self._feat_dim = feat_dim
        self._pred_fine = GRUPrediction(feat_dim, len(CLASSES) + 1, min(feat_dim, MAX_GRU_HIDDEN_DIM))

    def forward(self, x):
        batch_size, true_clip_len, channels, height, width = x.shape
        clip_len = true_clip_len
        if true_clip_len < self._require_clip_len:  # the shift modules need the full clip length
            x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
            clip_len = self._require_clip_len
        feat = self._features(x.reshape(-1, channels, height, width)).reshape(batch_size, clip_len, self._feat_dim)
        return self._pred_fine(feat[:, :true_clip_len])


def load(weights: str, device: str) -> E2ESpotTennis:
    model = E2ESpotTennis()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()
