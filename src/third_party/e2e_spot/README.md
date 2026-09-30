E2E-Spot (Hong et al., "Spotting Temporally Precise, Fine-Grained Events in
Video", ECCV 2022), vendored from https://github.com/jhong93/spot at
edec4201471beed631bed374bd0b95fcdc8a2f4f (BSD-3, see LICENSE; GSM and TSM
carry their own licences in their headers).

Kept: model/shift.py, model/impl/gsm.py, model/impl/tsm.py, and the
inference parts of train_e2e.py's E2EModel as model.py. Changed: package-
relative imports; timm 1.x's `timm.layers.ConvNormAct` for the old
`timm.models.layers.conv_bn_act.ConvBnAct`; GSM's zero padding built with
`torch.zeros_like` instead of `torch.cuda.FloatTensor`, so it runs on MPS
and CPU; progress prints removed.

Weights: tennis_rny008gsm_gru_rgb, checkpoint_041.pt from
https://github.com/jhong93/e2e-spot-models, in weights/e2e_spot/.
