BST (Badminton Stroke-type Transformer), vendored from
https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer at
fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df (MIT, see LICENSE; Chang, CVPRW 2026).

Only `stroke_classification/model/bst.py` and `tempose.py`, with imports
made package-relative and two dev-only dependencies removed
(`torchinfo`, and `positional_encodings` replaced by `_pos_enc.py`).

Weights: the TenniSet-trained BST-AP checkpoint, `bst_AP_JnB_bone.pt`, from
the authors' Google Drive folder linked in their README, in `weights/bst/`.
