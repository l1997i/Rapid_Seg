"""Train R-/C-RAPiD-Seg on SemanticKITTI.

This is a thin wrapper around ``scripts/train.py``; see that file for all
options. On an NVIDIA GPU with MinkowskiEngine the MinkUNet34 backbone is used
automatically.

    # official .bin/.label layout
    python examples/02_train_semantickitti.py --dataset full --data_root /path/to/dataset

    # preprocessed .pth subset
    python examples/02_train_semantickitti.py --dataset sp --variant C
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import main

if __name__ == "__main__":
    main()
