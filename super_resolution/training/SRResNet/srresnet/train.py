# flake8: noqa
import sys
import os.path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

import srresnet.archs
import srresnet.data
import srresnet.models
from basicsr.train import train_pipeline

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
