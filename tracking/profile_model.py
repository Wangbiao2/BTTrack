import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
from lib.models.bttrack.vit import Attention
from lib.models.bttrack.vit import vit_base_patch16_224, vit_tiny_patch16_224


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='bttrack', choices=['bttrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vit_tiny_ep300', help='yaml configure file name')
    args = parser.parse_args()

    return args

def custom_module_flops(module, input, output):
    total_ops = 0
    B, N, C = input[0].shape
    total_ops += B * 4 * N * C ** 2
    # total_ops += B * 2 * C * N ** 2
    type_1 = 64 * 64 + 1 * 65 + 256 * 257
    type_2 = 64 * 64 + 1 * 65 + 256 * 321
    type_3 = 64 * 1 + 1 * 320 + 256 * 1
    type_4 = 64 * 256 + 1 * 320 + 256 * 64

    total_ops += B * 2 * C * (type_1 * 4 + type_2 * 0 + type_3 * 4 + type_4 * 2) / 12.
    module.total_ops += torch.DoubleTensor([int(total_ops)])

custom_ops = {
    Attention: custom_module_flops,
}

def evaluate_vit(model, template, search, template_anno):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search, template_anno),
                             custom_ops=custom_ops, verbose=True)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "bttrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_bttrack
        model = model_constructor(cfg, training=False)

        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        template_anno = torch.randn(1, bs, 4)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        template_anno = template_anno.to(device)

        evaluate_vit(model, template, search, template_anno)


    else:
        raise NotImplementedError