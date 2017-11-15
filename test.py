#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation
from erfnet.data_util.cityscapes.cityscapes_utils import cityscapes_label_colors
from erfnet.data_util.cityscapes.cityscapes_utils import cityscapes_label_names
from erfnet.config_utils import *
from chainercv.utils import apply_prediction_to_iterator
from chainercv.evaluations import eval_semantic_segmentation

from erfnet.config_utils import *

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

from erfnet.models import erfnet_paper

def test_erfnet():
    """Demo ERFNet."""
    config, img_path = parse_args()
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data, config['iterator'])
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'])

    if devices:
        model.to_gpu(devices['main'])

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, test_iter)

    del imgs
    pred_labels, = pred_values
    gt_labels, = gt_values

    result = eval_semantic_segmentation(pred_labels, gt_labels)

    for iu, label_name in zip(result['iou'], cityscapes_label_names):
        print('{:>23} : {:.4f}'.format(label_name, iu))
    print('=' * 34)
    print('{:>23} : {:.4f}'.format('mean IoU', result['miou']))
    print('{:>23} : {:.4f}'.format(
        'Class average accuracy', result['mean_class_accuracy']))
    print('{:>23} : {:.4f}'.format(
        'Global average accuracy', result['pixel_accuracy']))

def main():
    test_erfnet()

if __name__ == '__main__':
    main()
