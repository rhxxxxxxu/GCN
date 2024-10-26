#!/usr/bin/env python
from __future__ import print_function

import argparse
import pickle
import sys
import time
import traceback
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# torchlight
from torchlight.torchlight import DictAction


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--weights', default='./work_dir/uav/algcn_uav/runs-40-5120.pt', help='the weights for network initialization')
    parser.add_argument('--model', default='model.algcn_uav.Model', help='the model will be used')
    parser.add_argument('--model-args', action=DictAction,
                        default=dict(num_class=155, num_point=17, num_person=2, graph='graph.uav.Graph',
                                     graph_args=dict(labeling_mode='spatial')), help='the arguments of model')
    parser.add_argument('--device', type=int, default=0, help='the indexes of GPUs for testing')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--feeder', default='feeders.feeder_uav.Feeder', help='data loader will be used')
    parser.add_argument('--test-feeder-args', action=DictAction,
                        default=dict(data_path='data/uav/test_data.npz', split='test', window_size=64,
                                     p_interval=[0.95], vel=False, bone=False, debug=False),
                        help='the arguments of data loader for test')
    parser.add_argument('--work-dir', default='./work_dir/test_uav', help='the work folder for storing results')
    parser.add_argument('--save-score', type=str2bool, default=True,
                        help='if true, the classification score will be stored')
    parser.add_argument('--phase', default='test', help='must be test')
    return parser


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Tester:
    def __init__(self, arg):
        self.arg = arg
        self.load_model()
        self.load_data()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

    def load_model(self):
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(self.arg.device)
        if self.arg.weights:
            model_statue = torch.load(self.arg.weights, map_location=lambda storage, loc: storage.cuda(self.arg.device))
            self.model.load_state_dict(
                model_statue
            )
        self.model.eval()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)

    def eval(self, epoch, save_score=False):
        self.model.eval()
        self.print_log(f'Evaluating epoch: {epoch + 1}')

        # Ensure the output directory exists
        os.makedirs(self.arg.work_dir, exist_ok=True)

        score_frag = []
        process = tqdm(self.data_loader, ncols=40)

        for batch_idx, (data, _, index) in enumerate(process):  # Ignore labels using _
            with torch.no_grad():
                data = data.float().cuda(self.arg.device)
                output = self.model(data)
                score_frag.append(output.data.cpu().numpy())

        score = np.concatenate(score_frag)
        print("Shape of score:", score.shape)
        if save_score:
            score_path = os.path.join(self.arg.work_dir, 'pred.npy')
            np.save(score_path, score)
            self.print_log(f'Saved scores to {score_path}')


def main():
    parser = get_parser()
    arg = parser.parse_args()
    tester = Tester(arg)
    tester.eval(epoch=0, save_score=arg.save_score)


if __name__ == '__main__':
    main()
