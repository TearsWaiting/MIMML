"""
Project for meta learning in peptide
Please set the config in the 'config' folder

Usage:
    run.py pretrain

Options:
    -h --help                               show this screen.
"""

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from docopt import docopt
from config import config_SL
from Framework import Learner


def pretrain_BPD_ALL_RT():
    config = config_SL.get_config()
    config.learn_name = 'SL_pretrain_BPD_ALL_RT'
    config.path_dataset = '../data/task_data/Meta Dataset/BPD-ALL-RT'
    config.alpha = None

    config.num_meta_train = 15
    config.num_meta_valid = 15
    config.num_meta_test = 15
    config.num_class = 16

    config.threshold = 0.30
    config.batch_size = 320
    config.lr = 0.0001

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


if __name__ == '__main__':
    print('__doc__', __doc__)
    args = docopt(__doc__)
    print('args', args)

    if args['pretrain']:
        pretrain_BPD_ALL_RT()
