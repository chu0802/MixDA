from tensorboard import program
import logging

import argparse
import pickle as pk
import os
from pathlib import Path, PurePath
from random import randint
import sys
sys.path.append('../src/')

from util import config_loading, model_handler

class TensorboardTool:
    def __init__(self, logdir, host):
        self.dir_path = logdir
        self.host = host

    def run(self):
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.dir_path, '--host', self.host])
        url = tb.launch()
        print('TensorBoard at %s \n' % url)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome')
    parser.add_argument('--host', type=str, default='clais1.csie.org')

    return parser.parse_args()


if __name__ == '__main__':
    # Get configuration
    args = argument_parsing()
    args.config = config_loading(args.config) 
    args.dataset = args.config['datasets'][args.dataset]
    model_path = Path(args.dataset['path']) / 'model'

    mdh = model_handler(
        model_path, 
        args.config['hash_table_path'], 
        title='Tensorboard Binding System',
        allow_none=False
    )
    mdh.list()

    while True:
        try:
            model = mdh.select(slogan='Select a model')
            tftool = TensorboardTool(str(model['log_dir']), args.host)
            tftool.run()
        except KeyboardInterrupt:
            break
    print('\nShutting Down')
