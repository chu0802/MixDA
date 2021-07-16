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
    def __init__(self, host):
        self.processes = {}
        self.host = host

    def run(self, logdir, cfg):
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir, '--host', self.host])
        url = tb.launch()
        self.processes[cfg] = url

    def list(self, cont='c'):
        print('All starting up processes: ')
        for k, v in self.processes.items():
            print('%s: %s' % (k, v))

        print('press %s to continue...' % (cont))
        while True:
            opt = input()
            if opt == cont:
                break

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
    )

    tftool = TensorboardTool(args.host)
    while True:
        cfg = mdh.select_config()
        if cfg:
            log_dir = mdh.get_log(cfg)
            tftool.run(str(log_dir), cfg=str(cfg))
        else:
            tftool.list()
        
            
    print('\nShutting Down')
