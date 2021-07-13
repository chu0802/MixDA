from tensorboard import program
import logging
from util import config_loading

import argparse
import pickle as pk
import os
from pathlib import Path, PurePath
from random import randint


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

def get_contents(model_path, hashtable_path):
    # Sorted according to the creating date
    dirs = [PurePath(x).name for x in sorted(Path(model_path).iterdir(), key=os.path.getmtime)]
    # Get the hashtable (mapping hashstr to model config)
    with open(hashtable_path, 'rb') as f:
        hashtable = pk.load(f)

    # List all the logging file
    log_path, opts = [], []
    for i, hashstr in enumerate(dirs):
        log_path.append(model_path / hashstr / 'log')
        opts.append('\t(%d): %s' % (i, str(hashtable[hashstr])))

    return log_path, opts

def list_contents(opts):
    os.system('clear')
    print('\n', '-'*10, 'Tensorboard Binding System', '-'*10, '\n')

    print('Options:')
    for opt in opts: 
        print(opt)
    print('\t(r): Refresh the options')

if __name__ == '__main__':
    # Get configuration
    args = argument_parsing()
    args.config = config_loading(args.config) 
    model_path = Path(args.config['datasets'][args.dataset]['path']) / 'model'

    log_path, opts = get_contents(model_path, args.config['hash_table_path'])
    list_contents(opts)
    # Start binding
    while True:
        print('Select an option: ', end='')
        try:
            opt = input()
            if opt == 'r':
                log_path, opts = get_contents(model_path, args.config['hash_table_path'])
                list_contents(opts)
            if opt.isdigit() and (0 <= int(opt) < len(log_path)):
                tftool = TensorboardTool(str(log_path[int(opt)]), args.host)
                tftool.run()
        except KeyboardInterrupt:
            break
    print('\nShutting Down')


            




