import yaml
import pickle as pk
import hashlib
from pathlib import Path, PurePath
import os

def config_loading(cfg_path):
    return yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader) if cfg_path is not None else None

class model_handler:
    def __init__(self, model_dir, hash_table_path, title=None, allow_none=False):
        self.model_dir = model_dir

        self.hash_table_path = hash_table_path
        self.title = title
        self.allow_none = allow_none

        self.hash_table = {}
        self.models = []
        self.load()
        
    def load(self):
        # Load the hash table (mapping hashstr to model config)
        self.hash_table = (
            pk.load(open(self.hash_table_path, 'rb'))
            if Path(self.hash_table_path).exists()
            else {}
        )

        # Sorted according to the modifying date
        dirs = [PurePath(x).name for x in sorted(Path(self.model_dir).iterdir(), key=os.path.getmtime)]

        self.models.clear()
        # List all the logging file
        for i, hashstr in enumerate(dirs):
            model = {}
            model['log_dir'] = self.model_dir / hashstr / 'log'
            model['config'] = self.hash_table[hashstr]

            self.models.append(model)

    def list(self):
        os.system('clear')
        if self.title:
            print('\n', '-'*10, self.title, '-'*10, '\n')

        print('Options:')
        for i, model in enumerate(self.models): 
            print('\t(%d): %s' % (i, str(model['config'])))
        print('\t(r): Refresh the options')

        if self.allow_none:
            print('\t(n): None')

    def select(self, slogan=None):
        while True:
            if slogan:
                print(slogan, end=': ')
            opt = input()
            if opt == 'r':
                self.load()
                self.list()
            elif self.allow_none and opt == 'n':
                return None
            elif opt.isdigit() and (0 <= int(opt) < len(self.models)):
                return self.models[int(opt)]
            else:
                print('\tInvalid')
    def update(self, cfg):
        hashstr = hashlib.md5(str(cfg).encode('utf-8')).hexdigest()
        self.hash_table[hashstr] = cfg
        with open(self.hash_table_path, 'wb') as f:
            pk.dump(self.hash_table, f)
        log_dir = self.model_dir / hashstr / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.load()
        return self.model[-1]
