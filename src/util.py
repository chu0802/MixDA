import yaml
import pickle as pk
import hashlib

def config_loading(cfg_path):
    if cfg_path is None:
        return None
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def config_hashing(m_cfg):
    return hashlib.md5(str(m_cfg).encode('utf-8')).hexdigest()

def config_handler(hashstr, args):
    m_cfg = args.config['model']
    hash_table_path = args.config['hash_table_path']
    # storing config before hashing
    try:
        with open(hash_table_path, 'rb') as f:
            table = pk.load(f)
    except:
        table = {}
    table[hashstr] = m_cfg
    with open(hash_table_path, 'wb') as f:
        pk.dump(table, f)
