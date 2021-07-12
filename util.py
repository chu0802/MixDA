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

def config_handler(hashstr, hash_table_path):
    # storing config before hashing
    with open(hash_table_path, 'r') as f:
        table = pk.load(f)
    table[hashstr] = m_cfg
    with open(hash_table_path, 'w') as f:
        pk.dump(table, f)
