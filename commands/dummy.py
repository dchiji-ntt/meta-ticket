
from pprint import PrettyPrinter
import copy

def dummy(exp_name, cfg, prefix=""):
    print(f'exp_name: {exp_name}')
    print(f'prefix: {prefix}')

    print(f'cfg: ')
    cfg_cp = copy.deepcopy(cfg)
    del cfg_cp['__other_configs__']
    pp = PrettyPrinter(indent=1)
    pp.pprint(cfg_cp)

