import yaml
from easydict import EasyDict as edict


class Config:
    @classmethod
    def load_from_yaml(cls, filename):
        with open(filename, 'r') as f:
            return edict(yaml.load(f, Loader=yaml.SafeLoader))
