import yaml

def read_config(file):
    with open(file,'r') as stream:
        config = yaml.full_load(stream)
    return config