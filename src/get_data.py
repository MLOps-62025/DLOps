import argparse
import yaml

def get_data(config_file):
    config = read_params(config_file)
    return config

def read_params(config_file):
    with open(config_file) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passes_args= args.parse_args()
    a = get_data(config_file=passes_args.config)