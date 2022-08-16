import argparse, json
from utils_data import create_cv_data


def main(config):
    create_cv_data(config)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_preprocess.json', help='specify the config file')
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    
    main(config)