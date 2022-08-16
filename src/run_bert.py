import argparse, datetime, json, os
from train_bert import train
from test_bert import test



def main(config, dirpath):
    train(config, dirpath)
    test(dirpath)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_bert.json', help='specify the config file')
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    
    model_name = config["network"]["model_name"]
    if "/" in model_name: model_name = model_name.rsplit("/", 1)[1]
    number_of_date = config["train"]["number_of_date"]
    dt_now = datetime.datetime.now()
    dirpath = os.path.join("../results", "{:02}-{}{}".format(dt_now.day, model_name, number_of_date))
    # dirpath = os.path.join("../results", "{:02}".format(dt_now.day))
    main(config, dirpath)