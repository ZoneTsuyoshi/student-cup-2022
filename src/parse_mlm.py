import argparse
import json
import copy
import pathlib
from distutils.util import strtobool

from run_mlm import main
from utils import is_num


f = open("config_mlm.json", "r")
config = json.load(f)
f.close()

flatten_config = {}
for key in config.keys():
    flatten_config.update(config[key])
# print(flatten_config)

parser = argparse.ArgumentParser(description="subprocess for grid search")

for key in flatten_config.keys():
    if type(flatten_config[key])==bool:
        parser.add_argument("--{}".format(key), default=flatten_config[key], type=strtobool)
    elif type(flatten_config[key])==list:
        parser.add_argument("--{}".format(key), default=flatten_config[key], type=type(flatten_config[key][0]) if len(flatten_config[key])>0 else None, nargs="*")
    elif type(flatten_config[key])==str and "/" in flatten_config[key]:
        parser.add_argument("--{}".format(key), default=flatten_config[key], type=pathlib.Path)
    elif flatten_config[key] is None or type(flatten_config[key])==int:
        parser.add_argument("--{}".format(key), default=flatten_config[key])
    else:
        parser.add_argument("--{}".format(key), default=flatten_config[key], type=type(flatten_config[key]))

new_flatten_config = vars(parser.parse_args())

new_config = copy.deepcopy(config)
for top_key in config.keys():
    for key in config[top_key].keys():
        if type(config[top_key][key])==bool:
            new_config[top_key][key] = bool(new_flatten_config[key])
        elif type(config[top_key][key])==str and "/" in config[top_key][key]:
            new_config[top_key][key] = str(new_flatten_config[key])
        elif new_flatten_config[key]=="None" or new_flatten_config[key] is None:
            new_config[top_key][key] = None
        elif type(new_flatten_config[key])==str:
            if str.isdigit(new_flatten_config[key]):
                new_config[top_key][key] = int(new_flatten_config[key])
            elif is_num(new_flatten_config[key]):
                new_config[top_key][key] = float(new_flatten_config[key])
            else:
                new_config[top_key][key] = new_flatten_config[key]
        else:
            new_config[top_key][key] = new_flatten_config[key]
            
main(new_config)