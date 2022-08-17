import subprocess


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


class Config:
    def __init__(self, d: dict):
        for k,v in d.items():
            setattr(self, k, v)
            
            
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=("index", "uuid", "name", "timestamp", "memory.total", "memory.free", "memory.used", "utilization.gpu", "utilization.memory"), no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]


def get_default_parameter(model_name, param_list):
    if "/" in model_name: model_name = model_name.rsplit("/", 1)[1]
    default = {"deberta-v3-base": {"epoch":[10, 15, 20],
                                "batch_size":16,
                                "gradient_clipping":1,
                                "warmup_rate":[50, 100, 500, 1000],
                                "lr":[1.5e-5, 2e-5, 3e-5, 4e-5],
                                "dropout":[0, 0.1, 0.15],
                                "weight_decay":0.01,
                                "beta1":0.9,
                                "beta2":0.999,
                                "epsilon":1e-6,
                                "scheduler":"linear"},
              "deberta-v3-large": {"epoch":[10 ,15, 20],
                                "batch_size":8,
                                "gradient_clipping":1,
                                "warmup_rate":[50, 100, 500, 1000],
                                "lr":[5e-6, 8e-6, 9e-6, 1e-5],
                                "dropout":[0, 0.1, 0.15],
                                "weight_decay":0.01,
                                "beta1":0.9,
                                "beta2":0.999,
                                "epsilon":1e-6,
                                "scheduler":"linear"},
               "deberta-base": {"epoch":[10, 15, 20],
                                "batch_size":16,
                                "gradient_clipping":1,
                                "warmup_rate":[50, 100, 500, 1000],
                                "lr":[1.5e-5, 2e-5, 2.5e-5, 3e-5],
                                "dropout":[0, 0.1, 0.15],
                                "weight_decay":0.01,
                                "beta1":0.9,
                                "beta2":0.999,
                                "epsilon":1e-6,
                                "scheduler":"linear"},
               "deberta-large": {"epoch":[10 ,15, 20],
                                "batch_size":8,
                                "gradient_clipping":1,
                                "warmup_rate":[50, 100, 500, 1000],
                                "lr":[5e-6, 8e-6, 9e-6, 1e-5],
                                "dropout":[0, 0.15, 0.3],
                                "weight_decay":0.01,
                                "beta1":0.9,
                                "beta2":0.999,
                                "epsilon":1e-6,
                                "scheduler":"linear"},
               "roberta-base": {"epoch":10,
                                "batch_size":16,
                                "gradient_clipping":1,
                                "warmup_rate":100,
                                "lr":2e-5,
                                "dropout":0.1,
                                "weight_decay":0.1,
                                "beta1":0.9,
                                "beta2":0.999,
                                "epsilon":1e-8,
                                "scheduler":"linear"},
               "roberta-large": {"epoch":10,
                                "batch_size":8,
                                "gradient_clipping":1,
                                "warmup_rate":100,
                                "lr":1e-5,
                                "dropout":0.1,
                                "weight_decay":0.1,
                                "beta1":0.9,
                                "beta2":0.999,
                                "epsilon":1e-8,
                                "scheduler":"linear"}
              }
    return [default["model_name"][param] for param in param_list]
    