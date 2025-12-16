import yaml
import easydict

class HyperParameters():
    def __init__(self, config):
        self.unet_dim = config.unet_cfg.dim
        self.time_steps = config.diff_cfg.time_steps
        self.beta_last = config.diff_cfg.beta_last
        self.cond_weight = config.diff_cfg.cond_weight
        self.uncond_prop = config.diff_cfg.uncond_prop
        self.learning_rate = config.train_cfg.learning_rate
        self.dropout_rate = config.train_cfg.dropout_rate
        self.end_epoch = config.train_cfg.end_epoch



def load_cfg(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        config = easydict.EasyDict(config)
    hyper_params = HyperParameters(config)
    return hyper_params

if __name__ == "__main__":
    config = load_cfg("HPO_dropout.yaml")