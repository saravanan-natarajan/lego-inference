import argparse
import json
import logging
import sys
import torch


# grab some arguments
def args_parse():
    # ArgumentParser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Provide a web service to run inference against pytorch model')
    parser.add_argument('--config', help='Name of the config set to be used', default='default.conf')
    parser.add_argument('--log', help='Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)', default='INFO')
    args = parser.parse_args()
    return args


# create the standard logger
def my_logger(log_file_name, log_level):
    logging.basicConfig(filename=log_file_name,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(filename)s %(levelname)s: %(message)s')
    my_logger = logging.getLogger(__name__)

    # add a handler to log to the console also
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    my_logger.addHandler(console_handler)
    return my_logger


def load_config(config_file):
    config_dict = {}
    with open(config_file) as my_config:
        config_dict = json.load(my_config)
    my_config.close()
    return config_dict


def device_used():
    if torch.cuda.is_available():
        return 'cuda'
    # elif torch.backends.mps.is_available():
    #     return 'mps'
    else:
        return 'cpu'
