from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from typing import Optional,Union
import os, importlib

def get_default_config_arg_parser(arg_parser:Optional[ArgumentParser],
                                  name="-c",
                                  flag="--config",
                                  metavar="config.yaml",
                                  help="paths to base configs. Loaded from left-to-right. "
                "Parameters can be overwritten or added with command-line options of the form `--key value`.",
                **kwargs) -> ArgumentParser:
    """
    Returns a default ArgumentParser for configuration files.

    Args:
        arg_parser (ArgumentParser, optional): An existing ArgumentParser. Defaults to None.
        name (str, optional): The name of the argument. Defaults to "-c".
        flag (str, optional): The flag of the argument. Defaults to "--config".
        metavar (str, optional): The metavar of the argument. Defaults to "config.yaml".
        help (str, optional): The help message of the argument. Defaults to "paths to base configs. Loaded from left-to-right. "
                "Parameters can be overwritten or added with command-line options of the form `--key value`.".
        **kwargs: Additional keyword arguments for the ArgumentParser.

    Returns:
        ArgumentParser: The ArgumentParser.
    """
    if arg_parser is None:
        arg_parser = ArgumentParser(**kwargs)
    arg_parser.add_argument(
        name,
        flag,
        nargs="*",
        metavar=metavar,
        help=help,
        default=list(),
    )
    return arg_parser


def load_external_config(config: DictConfig,
                   external_file_key: str = "_file"
                   ) -> OmegaConf:
    """
    Recursively parses a configuration dictionary and returns an OmegaConf object.
    If a configuration item is a file, it will be loaded and merged into the configuration.

    Args:
        config (dict): The configuration dictionary to parse.
        external_file_key (str, optional): The key used to identify external files. Defaults to "_file".

    Returns:
        OmegaConf: The parsed configuration as an OmegaConf object.
    """
    conf_ = OmegaConf.create({})
    for key, value in config.items_ex(resolve=False):
        if (isinstance(value, dict) or isinstance(value, OmegaConf)
                or isinstance(value, DictConfig)):
            conf_[key] = load_external_config(value)
        elif isinstance(value, str) and key == external_file_key:
            conf_ = OmegaConf.merge(conf_, load_external_config(OmegaConf.load(value)))
        else:
            conf_[key] = value
    return conf_

def create_config_from_args(args: ArgumentParser,
                            config_key: str = "config",
                            default_config_path: Optional[str] = None,
                            additional_config_files: Optional[list] = None,
                            additional_configs: Optional[Union[DictConfig,dict]]=None,
                            external_file_key: str = "_file",
                            resolve=True,
                            dict=False
                            ) -> OmegaConf:
    """
    Creates a configuration object from the parsed arguments.

    Args:
        args (ArgumentParser): The parsed arguments.
        config_key (str, optional): The key used to identify the configuration file. Defaults to "config".
            The `config_key` in the arguments should be a list, here is an example:
            ```
            parser.add_argument(
                "-c",
                "--config",
                nargs="*",
                metavar="config.yaml",
                help="paths to base configs. Loaded from left-to-right. "
                        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
                default=list(),
            )
            ```
            You can use `get_default_config_arg_parser` function to create the ArgumentParser.
        default_config_path (str, optional): The default configuration file path. Defaults to None. If not specified, the value of the `config_key` needs to be a full path.
        additional_config_files (list, optional): Additional configuration files to load. Defaults to None.
        additional_configs (Union[DictConfig,dict], optional): Additional configurations to load. Defaults to None.
        external_file_key (str, optional): The key used to identify external files. Defaults to "_file".
        resolve (bool, optional): Whether to resolve the configuration. Defaults to True.
        dict (bool, optional): Whether to return the configuration as a dictionary. Defaults to False.

    Returns:
        OmegaConf: The configuration object.
    """
    opt, unknown = args.parse_known_args()
    if default_config_path is not None:
        config_paths=[os.path.join(default_config_path,cfg) for cfg in getattr(opt, config_key)]
    else:
        config_paths = getattr(opt, config_key)
    if additional_config_files is not None:
        config_paths.extend(additional_config_files)
    configs = [load_external_config(OmegaConf.load(cfg_path),external_file_key=external_file_key) for cfg_path in config_paths]
    if additional_configs is not None:
        if isinstance(additional_configs,dict):
            additional_configs=OmegaConf.create(additional_configs)
        configs.append(additional_configs)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.create(OmegaConf.merge(*configs, cli))
    if resolve:
        OmegaConf.resolve(config)
    if dict:
        return OmegaConf.to_container(config)
    return config

def create_config_from_file(config_path: str) -> OmegaConf:
    """
    Creates a configuration object from a file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        OmegaConf: The configuration object.
    """
    return OmegaConf.load(config_path)

def load_object(config:Union[DictConfig,dict],
                object_key:str="_object",
                params_key:str="_params",
                ):
    if isinstance(config,dict):
        config=OmegaConf.create(config)
    if not object_key in config:
        raise ValueError(f"Key {object_key} not found in the configuration")
    object_params={}
    if params_key in config:
        for key,value in config[params_key].items():
            if isinstance(value,dict) or isinstance(value,DictConfig):
                if "_object" in value.keys():
                    value=load_object(value)
            object_params[key]=value
    obj_names=config[object_key].split(".")
    return getattr(importlib.import_module(".".join(obj_names[0:-1])),obj_names[-1])(**object_params)