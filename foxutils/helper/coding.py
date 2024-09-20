#usr/bin/python3

#version:0.0.9
#last modified:20240102

from inspect import isfunction
import time,yaml
import torch,random
import numpy as np
import os
from typing import Dict,Any,Optional,Callable,Sequence

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def seconds_to_hms(seconds,str=True):
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    if str:
        return "%02dh:%02dm:%02ds" % (h, m, s)
    else:
        return h,m,s

def time_estimation(start_time, index_now, end_index, start_index=0):
    """
    Calculate the estimated time remaining based on the current progress.

    Args:
        start_time (float): The start time of the process.
        index_now (int): The current index of the process.
        end_index (int): The final index of the process.
        start_index (int, optional): The starting index of the process. Defaults to 0.

    Returns:
        str: A string representing the estimated time remaining or the total time used.
    """
    time_now = time.perf_counter()

    def format_time(seconds):
        """
        Format the given time in seconds to HH:MM:SS format.

        Args:
            seconds (int): The time in seconds.

        Returns:
            str: The formatted time in HH:MM:SS format.
        """
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "{:0>2d}:{:0>2d}:{:0>2d}".format(h, m, s)

    used = time_now - start_time

    if index_now != end_index:
        left = used * (end_index - index_now) / (index_now - start_index + 1)
        return "Time used: {} Time left: {}".format(format_time(used), format_time(left))
    else:
        return "Total time used: {}".format(format_time(used))

class GeneralDataClass():
    
    def __init__(self,generation_dict=None,**kwargs) -> None:
        if generation_dict is not None:
            for key,value in generation_dict.items():
                self.set_item(key,value)
        for key,value in kwargs.items():
            self.set_item(key,value)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self,key):
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__.items())
    
    def keys(self):
        return self.__dict__.keys()

    def set_item(self,key,value):
        if isinstance(value,dict):
            setattr(self,key,GeneralDataClass(value))
        else:
            setattr(self,key,value)

    def set_items(self,**kwargs):
        for key,value in kwargs.items():
            self.set_item(key,value)
    
    def remove(self,*args):
        for key in args:
            delattr(self,key)

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self.__dict__)
    
    def to_dict(self):
        output={}
        for key,value in self.__dict__.items():
            if isinstance(value,GeneralDataClass):
                output[key]=value.to_dict()
            else:
                output[key]=value
        return output

class ConfigurationsHandler():
    """
    A class that handles configurations for a specific application or module.
    """
    _configs_feature:dict
    _configs:GeneralDataClass
    _config_changed=True
        
    def add_config_item(self,
                        name:str,
                        default_value:Optional[Any]=None,
                        default_value_func:Optional[Callable]=None,
                        mandatory:bool=False,
                        description:str="",
                        value_type=None,
                        options:Optional[Sequence]=None,
                        in_func:Optional[Callable]=None,
                        out_func:Optional[Callable]=None):
        """
        Adds a new configuration item to the handler.

        Args:
            name (str): The name of the configuration item.
            default_value (Any, optional): The default value for the configuration item. Defaults to None.
            default_value_func (Callable, optional): A function that returns the default value for the configuration item. Defaults to None.
            mandatory (bool, optional): Indicates whether the configuration item is mandatory. Defaults to False.
            description (str, optional): The description of the configuration item. Defaults to "".
            value_type (type, optional): The expected type of the configuration item. Defaults to None.
            options (List[Any], optional): The list of possible values for the configuration item. Defaults to None.
            in_func (Callable, optional): A function to transform the input value of the configuration item. Defaults to None.
            out_func (Callable, optional): A function to transform the output value of the configuration item. Defaults to None.
        """
        #if not mandatory and default_value is None and default_value_func is None:
        #    raise Exception("Default value or default value func must be set for non-mandatory configuration.")
        if not hasattr(self,"_configs_feature"):
            self._configs_feature={}
        if not hasattr(self,"_configs"):
            self._configs=GeneralDataClass()
        if not mandatory and default_value is not None and default_value_func is not None:
            raise Exception("Default value and default value func must not be set at the same time for non-mandatory configuration.")
        if mandatory and (default_value is not None or default_value_func is not None):
            raise Exception("Default value or default value func must not be set for mandatory configuration.")
        if default_value is not None and not isinstance(default_value,value_type):
            raise Exception("Default value must be {}, but find {}.".format(value_type,type(default_value)))
        if options is not None:
            if type(options)!=list:
                raise Exception("Option must be list, but find {}.".format(type(options)))
            if len(options)==0:
                raise Exception("Option must not be empty.")
            for item in options:
                if not isinstance(item,value_type):
                    raise Exception("Option must be list of {}, but find {}.".format(value_type,type(item)))
        self._configs_feature[name]={
            "default_value_func":default_value_func, #default_value_func must be a function with one parameter, which is the current configures
            "mandatory":mandatory,
            "description":description,
            "value_type":value_type,
            "options":options,
            "in_func":in_func,
            "out_func":out_func,
            "default_value":default_value,
            "in_func_ran":False,
            "out_func_ran":False
        }
        self._config_changed=True
    
    def get_config_features(self,key):
        """
        Retrieves the features of a specific configuration item.

        Args:
            key (str): The name of the configuration item.

        Returns:
            dict: A dictionary containing the features of the configuration item.
        """
        if key not in self._configs_feature.keys():
            raise Exception("{} is not a supported configuration.".format(key))
        return self._configs_feature[key]
    
    def set_config_features(self,key,**feature):
        """
        Sets the features of a specific configuration item.

        Args:
            key (str): The name of the configuration item.
            feature (dict): A dictionary containing the features of the configuration item.
        """
        self.add_config_item(key,**feature)
        self._config_changed=True

    def set_config_items(self,**kwargs):
        """
        Sets the values of multiple configuration items.

        Args:
            kwargs (Any): Keyword arguments representing the configuration items and their values.
        """
        for key in kwargs.keys():
            if key not in self._configs_feature.keys():
                raise Exception("{} is not a supported configuration.".format(key))
            if self._configs_feature[key]["value_type"] is not None and not isinstance(kwargs[key],self._configs_feature[key]["value_type"]):
                raise Exception("{} must be {}, but find {}.".format(key,self._configs_feature[key]["value_type"],type(kwargs[key])))
            if self._configs_feature[key]["options"] is not None and kwargs[key] not in self._configs_feature[key]["options"]:
                raise Exception("{} must be one of {}, but find {}.".format(key,self._configs_feature[key]["options"],kwargs[key]))
            self._configs.set_item(key,kwargs[key])
            self._configs_feature[key]["in_func_ran"]=False
            self._configs_feature[key]["out_func_ran"]=False
        self._config_changed=True
    
    @property
    def configs(self):
        """
        Retrieves the current configurations.

        Returns:
            GeneralDataClass: An instance of the GeneralDataClass containing the current configurations.
        """
        if self._config_changed:
            for key in self._configs_feature.keys():
                not_set=False
                if not hasattr(self._configs,key):
                    not_set=True
                elif self._configs[key] is None:
                    not_set=True
                if not_set:
                    if self._configs_feature[key]["mandatory"]:
                        raise Exception("Configuration {} is mandatory, but not set.".format(key))
                    elif self._configs_feature[key]["default_value"] is not None:
                        self._configs.set_item(key,self._configs_feature[key]["default_value"])
                        self._configs_feature[key]["in_func_ran"]=False
                        self._configs_feature[key]["out_func_ran"]=False
                    elif self._configs_feature[key]["default_value_func"] is not None:
                        self._configs.set_item(key,None)        
                    else:
                        raise Exception("Configuration {} is not set.".format(key))
            #default_value_func and infunc may depends on other configurations
            for key in self._configs.keys():
                if self._configs[key] is None and self._configs_feature[key]["default_value_func"] is not None:
                    self._configs.set_item(key,self._configs_feature[key]["default_value_func"](self._configs))
                    self._configs_feature[key]["in_func_ran"]=False
                    self._configs_feature[key]["out_func_ran"]=False
            for key in self._configs_feature.keys():
                if self._configs_feature[key]["in_func"] is not None and not self._configs_feature[key]["in_func_ran"]:
                    self._configs.set_item(key,self._configs_feature[key]["in_func"](self._configs[key],self._configs))
                    self._configs_feature[key]["in_func_ran"]=True
        self._config_changed=False
        return self._configs

    def read_configs_from_yaml(self,yaml_file:str):
        """
        Sets the values of configuration items from a YAML file.

        Args:
            yaml_file (str): The path to the YAML file.
        """
        with open(yaml_file,"r") as f:
            yaml_configs=yaml.safe_load(f)
        self.set_config_items(**yaml_configs)
        
    def to_yaml(self,only_optional=False,with_description=True):
        output_dict=self.str_dict(only_optional,sort=True)
        if with_description:
            yaml_str=""
            for key,value in output_dict.items():
                yaml_str+="# "+self.get_config_description(key)+os.linesep+yaml.dump({key:value})+os.linesep
        else:
            yaml_str=yaml.dump(output_dict)
        return yaml_str       
    
    def save_configs_to_yaml(self,
                             yaml_file:str,
                             only_optional=False,
                             with_description=True,):
        """
        Saves the values of configuration items to a YAML file.

        Args:
            yaml_file (str,): The path to the YAML file. 
            only_optional (bool, optional): Indicates whether to save only the optional configuration items. Defaults to False.
        """
        yaml_str=self.to_yaml(only_optional,with_description)
        with open(yaml_file,"w") as f:
            f.write(yaml_str)
    
    def str_dict(self,only_optional=False,sort=True) -> dict:
        """
        Retrieves the current configurations as a dictionary whose item are all strs.
        
        Args:
            only_optional (bool, optional): Indicates whether to retrieve only the optional configuration items. Defaults to False.
            sort (bool, optional): Indicates whether to sort the dictionary alphabetically. Defaults to True.
            
        Returns:
            dict: A dictionary containing the current configurations.
        """
        config_dict=self.configs.to_dict()
        if only_optional:
            output_dict={}
            for key in config_dict.keys():
                if self._configs_feature[key]["mandatory"]:
                    continue
                output_dict[key]=config_dict[key]    
        else:
            output_dict=config_dict
        for key in output_dict.keys():
            if self._configs_feature[key]["out_func"] is not None and not self._configs_feature[key]["out_func_ran"]:
                output_dict[key]=self._configs_feature[key]["out_func"](self._configs[key],self._configs)
                self._configs_feature[key]["out_func_ran"]=True
        if sort:
            return self._sorted_dict(output_dict)
        else:
            return output_dict
          
    def info_available_configs(self,print_info=True,sort=True) -> str:
        """
        Shows the available configuration items and their descriptions.
        
        Args:
            print_info (bool, optional): Indicates whether to print the information. Defaults to True.
            
        Returns:
            str: A string containing the information.
            sort (bool, optional): Indicates whether to sort the dictionary alphabetically. Defaults to True.
        """
        mandatory_configs=[]
        optional_configs=[]
        target_dict=self._sorted_dict(self._configs_feature) if sort else self._configs_feature
        for key in  target_dict.keys():
            text="    "+self.get_config_description(key)
            if self._configs_feature[key]["mandatory"]:
                mandatory_configs.append(text)
            else:
                optional_configs.append(text)
        output=["Mandatory Configuration:"]
        for key in mandatory_configs:
            output.append(key)
        output.append("")
        output.append("Optional Configuration:")
        for key in optional_configs:
            output.append(key)
        output=os.linesep.join(output)
        if print_info:
            print(output)
        return output
   
    def get_config_description(self,key) -> str:
        """
        Retrieves the description of a specific configuration item.
        
        Args:
            key (str): The name of the configuration item.
            
        Returns:
            str: A string containing the description of the configuration item.
        """
        text=str(key)
        texts=[]
        if self._configs_feature[key]["value_type"] is not None:
            texts.append(str(self._configs_feature[key]["value_type"].__name__))
        if self._configs_feature[key]["options"] is not None:
            texts.append("possible options: "+str(self._configs_feature[key]["options"]))
        if self._configs_feature[key]["default_value"] is not None:
            texts.append("default value: "+str(self._configs_feature[key]["default_value"]))
        if len(texts)>0:
            text+=" ("+", ".join(texts)+")"
        text+=": "
        text+=str(self._configs_feature[key]["description"])
        return text
   
    def info_current_configs(self,print_info=True,sort=True) -> str:
        """
        Shows the current configuration items and their values.
        
        Args:
            print_info (bool, optional): Indicates whether to print the information. Defaults to True.
            sort (bool, optional): Indicates whether to sort the dictionary alphabetically. Defaults to True.
            
        Returns:
            str: A string containing the information.
        """
        output=[]
        target_dict=self._sorted_dict(self.configs.to_dict()) if sort else self.configs.to_dict()
        for key,value in target_dict.items():
            output.append("{}: {}".format(key,value))
        output=os.linesep.join(output)
        if print_info:
            print(output)
        return output

    def _sorted_dict(self,target_dict:Dict):
        return dict(sorted(target_dict.items(), key=lambda x: x[0].lower()))


class GroupedConfigurationsHandler(ConfigurationsHandler):
    
    def add_config_item(self,
                        name:str,
                        default_value:Optional[Any]=None,
                        default_value_func:Optional[Callable]=None,
                        mandatory:bool=False,
                        description:str="",
                        value_type=None,
                        options:Optional[Sequence]=None,
                        in_func:Optional[Callable]=None,
                        out_func:Optional[Callable]=None,
                        group:str="default"):
        super().add_config_item(name,
                                default_value=default_value,
                                default_value_func=default_value_func,
                                mandatory=mandatory,
                                description=description,
                                value_type=value_type,
                                options=options,
                                in_func=in_func,
                                out_func=out_func)
        self._configs_feature[name]["group"]=group

    def get_config_description(self,key) -> str:
        """
        Retrieves the description of a specific configuration item.
        
        Args:
            key (str): The name of the configuration item.
            
        Returns:
            str: A string containing the description of the configuration item.
        """
        text=str(key)
        texts=[]
        if self._configs_feature[key]["value_type"] is not None:
            texts.append(str(self._configs_feature[key]["value_type"].__name__))
        if self._configs_feature[key]["options"] is not None:
            texts.append("possible options: "+str(self._configs_feature[key]["options"]))
        if self._configs_feature[key]["default_value"] is not None:
            texts.append("default value: "+str(self._configs_feature[key]["default_value"]))
        texts.append("group: "+str(self._configs_feature[key]["group"]))
        if len(texts)>0:
            text+=" ("+", ".join(texts)+")"
        text+=": "
        text+=str(self._configs_feature[key]["description"])
        return text

    def to_yaml_group(self,only_optional=False,with_description=True):
        output_dict=self.str_dict(only_optional,sort=True)
        yaml_group={}
        for key,value in self._configs_feature.items():
            if value["group"] not in yaml_group.keys():
                yaml_group[value["group"]]=""
            yaml_str=yaml.dump({key:output_dict[key]})
            if with_description:
                yaml_group[value["group"]]+="# "+self.get_config_description(key)+os.linesep+yaml_str+os.linesep
            else:
                yaml_group[value["group"]]+=yaml_str
        return yaml_group

    def save_configs_to_yaml(self,
                             yaml_path_dir:str,
                             only_optional=False,
                             with_description=True,):
        """
        Saves the values of configuration items to a YAML file.

        Args:
            yaml_path_dir (str,): The path to the YAML file. If the path is a directory, the configurations will be saved to multiple files based on the groups. 
            only_optional (bool, optional): Indicates whether to save only the optional configuration items. Defaults to False.
        """
        if os.path.isdir(yaml_path_dir):
            yaml_group=self.to_yaml_group(only_optional,with_description)
            for group_name in yaml_group.keys():
                with open(os.path.join(yaml_path_dir,group_name+".yaml"),"w") as f:
                    f.write(yaml_group[group_name])
        else:
            super().save_configs_to_yaml(yaml_path_dir,only_optional,with_description)
            
    def read_configs_from_yaml(self,yaml_path_dir:str):
        """
        Sets the values of configuration items from a YAML file.

        Args:
            yaml_path_dir (str): The path to the YAML file. If the path is a directory, the configurations will be read from multiple files based on the groups.
        """
        if os.path.isdir(yaml_path_dir):
            paths=[os.path.join(yaml_path_dir,group) for group in os.listdir(yaml_path_dir)]
        else:
            paths=[yaml_path_dir]
        for yaml_path in paths:
            with open(yaml_path,"r") as f:
                yaml_configs=yaml.safe_load(f)
            self.set_config_items(**yaml_configs)
'''
class GroupedConfigurationsHandler():
    
    def __init__(self) -> None:
        self._config_handlers:dict
        self._config_changed=True
        self._grouped_configs=None
        self._configs=None
    
    def find_group(self,key,group:Optional[str]=None):
        if group is None:
            for group in self._config_handlers.keys():
                if key in self._config_handlers[group]._configs_feature.keys():
                    return group
            raise Exception("{} is not a supported configuration.".format(key))
        else:
            if group not in self._config_handlers.keys():
                raise Exception("{} is not a supported group.".format(group))
            return group
      
    def add_config_item(self,
                        name:str,
                        group:str="default",
                        default_value:Optional[Any]=None,
                        default_value_func:Optional[Callable]=None,
                        mandatory:bool=False,
                        description:str="",
                        value_type=None,
                        options:Optional[Sequence]=None,
                        in_func:Optional[Callable]=None,
                        out_func:Optional[Callable]=None):
        if not hasattr(self,"_config_handlers"):
            self._config_handlers={}
        if group not in self._config_handlers.keys():
            self._config_handlers[group]=ConfigurationsHandler()
        self._config_handlers[group].add_config_item(name,
                                                      default_value=default_value,
                                                      default_value_func=default_value_func,
                                                      mandatory=mandatory,
                                                      description=description,
                                                      value_type=value_type,
                                                      options=options,
                                                      in_func=in_func,
                                                      out_func=out_func)
        self._config_changed=True
    
    def get_config_features(self,key,group:Optional[str]=None):
        return self._config_handlers[self.find_group(key,group)].get_config_features(key)
    
    def set_config_features(self,key,group:Optional[str]=None,**feature):
        self._config_handlers[self.find_group(key,group)].set_config_features(key,**feature)
        self._config_changed=True
    
    def set_config_items(self,group:Optional[str]=None,**kwargs): 
        for key in kwargs.keys():
            self._config_handlers[self.find_group(key,group)].set_config_items(**{key:kwargs[key]})
        self._config_changed=True
    
    @property
    def grouped_configs(self):
        if self._config_changed or self._grouped_configs is None:
            output={}
            for group in self._config_handlers.keys():
                output[group]=self._config_handlers[group].configs
            self._grouped_configs=GeneralDataClass(output)
        self._config_changed=False
        return self._grouped_configs
    
    @property
    def configs(self):
        if self._config_changed or self._configs is None:
            output={}
            for group in self._config_handlers.keys():
                output.update(self._config_handlers[group].configs.to_dict())
            self._configs=GeneralDataClass(output)
        self._config_changed=False
        return self._configs
        
    def read_configs_from_yaml(self,yaml_file_path_dir:str):
        if os.path.isdir(yaml_file_path_dir):
            for group in os.listdir(yaml_file_path_dir):
                yaml_file=os.path.join(yaml_file_path_dir,group)
                self._config_handlers[group.split(".")[0]].read_configs_from_yaml(yaml_file)
        else:
            with open(yaml_file,"r") as f:
                yaml_configs=yaml.safe_load(f)
            self.set_config_items(**yaml_configs)
    
    def to_yaml(self,only_optional=False,with_description=True):
        yam_str=""
        for group,current_handler in self._config_handlers.items():
            yam_str+=current_handler.to_yaml(only_optional,with_description)
        return yam_str
    
    def save_configs_to_yaml(self,
                             yaml_file_path_dir:str,
                             only_optional=False,
                             with_description=True,):
        if os.path.isdir(yaml_file_path_dir):
            for group in self._config_handlers.keys():
                yaml_file=os.path.join(yaml_file_path_dir,group+".yaml")
                self._config_handlers[group].save_configs_to_yaml(yaml_file,only_optional,with_description)
        else:
            with open(yaml_file,"w") as f:
                    f.write(self.to_yaml(only_optional,with_description))
    
    def str_dict(self,
                 only_optional=False,
                 sort=True,
                 grouped=False) -> dict:
        output_dict={}
        if grouped:
            for group in self._config_handlers.keys():
                output_dict[group]=self._config_handlers[group].str_dict(only_optional,sort)
        else:
            for group in self._config_handlers.keys():
                for key, value in self._config_handlers[group].str_dict(only_optional,sort):
                    output_dict[key]=value
     
    def info_available_configs(self,print_info=True,sort=True) -> str:
        mandatory_configs={}
        optional_configs={}
        for group,current_handler in self._config_handlers.items():
            target_dict=current_handler._sorted_dict(current_handler._configs_feature) if sort else current_handler._configs_feature
            for key in target_dict.keys():
                texts="        "+current_handler.get_config_description(key)
                if current_handler._configs_feature[key]["mandatory"]:
                    if group not in mandatory_configs.keys():
                        mandatory_configs[group]=[]
                    mandatory_configs[group].append(texts)
                else:
                    if group not in optional_configs.keys():
                        optional_configs[group]=[]
                    optional_configs[group].append(texts)
        output=["Mandatory Configuration:"]
        for group in mandatory_configs.keys():
            output.append("    "+group+":")
            output+=mandatory_configs[group]
        output.append("")
        output.append("Optional Configuration:")
        for group in optional_configs.keys():
            output.append("    "+group+":")
            output+=optional_configs[group]
        print(output)
        output=os.linesep.join(output)
        if print_info:
            print(output)
        return output
    
    def get_config_description(self,key,group:Optional[str]=None):
        return self._config_handlers[self.find_group(key,group)].get_config_description(key)
    
    def info_current_configs(self,print_info=True,sort=True) -> str:
        output=[]
        for group,current_handler in self._config_handlers.items():
            output.append("Group: "+group)
            output.append(current_handler.info_current_configs(print_info=False,sort=sort))
            output.append("")
        output=os.linesep.join(output)
        if print_info:
            print(output)
        return output
'''  
        
    
def set_random_seed(random_seed):
        """
        Set the random seed for various libraries.

        Args:
                random_seed (int): The random seed value to set.

        Returns:
                None
        """
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

def get_random_state():
        """
        Get the random state of various libraries.

        Returns:
                dict: A dictionary containing the random state of various libraries.
        """
        random_state={
                "torch":torch.get_rng_state(),
                "torch_cuda":torch.cuda.get_rng_state(),
                "numpy":np.random.get_state(),
                "random":random.getstate()
        }
        return random_state

def set_random_state(random_state):
        """
        Set the random state for various libraries.

        Args:
                random_state (dict): A dictionary containing the random state of various libraries.

        Returns:
                None
        """
        torch.set_rng_state(random_state["torch"])
        torch.cuda.set_rng_state(random_state["torch_cuda"])
        np.random.set_state(random_state["numpy"])
        random.setstate(random_state["random"])
