# Copyright (c) 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class Registry:
    def __init__(self, name):
        self.name = name
        self.registry_dict = {}

    def register_module(self, module_name):
        def decorator(cls):
            self.registry_dict[module_name] = cls
            return cls
        return decorator
    
register_model = Registry("model")
register_backbone = Registry("backbone")
register_pooling = Registry("pooling")
register_embedding = Registry("embedding")
register_loss = Registry("loss")
register_trainer = Registry("trainer")
register_scheduler = Registry("scheduler")


# 创建/加载 各类组件

def create_module(register, cfg_dict):
    module_name = cfg_dict["type"]
    if module_name not in register.registry_dict:
        raise ValueError(f"Invalid module name: {module_name}")
    args = cfg_dict.copy()
    args.pop('type')
    cls = register.registry_dict[module_name]
    return cls(**args)

def create_model(config, register=register_model): 
    return create_module(register_model, config)

def create_backbone(config, register=register_backbone): 
    return create_module(register_backbone, config)

def create_pooling(config, register=register_pooling): 
    return create_module(register_pooling, config)

def create_embedding(config, register=register_embedding): 
    return create_module(register_embedding, config)

def create_loss(config, register=register_loss): 
    return create_module(register_loss, config)

def create_trainer(config, register=register_trainer): 
    return create_module(register_trainer, config)

def create_scheduler(optimizer, config, register=register_scheduler):
    module_name = config["type"]
    if module_name not in register.registry_dict:
        raise ValueError(f"Invalid module name: {type}")
    args = config.copy()
    args.pop('type')
    cls = register.registry_dict[module_name]
    return cls(optimizer, **args["config"])