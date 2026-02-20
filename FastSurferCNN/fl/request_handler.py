from __future__ import annotations

import os
import pickle
from typing import Dict, Literal
from yacs.config import CfgNode 
from os.path import join
import json


from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from di.config_loader import ConfigLoader, FLConfigLoader
from di.container import create_injector
from FastSurferCNN.train import Trainer
from FastSurferCNN.fl.task import FastSurferCNN_FL_Trainer
from fl.api import FederatedClientAPI, FederatedOrchestrator
from fl.backends import FederatedBackend
from typing import Callable

client_app = ClientApp()
injector = create_injector()
   

commands = Literal["train", "evaluate", "sync", "aggregate"]

class CommandDispatcher:
    def __init__(self):
        self._handlers: Dict[str, Callable[[Message, Context], Message]] = {}

    def register(self, command: str, handler: Callable[[Message, Context], Message]) -> None:
        """
        Registers a handler function for a specific command.
        
        Parameters
        -------
            command: str
                The command string to register the handler for. 
            handler: Callable[[Message, Context]
                A function that takes a Message and Context as input and returns a Message as output.
        """
        self._handlers[command] = handler

    def dispatch(self, msg: Message, context: Context) -> Message:
        """
        Dispatches the incoming message to the appropriate handler based on the command specified in the message content.

        Parameters
        -------
            msg:  Message
                The incoming message containing the command and any associated data. 
            context: Context
                The context in which the message is being processed, providing access to client state and configuration. Returns: - A Message object containing the response from the handler function, or an error message if no handler is found for the specified command.
        
        Returns
        ------- 
            message: Message
                The response message from the handler function, or an error message if no handler is found for the specified command.
        
        """
        
        cmd: commands = msg.content.get("cmd", "unknown")  
        handler = self._handlers.get(cmd)
        return handler(msg, context)

################# REQUEST HANDLERS ##################

def request_handler(msg: Message, context: Context) -> Message:
    command_handler = CommandDispatcher()
    command_handler.register("train", handle_train_request)
    command_handler.register("evaluate", handle_evaluate_request)
    command_handler.register("sync", handle_sync_request)
    command_handler.register("inference", handle_inference_request)
    command_handler.register("unknown", handle_unknown_request)
    return command_handler.dispatch(msg, context)

def handle_train_request(msg: Message, context: Context) -> Message:
    # Unpack config sent by server
    cfg, model_name, plane = unpack_server_config(msg.content["config"])
    trainer = injector.get(FastSurferCNN_FL_Trainer)
    # Start training loop
    last_loss, metrics, num_examples, model_state = trainer.fit()
    metric_records: dict[str, float] = {
        f"train_loss_{model_name}_{plane}": float(last_loss),
        f"num_examples_{model_name}_{plane}": float(len(num_examples)),
    }
    for metric_name, metric_value in metrics.items():
        metric_records[f"{metric_name}_{model_name}_{plane}"] = float(metric_value)
    return _build_result_message(weights=model_state, metric_records=metric_records, reply_to=msg)

def handle_evaluate_request(msg: Message, context: Context) -> Message:
    # Unpack config sent by server
    cfg, model_name, plane = unpack_server_config(msg.content["config"])
    trainer = injector.get(FastSurferCNN_FL_Trainer)
    val_loss, metrics, num_examples = trainer.evaluate(epoch=0)
    metric_records: dict[str, float] = {
        f"val_loss_{model_name}_{plane}": float(val_loss),
        f"num_examples_{model_name}_{plane}": float(len(num_examples)),
    }
    for metric_name, metric_value in metrics.items():
        metric_records[f"{metric_name}_{model_name}_{plane}"] = float(metric_value)
    return _build_result_message(model=trainer.model, metric_records=metric_records, reply_to=msg)

def handle_sync_request(msg: Message, context: Context) -> Message:
    incoming_weights = msg.content["weights"].to_torch_state_dict()
    federated_client = injector.get(FederatedClientAPI)
    federated_client.update_internal_state(new_state=incoming_weights, round_idx=0)
    # TODO: acknowledge sync request and report sync status to message queue

def handle_inference_request(msg: Message, context: Context) -> Message:
    pass

def handle_unknown_request(msg: Message, context: Context) -> Message:
    pass


################# UTILS ##################

def unpack_server_config(cfg_content: ConfigRecord) -> tuple[CfgNode, str, str]:
    # Unpack config sent by server
    federated_config = cfg_content
    train_params = json.loads(federated_config["TRAIN"])
    data_params = json.loads(federated_config["DATA"])
    
    # overwrite local config entries with incoming federated config values
    fl_cfg = injector.get(FLConfigLoader).load_config()
    fl_cfg.merge_from_other_cfg(CfgNode(train_params))
    fl_cfg.merge_from_other_cfg(CfgNode(data_params)) 

    plane = fl_cfg.DATA.PLANE.lower()
    model_name = fl_cfg.MODEL.NAME
    cfg_path = os.path.join(f"{fl_cfg.LOG_DIR}", "config", f"{model_name}_{plane}.yaml")
    cfg = injector.get(ConfigLoader).load_from_file(cfg_path)
    cfg.merge_from_other_cfg(fl_cfg) 
    
    return cfg, model_name, plane

def _build_result_message(*, weights, metric_records: dict[str, float], reply_to: Message) -> Message:
    arrays = ArrayRecord(array_dict=weights)
    metrics = MetricRecord(metric_records)
    content = RecordDict(records={"weights": arrays, "metrics": metrics})
    return Message(content=content, reply_to=reply_to)