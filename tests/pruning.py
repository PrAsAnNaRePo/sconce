import torch
from torch import nn
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from utils import get_input_channel_importance

class Pruning:
    def __init__(self, prune_mode, layer_idx) -> None:
        self.prune_mode = prune_mode
        self.layer_idx = layer_idx

    @torch.no_grad()
    def apply_channel_sorting(self, model):
        """
        Applies channel sorting to the model's convolutional and batch normalization layers.
        Returns a copy of the model with sorted channels.

        Returns:
        model (torch.nn.Module): A copy of the model with sorted channels.
        """
        
        # model = copy.deepcopy(self.model)  # do not modify the original model
        # fetch all the conv and bn layers from the backbone
        
        all_convs = []
        all_bns = []
        
        # Universal Layer Seeking by Parsing
        def find_instance(obj, object_of_importance):
            if isinstance(obj, object_of_importance):
                if object_of_importance == nn.Conv2d:
                    all_convs.append(obj)
                elif object_of_importance == nn.BatchNorm2d:
                    all_bns.append(obj)
                return None
            elif isinstance(obj, list):
                for internal_obj in obj:
                    find_instance(internal_obj, object_of_importance)
            elif hasattr(obj, "__class__"):
                for internal_obj in obj.children():
                    find_instance(internal_obj, object_of_importance)
            elif isinstance(obj, OrderedDict):
                for key, value in obj.items():
                    find_instance(value, object_of_importance)
        
        find_instance(obj=model, object_of_importance=nn.Conv2d)
        find_instance(obj=model, object_of_importance=nn.BatchNorm2d)
        
        # iterate through conv layers
        for i_conv in range(len(all_convs) - 1):
            # each channel sorting index, we need to apply it to:
            # - the output dimension of the previous conv
            # - the previous BN layer
            # - the input dimension of the next conv (we compute importance here)
            prev_conv = all_convs[i_conv]
            prev_bn = all_bns[i_conv]
            next_conv = all_convs[i_conv + 1]
            # note that we always compute the importance according to input channels
            importance = get_input_channel_importance(next_conv.weight)
            # sorting from large to small
            sort_idx = torch.argsort(importance, descending=True)
            
            # apply to previous conv and its following bn
            prev_conv.weight.copy_(
                torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
            )
            for tensor_name in ["weight", "bias", "running_mean", "running_var"]:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                tensor_to_apply.copy_(
                    torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                )
            
            # apply to the next conv input (hint: one line of code)
            
            next_conv.weight.copy_(
                torch.index_select(next_conv.weight.detach(), 1, sort_idx)
            )
        
        return model

    @torch.no_grad()
    def sensitivity_scan(
            self,
            model,
            dense_model_accuracy,
            scan_step=0.05,
            scan_start=0.1,
            scan_end=1.0,
            verbose=True,
    ):
        """
        Scans the sensitivity of the model to weight pruning by gradually increasing the sparsity of each layer's weights
        and measuring the resulting accuracy. Returns a dictionary mapping layer names to the sparsity values that resulted
        in the highest accuracy for each layer.

        :parameter dense_model_accuracy: the accuracy of the original dense model
        :parameter scan_step: the step size for the sparsity scan
        :parameter scan_start: the starting sparsity for the scan
        :parameter scan_end: the ending sparsity for the scan
        :parameter verbose: whether to print progress information during the scan
        :return: a dictionary mapping layer names to the sparsity values that resulted in the highest accuracy for each layer
        """
        
        self.sparsity_dict = {}
        sparsities = np.flip(np.arange(start=scan_start, stop=scan_end, step=scan_step))
        accuracies = []
        named_all_weights = [
            (name, param)
            for (name, param) in model.named_parameters()
            if param.dim() > 1
        ]
        named_conv_weights = [
            (name, param)
            for (name, param) in model.named_parameters()
            if param.dim() > 2
        ]
        param_names = [i[0] for i in named_all_weights]
        original_model = copy.deepcopy(model)
        # original_dense_model_accuracy = self.evaluate()
        conv_layers = [
            module for module in model.modules() if (isinstance(module, nn.Conv2d))
        ]
        linear_layers = [
            module for module in model.modules() if (isinstance(module, nn.Linear))
        ]
        
        if self.prune_mode == "CWP":
            sortd = self.apply_channel_sorting()
            sorted_model = copy.deepcopy(sortd)
        
        if "venum" in self.prune_mode:
            if self.prune_mode == "venum-cwp":
                named_all_weights = named_conv_weights
            
            list_of_sparsities = [0] * (len(named_all_weights) - 1)
            sparsity_dict = {count: ele for count, ele in enumerate(list_of_sparsities)}
            self.venum_apply(sparsity_dict)
        
        layer_iter = tqdm(named_all_weights, desc="layer", leave=False)
        original_prune_mode = self.prune_mode
        for i_layer, (name, param) in enumerate(layer_iter):
            param_clone = param.detach().clone()
            accuracy = []
            desc = None
            if verbose:
                desc = f"scanning {i_layer}/{len(named_all_weights)} weight - {name}"
                picker = tqdm(sparsities, desc)
            else:
                picker = sparsities
            hit_flag = False
            
            for sparsity in picker:
                if (
                        "venum" in self.prune_mode
                        and len(param.shape) > 2
                        and i_layer < (len(conv_layers) - 1)
                ):
                    # self.temp_sparsity_list[i_layer] = sparsity
                    self.layer_idx = i_layer
                    self.prune_mode = original_prune_mode
                    
                    list_of_sparsities = [0] * (len(layer_iter) - 1)
                    list_of_sparsities[i_layer] = sparsity
                    sparsity_dict = {
                        count: ele for count, ele in enumerate(list_of_sparsities)
                    }
                    if self.prune_mode == "venum-cwp":
                        self.venum_CWP_Pruning(original_model, sparsity_dict)
                    else:
                        self.venum_apply(sparsity_dict)
                    
                    hit_flag = True
                elif self.prune_mode == "GMP":
                    sparse_list = np.zeros(len(named_all_weights))
                    sparse_list[i_layer] = sparsity
                    local_sparsity_dict = dict(zip(param_names, sparse_list))
                    self.GMP_Pruning(
                        prune_dict=local_sparsity_dict
                    )  # FineGrained Pruning
                    self.callbacks = [lambda: self.GMP_apply()]
                    hit_flag = True
                
                elif (
                        self.prune_mode == "CWP"
                        and len(param.shape) > 2
                        and i_layer < (len(conv_layers) - 1)
                ):
                    # self.model = sorted_model
                    model = self.channel_prune_layerwise(
                        sorted_model, sparsity, i_layer
                    )
                    hit_flag = True
                ## TODO:
                ## Add conv CWP and linear CWP
                
                if hit_flag == True:
                    # if self.prune_mode == "venum_sensitivity":
                    #     self.prune_mode = original_prune_mode
                    acc = self.preformance_eval.evaluate(model, self.dataloader, device=self.device, Tqdm=False) - dense_model_accuracy
                    # if ("venum" in self.prune_mode):
                    #     self.prune_mode = "venum_sensitivity"
                    model = copy.deepcopy(original_model)
                    if abs(acc) <= self.degradation_value:
                        self.sparsity_dict[name] = sparsity
                        model = copy.deepcopy(original_model)
                        break
                    elif sparsity == scan_start:
                        accuracy = np.asarray(accuracy)
                        
                        if np.max(accuracy) > -0.75:  # Allowed Degradation
                            acc_x = np.where(accuracy == np.max(accuracy))[0][0]
                            best_possible_sparsity = sparsities[acc_x]
                        
                        else:
                            best_possible_sparsity = 0
                        self.sparsity_dict[name] = best_possible_sparsity
                        model = copy.deepcopy(original_model)
                    else:
                        accuracy.append(acc)
                        hit_flag = False
                    model = copy.deepcopy(original_model)