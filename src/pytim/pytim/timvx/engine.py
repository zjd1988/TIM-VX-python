# -*- coding: utf-8 -*-
import copy
import json
import numpy as np
from .lib.pytimvx import *

TimVxDataType = ["INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FLOAT16", "FLOAT32", "BOOL8"]

class Engine():
    def __init__(self, name:str):
        self.engine = TimVXEngine(name)
        self.mean_value = {}
        self.std_value = {}
        self.reorder = {}
        self.inputs_info = {}
        self.outputs_info = {}
        self.nodes_info = []
        self.tensors_info = []
        self.inputs_alias = {}
        self.outputs_alias = {}


    def set_mean_value(self, input_name:str, mean_value:list):
        self.mean_value[input_name] = mean_value


    def set_std_value(self, input_name:str, std_value:list):
        self.std_value[input_name] = std_value


    def set_reorder(self, input_name:str, reorder:list):
        if reorder != [0, 1, 2] and reorder != [2, 1, 0]:
            assert False, "invaid channel reorder {}".format(reorder)
        self.reorder[input_name] = reorder


    def add_inputs_info(self, tensor_info:dict):
        input_name = tensor_info["name"]
        assert input_name not in self.inputs_info, "tensor {} already exists!".format(input_name)
        self.inputs_info[input_name] = tensor_info
        if "alias" in tensor_info.keys():
            alias_name = tensor_info["alias"]
            self.inputs_alias[alias_name] = input_name
        print("add input tensor {}:\n{}".format(input_name, tensor_info))


    def add_outputs_info(self, tensor_info:dict):
        output_name = tensor_info["name"]
        assert output_name not in self.outputs_info, "tensor {} already exists!".format(output_name)
        self.outputs_info[output_name] = tensor_info
        if "alias" in tensor_info.keys():
            alias_name = tensor_info["alias"]
            self.outputs_alias[alias_name] = output_name
        print("add output tensor {}:\n{}".format(output_name, tensor_info))


    def add_nodes_info(self, node_info:dict):
        self.nodes_info.append(node_info)


    def add_tensors_info(self, tensor_info:dict):
        tensor_name = tensor_info["name"]
        self.tensors_info.append(tensor_info)
        print("add tensor {}:\n{}".format(tensor_name, tensor_info))


    def convert_np_dtype_to_tim_dtype(self, datatype):
        if datatype == np.int8:
            return "INT8"
        elif datatype == np.uint8:
            return "UINT8"
        elif datatype == np.int16:
            return "INT16"
        elif datatype == np.uint16:
            return "UINT16"
        elif datatype == np.int32:
            return "INT32"
        elif datatype == np.uint32:
            return "UINT32"
        elif datatype == np.float16:
            return "FLOAT16"
        elif datatype == np.float32:
            return "FLOAT32"
        elif datatype == np.bool:
            return "BOOL8"
        else:
            assert False, "unspoorted datatype {}, when convert np type to tim type".format(datatype)


    def convert_tim_dtype_to_np_dtype(self, datatype:str):
        if datatype == "INT8":
            return np.int8
        elif datatype == "UINT8":
            return np.uint8
        elif datatype == "INT16":
            return np.int16
        elif datatype == "UINT16":
            return np.uint16
        elif datatype == "INT32":
            return np.int32
        elif datatype == "UINT32":
            return np.uint32
        elif datatype == "FLOAT16":
            return np.float16
        elif datatype == "FLOAT32":
            return np.float32
        elif datatype == "BOOL8":
            return np.bool
        else:
            assert False, "unspoorted datatype {}, when convert tim tensor type to np type".format(datatype)


    def get_graph_name(self):
        return get_graph_name(self.engine)


    def get_tensor_size(self, tensor_name:str):
        return get_tensor_size(self.engine, tensor_name)


    def create_tensor(self, tensor_name:str, tensor_dtype:str, tensor_attr:str, \
        tensor_shape:list, alias_name:str="", quant_info:dict={}, np_data:np.array=np.array([])):

        assert tensor_dtype in TimVxDataType, "tim-vx not support {} datatype".format(tensor_dtype)
        tensor_info = {}
        tensor_info["shape"] = tensor_shape
        tensor_info["data_type"] = tensor_dtype
        tensor_info["attribute"] = tensor_attr
        if len(quant_info.keys()) != 0:
            tensor_info["quant_info"] = quant_info
        assert create_tensor(self.engine, tensor_name, tensor_info, np_data), "creat tensor {} fail!".format(tensor_name)
        # add engine tensor_info stat
        tensor_stat_info = {}
        tensor_stat_info["name"] = tensor_name
        tensor_stat_info["dtype"] = self.convert_tim_dtype_to_np_dtype(tensor_dtype)
        tensor_stat_info["attr"] = tensor_attr
        tensor_stat_info["shape"] = tensor_shape
        tensor_stat_info["quant_info"] = quant_info
        if np_data.size != 0:
            tensor_stat_info["data"] = np_data
        if "alias" in tensor_info.keys():
            tensor_stat_info["alias"] = tensor_info["alias"]
        if tensor_attr == "INPUT":
            self.add_inputs_info(tensor_stat_info)
        elif tensor_attr == "OUTPUT":
            self.add_outputs_info(tensor_stat_info)
        else:
            self.add_tensors_info(tensor_stat_info)
        return tensor_stat_info


    def copy_data_from_tensor(self, tensor_name:str, np_data:np.array):
        return copy_data_from_tensor(self.engine, tensor_name, np_data)


    def copy_data_to_tensor(self, tensor_name:str, np_data:np.array):
        return copy_data_to_tensor(self.engine, tensor_name, np_data)


    def create_operation(self, op_info:dict):
        ret = create_operation(self.engine, op_info)
        op_name = op_info["op_name"]
        if ret and "op_inputs" in op_info.keys():
            op_inputs = op_info["op_inputs"]
            ret = bind_inputs(self.engine, op_name, op_inputs)
        if ret and "op_outputs" in op_info.keys():
            op_outputs = op_info["op_outputs"]
            ret = bind_outputs(self.engine, op_name, op_outputs)
        self.add_nodes_info(op_info)
        return ret


    def get_op_info(self, op_name:str):
        return get_op_info(self.engine, op_name)


    def bind_input(self, op_name:str, tensor_name:str):
        return bind_input(self.engine, op_name, tensor_name)


    def bind_output(self, op_name:str, tensor_name:str):
        return bind_output(self.engine, op_name, tensor_name)


    def bind_inputs(self, op_name:str, tensor_names:list):
        return bind_inputs(self.engine, op_name, tensor_names)


    def bind_outputs(self, op_name:str, tensor_names:list):
        return bind_outputs(self.engine, op_name, tensor_names)


    def create_graph(self):
        return create_graph(self.engine)


    def verify_graph(self):
        return verify_graph(self.engine)


    def compile_graph(self):
        return compile_graph(self.engine)


    def compile_to_binary(self):
        return compile_to_binary(self.engine)


    def run_graph(self, input_dict:dict, output_name_list:list=[]):
        for input_name in input_dict.keys():
            input_tensor_name = input_name
            assert input_tensor_name in self.inputs_info.keys() or input_tensor_name in self.inputs_alias.keys(), \
                "invalid input tensor name {}".format(input_tensor_name)
            if input_tensor_name in self.inputs_alias.keys():
                input_tensor_name = self.inputs_alias[input_tensor_name]
            input_data = input_dict[input_tensor_name]
            assert type(input_dict[input_tensor_name]) == np.ndarray, "{} tensor data only support numpy array"
            mean_value = [0.0, ]
            std_value = [1.0, ]
            if input_tensor_name in self.mean_value.keys():
                mean_value = self.mean_value[input_tensor_name]
            if input_tensor_name in self.std_value.keys():
                std_value = self.std_value[input_tensor_name]
            engine_input = (input_data.astype(np.float32) - mean_value) / std_value
            if self.reorder == [2, 1, 0]:
                assert len(input_data.shape) == 3, "need a hwc format input, please check!"
                h,w,c = input_data.shape
                assert c == 3 or c == 1, "input channel should be 1 or 3"
                engine_input = engine_input[:,:,::-1]
                engine_input = engine_input.transpose((2, 0, 1))
            # timvx tensor dims is reverse order with np dims
            shape = copy.deepcopy(self.inputs_info[input_tensor_name]["shape"])
            shape.reverse()
            dtype = self.inputs_info[input_tensor_name]["dtype"]

            scale = 1.0
            zero_point = 0.0
            if "scale" in self.inputs_info[input_tensor_name]["quant_info"]:
                scale = self.inputs_info[input_tensor_name]["quant_info"]["scale"]
            if "zero_point" in self.inputs_info[input_tensor_name]["quant_info"]:
                zero_point = self.inputs_info[input_tensor_name]["quant_info"]["zero_point"]
            engine_input = (engine_input / scale + zero_point).reshape(shape).astype(dtype)
            assert self.copy_data_to_tensor(input_tensor_name, engine_input), \
                "set input {} fail!".format(input_name)

        assert run_graph(self.engine), "run graph fail!"

        outputs = []
        if output_name_list == []:
            if self.outputs_alias != {}:
                output_name_list = list(self.outputs_alias.keys())
            else:
                output_name_list = list(self.outputs_info.keys())
        for output_index in range(len(output_name_list)):
            output_name = output_name_list[output_index]
            output_tensor_name = output_name
            if output_tensor_name in self.outputs_alias.keys():
                output_tensor_name = self.outputs_alias[output_tensor_name]
            assert output_tensor_name in self.outputs_info.keys(), \
                "{} not a valid output tensor".format(output_name)
            # timvx tensor dims is reverse order with np dims
            shape = copy.deepcopy(self.outputs_info[output_tensor_name]["shape"])
            shape.reverse()
            dtype = self.outputs_info[output_tensor_name]["dtype"]

            scale = 1.0
            zero_point = 0.0
            if "scale" in self.outputs_info[output_tensor_name]["quant_info"]:
                scale = self.outputs_info[output_tensor_name]["quant_info"]["scale"]
            if "zero_point" in self.outputs_info[output_tensor_name]["quant_info"]:
                zero_point = self.outputs_info[output_tensor_name]["quant_info"]["zero_point"]
            output_data = np.zeros(shape).astype(dtype)
            assert self.copy_data_from_tensor(output_tensor_name, output_data), \
                "get output {} fail!".format(output_name)
            output_data = output_data.astype(np.float32)
            output_data = (output_data - zero_point) * scale
            output_data = output_data.reshape(shape[::-1])
            outputs.append(output_data)
        return outputs


    def export_graph(self, graph_json_file:str, weight_bin_file:str, log_flag:bool=False):
        graph_json_dict = {}
        # init norm_info
        print("prepare norm ...")
        norm_info = {}
        norm_info["mean"] = self.mean_value
        norm_info["std"] = self.std_value
        norm_info["reorder"] = self.reorder
        graph_json_dict["norm"] = norm_info
        if log_flag:
            print(graph_json_dict["norm"])

        # init inputs_info
        print("prepare inputs ...")
        inputs_info = []
        for input_name in self.inputs_info.keys():
            input_tensor = {}
            tensor_info = self.inputs_info[input_name]
            for item_key in tensor_info.keys():
                item_value = tensor_info[item_key]
                if item_key == "dtype":
                    item_value = self.convert_np_dtype_to_tim_dtype(item_value)
                input_tensor[item_key] = item_value
            if log_flag:
                print(input_tensor)
            inputs_info.append(input_tensor)
        graph_json_dict["inputs"] = inputs_info

        # init outputs_info
        print("prepare outputs ...")
        outputs_info = []
        for output_name in self.outputs_info.keys():
            output_tensor = {}
            tensor_info = self.outputs_info[output_name]
            for item_key in tensor_info.keys():
                item_value = tensor_info[item_key]
                if item_key == "dtype":
                    item_value = self.convert_np_dtype_to_tim_dtype(item_value)
                output_tensor[item_key] = item_value
            if log_flag:
                print(output_tensor)
            outputs_info.append(output_tensor)
        graph_json_dict["outputs"] = outputs_info

        # init tensors_info
        print("prepare tensors ...")
        weight_offset = 0
        weight_bin_list = []
        tensors_info = []
        for index in range(len(self.tensors_info)):
            new_tensor_info = {}
            tensor_info = self.tensors_info[index]
            for item_key in tensor_info.keys():
                item_value = tensor_info[item_key]
                if item_key == "dtype":
                    item_value = self.convert_np_dtype_to_tim_dtype(item_value)
                if item_key == "data":
                    item_value = weight_offset
                    weight_offset += len(tensor_info[item_key].tobytes())
                    weight_bin_list.append(tensor_info[item_key].tobytes())
                    item_key = "offset"
                new_tensor_info[item_key] = item_value
            if log_flag:
                print(new_tensor_info)
            tensors_info.append(new_tensor_info)
        graph_json_dict["tensors"] = tensors_info

        # init nodes_info/inputs_alias/outputs_alias
        print("prepare nodes/inputs_alias/outputs_alias ...")
        graph_json_dict["nodes"] = self.nodes_info
        graph_json_dict["inputs_alias"] = self.inputs_alias
        graph_json_dict["outputs_alias"] = self.outputs_alias
        if log_flag:
            print(graph_json_dict["nodes"])
            print(graph_json_dict["inputs_alias"])
            print(graph_json_dict["outputs_alias"])
        # dump to json file/bin file
        print("write to file ...")
        graph_json_obj = json.dumps(graph_json_dict, indent=4)
        with open(graph_json_file, "w") as f:
            f.write(graph_json_obj)

        with open(weight_bin_file, "wb") as f:
            for index in range(len(weight_bin_list)):
                f.write(weight_bin_list[index])

        print("export success.")
        return True