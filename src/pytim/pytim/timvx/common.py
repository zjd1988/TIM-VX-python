# -*- coding: utf-8 -*-
import math
import numpy as np
from .lib.pytimvx import *
PadType = ["NONE", "AUTO", "VALID", "SAME"]
PoolType = ["MAX", "AVG", "L2", "AVG_ANDROID"]
RoundType = ["CEILING", "FLOOR"]
OverflowPolicy = ["WRAP", "SATURATE"]
RoundingPolicy = ["TO_ZERO", "RTNE"]
ResizeType = ["NEAREST_NEIGHBOR", "BILINEAR", "AREA"]
DataLayout = [ "ANY", "WHCN", "CWHN", "IcWHOc", "OcIcWH", "IcOcWH", "WHIcOc", "WCN", "WIcOc"]
TimVxDataType = ["INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FLOAT16", "FLOAT32", "BOOL8"]
QuantType = ["NONE", "ASYMMETRIC", "SYMMETRIC_PER_CHANNEL"]


def setLogLevel(log_level:str="DEBUG"):
    LOG_LEVEL_MAP = {"TRACE" : 0,
                     "DEBUG" : 1,
                     "INFO"  : 2,
                     "WARN"  : 3,
                     "ERROR" : 4}
    return set_log_level(LOG_LEVEL_MAP[log_level])


def quantizationParams(f_min:float, f_max:float, data_type:type):
    type_info = np.iinfo(data_type)
    zero_point = 0
    scale = 0
    qmin = type_info.min
    qmax = type_info.max
    qmin_double = float(qmin)
    qmax_double = float(qmax)
    #   // 0 should always be a representable value. Let's assume that the initial
    #   // min,max range contains 0.
    if f_min == f_max:
        # // Special case where the min,max range is a point. Should be {0}.
        return scale, zero_point


    #   // General case.
    #   //
    #   // First determine the scale.
    scale = (f_max - f_min) / (qmax_double - qmin_double)

    #   // Zero-point computation.
    #   // First the initial floating-point computation. The zero-point can be
    #   // determined from solving an affine equation for any known pair
    #   // (real value, corresponding quantized value).
    #   // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    #   // The arithmetic error on the zero point computed from either pair
    #   // will be roughly machine_epsilon * (sum of absolute values of terms)
    #   // so we want to use the variant that adds the smaller terms.
    zero_point_from_min = qmin_double - f_min / scale
    zero_point_from_max = qmax_double - f_max / scale

    zero_point_from_min_error = abs(qmin_double) + abs(f_min / scale)

    zero_point_from_max_error = abs(qmax_double) + abs(f_max / scale)

    zero_point_double = zero_point_from_min if zero_point_from_min_error < zero_point_from_max_error else zero_point_from_max

    #   // Now we need to nudge the zero point to be an integer
    #   // (our zero points are integer, and this is motivated by the requirement
    #   // to be able to represent the real value "0" exactly as a quantized value,
    #   // which is required in multiple places, for example in Im2col with SAME
    #   //  padding).

    nudged_zero_point = 0
    if zero_point_double < qmin_double:
        nudged_zero_point = qmin
    elif zero_point_double > qmax_double:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = round(zero_point_double)

    #   // The zero point should always be in the range of quantized value,
    #   // // [qmin, qmax].

    zero_point = nudged_zero_point
    #   // finally, return the values
    return scale, zero_point


def quantize(data:'list|np.array', scale:float, zero_point:int, dest_type:type)->list:
    type_info = np.iinfo(dest_type)
    min_value = type_info.min
    max_value = type_info.max
    if list == type(data):
        np_array = np.array(data)
    else:
        np_array = data
    np_array_q = np.round((np_array / scale) + zero_point)
    np_array_q[np_array_q > max_value] = max_value
    np_array_q[np_array_q < min_value] = min_value
    return np_array_q.astype(dest_type)


def dequantize(data:'list|np.array', scale:float, zero_point:int)->np.array:
    if list == type(data):
        np_array = np.array(data)
    else:
        np_array = data

    return ((np_array - zero_point) * scale).astype(np.float32)


class Quantization():
    def __init__(self, scale:'int|list', zp:'int|list', quant_type:str="NONE", channel_dim:int=-1):
        self.type = quant_type
        self.channel_dim = channel_dim
        self.scales = list(scale)
        self.zero_points = list(zp)


    def type(self)->str:
        return type


    def setType(self, type:str)->None:
        if type not in QuantType:
            print("")
        else:
            self.type = type
    

    def channelDim(self)->int:
        return self.channel_dim

    
    def setChannelDim(self, channel_dim:int)->None:
        self.channel_dim = channel_dim


    def scales(self)->list:
        return self.scales

    
    def setScales(self, scales:list)->None:
        self.scales = scales


    def zeroPoints(self)->list:
        return self.zero_points

    
    def setZeroPoints(self, zps:list)->None:
        self.zero_points = zps


def ConstructConv1dOpConfig(op_name:str, stride:int, dilation:int, ksize:int=0, padding:str="AUTO", 
    pad:list=[0, 0], weights:int=0, multiplier:int=0, kernel_layout:str="WHIcOc", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Conv1d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["weights"] = weights
    op_attr["multiplier"] = multiplier
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructConv2dOpConfig(op_name:str, stride:list, dilation:list, ksize:list=[0, 0], padding:str="AUTO", 
    pad:list=[0, 0, 0, 0], weights:int=0, multiplier:int=0, input_layout:str="WHCN", 
    kernel_layout:str="WHIcOc", op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert input_layout in DataLayout, "input_layout:{} is not in {}".format(input_layout, DataLayout)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Conv2d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["weights"] = weights
    op_attr["multiplier"] = multiplier
    op_attr["input_layout"] = input_layout
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructGroupedConv2dOpConfig(op_name:str, stride:list, dilation:list, grouped_number:int, padding:str="AUTO", 
    pad:list=[0, 0, 0, 0], input_layout:str="WHCN", kernel_layout:str="WHIcOc", 
    op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert input_layout in DataLayout, "input_layout:{} is not in {}".format(input_layout, DataLayout)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "GroupedConv2d"
    op_attr = {}
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["grouped_number"] = grouped_number
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["input_layout"] = input_layout
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructActivationOpConfig(op_name:str, activation_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # 1 prelu parameter
    # axis = None
    # 2 leakyrelu parameter
    # ratio = None
    # 3 linear parameter
    # a = None b = 0.0
    # 4 gelu parameter
    # approximate = True 
    valid_act_type = ["Relu", "Relu1", "Relu6", "Elu", "Sigmoid", "Mish", "HardSigmoid",
        "SoftRelu", "HardSwish", "Swish", "Prelu", "Tanh", "LeakyRelu", "Linear", "Gelu"]
    assert activation_type in valid_act_type, "activation_type:{} is not in {}".format(activation_type, valid_act_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Activation"
    op_attr = {}
    op_attr["activation_type"] = activation_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructEltwiseOpConfig(op_name:str, eltwise_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # Multiply/Div parameter
    # scale = 1.0
    valid_eltwise_type = ["Minimum", "Maximum", "Add", "Sub", "Pow", "FloorDiv", "Multiply", "Div"]
    assert eltwise_type in valid_eltwise_type, "eltwise_type:{} is not in {}".format(eltwise_type, valid_eltwise_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Eltwise"
    op_attr = {}
    op_attr["eltwise_type"] = eltwise_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    
    return op_info_dict

def ConstructReshapeOpConfig(op_name:str, size:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Reshape"
    op_attr = {}
    op_attr["size"] = size
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructTransposeOpConfig(op_name:str, perm:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Transpose"
    op_attr = {}
    op_attr["perm"] = perm
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict

def ConstructFullyConnectedOpConfig(op_name:str, axis:int, weights:int=0, op_inputs:list=[], op_outputs:list=[])->dict:

    assert axis >= 0, "axis:{} should >= 0".format(axis)
    assert weights >= 0, "weights:{} should >= 0".format(weights)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "FullyConnected"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["weights"] = weights
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSoftmaxOpConfig(op_name:str, beta:float, axis:int=0, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Softmax"
    op_attr = {}
    op_attr["beta"] = beta
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructResizeOpConfig(op_name:str, type:str, factor:float, align_corners:bool,
        half_pixel_centers:bool, target_height:int, target_width:int, 
        layout:str="WHCN", op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Resize"
    op_attr = {}
    op_attr["type"] = type
    op_attr["factor"] = factor
    op_attr["align_corners"] = align_corners
    op_attr["half_pixel_centers"] = half_pixel_centers
    op_attr["target_height"] = target_height
    op_attr["target_width"] = target_width
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructPool2dOpConfig(op_name:str, type:str, ksize:list=[], stride:list=[], padding:str="AUTO",
    pad:list=[0, 0, 0, 0], input_size:list=[], output_size:list=[], round_type:str="FLOOR", 
    layout:str="WHCN", op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert round_type in RoundType, "round_type:{} is not in {}".format(round_type, RoundType)
    assert layout in DataLayout, "layout:{} is not in {}".format(layout, DataLayout)
    if len(input_size) == 0:
        assert len(ksize) and len(stride), "ksize and stride len should > 0, when input_size len is 0"
    if len(input_size) > 0:
        assert len(ksize) == 0 and len(stride) == 0, "ksize and stride len should be 0, when input_size len > 0"
    if padding != "AUTO":
        assert pad == [0, 0, 0, 0], "pad should be [0, 0, 0, 0], when padding is not AUTO"
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Pool2d"
    op_attr = {}
    op_attr["type"] = type
    if len(input_size) > 0 and len(output_size) == 0:
        op_attr["input_size"] = input_size
    elif len(input_size) > 0 and len(output_size) > 0:
        op_attr["input_size"] = input_size
        op_attr["output_size"] = output_size
    elif len(input_size) == 0 and padding == "AUTO":
        op_attr["pad"] = pad
        op_attr["ksize"] = ksize
        op_attr["stride"] = stride
    else:
        op_attr["padding"] = padding
        op_attr["ksize"] = ksize
        op_attr["stride"] = stride
    op_attr["round_type"] = round_type
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructConcatOpConfig(op_name:str, axis:int, op_inputs:list=[], op_outputs:list=[])->dict:
    assert axis >= 0, "axis should greater than zero"
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Concat"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["input_cnt"] = len(op_inputs)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructDataConvertConfig(op_name:str, op_inputs:list=[], op_outputs:list=[])->dict:
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "DataConvert"
    op_attr = {}
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict