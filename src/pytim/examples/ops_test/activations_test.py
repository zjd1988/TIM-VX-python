# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

def test_Linear_shape_5_1_fp32():
    # create graph
    timvx_engine = Engine("lenet")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", [5, 1]), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [5, 1]), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "linear"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_parameter = {}
    op_parameter["a"] = 1
    op_parameter["b"] = 2
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type="Linear", parameter=op_parameter, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."

    # run graph with input data
    input_data = np.array([-2.5, -0.1, 0, 0.55, float('inf')]).reshape((5,1))
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    golden_data = np.array([-0.5, 1.9, 2.0, 2.55, float('inf')]).reshape((5,1))
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), "check gloden data with output data not equal!"

def test_Linear_shape_5_1_fp32_omit_b():
    # create graph
    timvx_engine = Engine("lenet")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    # construct tensors
    input_name = "input"
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", [5, 1]), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [5, 1]), \
        "construct tensor {} fail!".format(output_name)

    # construct operations
    op_name = "linear"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_parameter = {}
    op_parameter["a"] = 2
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type="Linear", parameter=op_parameter, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    # compile graph
    assert timvx_engine.compile_graph(), "compile graph fail...."
    
    # run graph with input data
    input_data = np.array([-2.5, -0.1, 0, 0.55, float('inf')]).reshape((5,1))
    input_dict = {}
    input_dict["input"] = input_data
    output_data = timvx_engine.run_graph(input_dict)

    # compare gloden data with output data
    golden_data = np.array([-5.0, -0.2, 0, 1.1, float('inf')]).reshape((5,1))
    assert np.allclose(golden_data, output_data[0], atol=1.e-6), "check gloden data with output data not equal!"

if __name__ == "__main__":
    test_Linear_shape_5_1_fp32()
    test_Linear_shape_5_1_fp32_omit_b()

