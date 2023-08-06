# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

if __name__ == "__main__":
    timvx_engine = Engine("lenet")
    assert timvx_engine.create_graph(), "engine create grah fail!"

    input_name = "input"
    assert timvx_engine.create_tensor(input_name, "FLOAT32", "INPUT", [5, 1]), \
        "construct tensor {} fail!".format(input_name)

    output_name = "output"
    assert timvx_engine.create_tensor(output_name, "FLOAT32", "OUTPUT", [5, 1]), \
        "construct tensor {} fail!".format(output_name)

    in_data = np.array([-2.5, -0.1, 0, 0.55, float('inf')]).reshape((5,1))
    golden = np.array([-0.5, 1.9, 2, 2.55, float('inf')]).reshape((5,1))

    op_name = "linear"
    op_inputs = ["input", ]
    op_outputs = ["output", ]
    op_parameter = {}
    op_parameter["a"] = 1
    op_parameter["b"] = 2
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type="Linear", parameter=op_parameter, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert timvx_engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    assert timvx_engine.compile_graph(), "compile graph fail...."

    input_dict = {}
    input_dict[input_name] = in_data
    out_data = timvx_engine.run_graph(input_dict)
    print(out_data)
