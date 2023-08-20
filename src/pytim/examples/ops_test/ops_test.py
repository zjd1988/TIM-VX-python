# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from examples.ops_test.activations_test import test_activations_op
from examples.ops_test.elementwise_test import test_elementwise_op
from examples.ops_test.conv1d_test import test_conv1d_op
from examples.ops_test.conv2d_test import test_conv2d_op
from examples.ops_test.deconv1d_test import test_deconv1d_op

if __name__ == "__main__":
    test_result = {}
    test_result = test_activations_op()
    test_result.update(test_elementwise_op())
    test_result.update(test_conv1d_op())
    test_result.update(test_conv2d_op())

    print("ops_test summary: ")
    for key, value in test_result.items():
        print("{}: {}".format(key, value))