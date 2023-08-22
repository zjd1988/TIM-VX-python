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
from examples.ops_test.deconv2d_test import test_deconv2d_op
from examples.ops_test.groupedconv2d_test import test_groupedconv2d_op
from examples.ops_test.instancenormalization_test import test_instancenormalization_op
from examples.ops_test.layernormalization_test import test_layernormalization_op
from examples.ops_test.logsoftmax_test import test_logsoftmax_op
from examples.ops_test.matmul_test import test_matmul_op
from examples.ops_test.maxpoolwithargmax_test import test_maxpoolwithargmax_op

if __name__ == "__main__":
    test_result = {}
    test_result.update(test_activations_op())
    test_result.update(test_elementwise_op())
    test_result.update(test_conv1d_op())
    test_result.update(test_conv2d_op())
    test_result.update(test_deconv1d_op())
    test_result.update(test_deconv2d_op())
    test_result.update(test_groupedconv2d_op())
    test_result.update(test_instancenormalization_op())
    test_result.update(test_layernormalization_op())
    test_result.update(test_logsoftmax_op())
    test_result.update(test_matmul_op())
    test_result.update(test_maxpoolwithargmax_op())

    print("ops_test summary: ")
    for key, value in test_result.items():
        print("{}: {}".format(key, value))