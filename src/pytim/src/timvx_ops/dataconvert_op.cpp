/***********************************
******  dataconvert_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/simple_operations.h"
#include "timvx_ops/dataconvert_op.h"

namespace TIMVXPY
{
    bool DataConvertCreator::parse_op_attr(const py::dict &op_info, DataConvertOpAttr &op_attr)
    {
        return true;
    }

    Operation* DataConvertCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        DataConvertOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        return graph->CreateOperation<ops::DataConvert>().get();
    }

    REGISTER_OP_CREATOR(DataConvertCreator, DataConvert);

} // namespace TIMVXPY