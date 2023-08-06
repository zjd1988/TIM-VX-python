/***********************************
******  dataconvert_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/simple_operations.h"
#include "timvx_ops/dataconvert_op.h"

namespace TimVX
{
    bool DataConvertCreator::parseOpAttr(const json& op_info, DataConvertOpAttr& op_attr)
    {
        return true;
    }

    Operation* DataConvertCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        DataConvertOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        return graph->CreateOperation<ops::DataConvert>().get();
    }

    REGISTER_OP_CREATOR(DataConvertCreator, DataConvert);

} // namespace TimVX