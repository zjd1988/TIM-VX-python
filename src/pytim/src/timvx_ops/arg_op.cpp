/***********************************
******  arg_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/arg.h"
#include "timvx_ops/arg_op.h"

namespace TimVX
{

    bool ArgCreator::parseOpAttr(std::string op_type, const json& op_info, ArgOpAttr& op_attr)
    {
        std::string full_name = m_op_name + op_type;
        return parseValue<int>(op_info, full_name, "axis", op_attr.axis);
    }

    Operation* ArgCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ArgOpAttr op_attr;
        std::string arg_type;
        if (!parseValue<std::string>(op_info, m_op_name, "arg_type", arg_type))
            return nullptr;
        if (!parseOpAttr(arg_type, op_info, op_attr))
            return nullptr;
        if ("Max" == arg_type)
        {
            int axis = op_attr.axis;
            return graph->CreateOperation<ops::ArgMax>(axis).get();
        }
        else if ("Min" == arg_type)
        {
            int axis = op_attr.axis;
            return graph->CreateOperation<ops::ArgMin>(axis).get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported arg op type: {}", arg_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(ArgCreator, Arg);

} // namespace TimVX