/***********************************
******  nbg_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/nbg.h"
#include "timvx_ops/nbg_op.h"

namespace TimVX
{

    bool NBGCreator::parseBinary(const json& op_info, NBGOpAttr& op_attr)
    {
        // use uint64_t to store void* binary
        uint64_t binary_ptr;
        if (parseValue<size_t>(op_info, m_op_name, "binary", binary_ptr))
            op_attr.binary = (void*)binary_ptr;
        else
            return false;
        return true;
    }

    bool NBGCreator::parseInputCount(const json& op_info, NBGOpAttr& op_attr)
    {
        return parseValue<size_t>(op_info, m_op_name, "input_count", op_attr.input_count);
    }

    bool NBGCreator::parseOutputCount(const json& op_info, NBGOpAttr& op_attr)
    {
        return parseValue<size_t>(op_info, m_op_name, "output_count", op_attr.output_count);
    }

    bool NBGCreator::parseOpAttr(const json& op_info, NBGOpAttr& op_attr)
    {
        op_attr.input_count = 0;
        op_attr.output_count = 0;
        return parseInputCount(op_info, op_attr) && parseOutputCount(op_info, op_attr);
    }

    Operation* NBGCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        NBGOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        const char* binary = (const char*)op_attr.binary;
        size_t input_count = op_attr.input_count;
        size_t output_count = op_attr.output_count;
        return graph->CreateOperation<ops::NBG>(binary, input_count, output_count).get();
    }

    REGISTER_OP_CREATOR(NBGCreator, NBG);

} // namespace TimVX