/***********************************
******  addn_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/addn.h"
#include "timvx_ops/addn_op.h"

namespace TimVX
{

    bool AddNCreator::parseNumInput(const json& op_info, AddNOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "num_inputs", op_attr.num_inputs);
    }

    bool AddNCreator::parseOpAttr(const json& op_info, AddNOpAttr& op_attr)
    {
        op_attr.num_inputs = 0;
        return parseNumInput(op_info, op_attr);
    }

    Operation* AddNCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        AddNOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t num_inputs = op_attr.num_inputs;
        return graph->CreateOperation<ops::AddN>(num_inputs).get();
    }

    REGISTER_OP_CREATOR(AddNCreator, AddN);

} // namespace TimVX