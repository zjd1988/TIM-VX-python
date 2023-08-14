/***********************************
******  clip_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/clip.h"
#include "timvx_ops/clip_op.h"

namespace TimVX
{

    bool ClipCreator::parseMinAttr(const json& op_info, ClipOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "min", op_attr.min);
    }

    bool ClipCreator::parseMaxAttr(const json& op_info, ClipOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "max", op_attr.max);
    }

    bool ClipCreator::parseOpAttr(const json& op_info, ClipOpAttr& op_attr)
    {
        return parseMinAttr(op_info, op_attr) && parseMaxAttr(op_info, op_attr);
    }

    Operation* ClipCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ClipOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        float min = op_attr.min;
        float max = op_attr.max;
        return graph->CreateOperation<ops::Clip>(min, max).get();
    }

    REGISTER_OP_CREATOR(ClipCreator, Clip);

} // namespace TimVX