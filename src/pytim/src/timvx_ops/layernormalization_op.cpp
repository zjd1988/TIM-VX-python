/***********************************
******  layernormalization_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/layernormalization.h"
#include "timvx_ops/layernormalization_op.h"

namespace TimVX
{

    bool LayerNormalizationCreator::parseAxisAttr(const json& op_info, LayerNormalizationOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "axis", op_attr.axis, false);
    }

    bool LayerNormalizationCreator::parseEpsAttr(const json& op_info, LayerNormalizationOpAttr& op_attr)
    {
        return parseValue<float>(op_info, m_op_name, "eps", op_attr.eps, false);
    }

    bool LayerNormalizationCreator::parseOpAttr(const json& op_info, LayerNormalizationOpAttr& op_attr)
    {
        op_attr.axis = 0;
        op_attr.eps = 1e-5f;
        return parseAxisAttr(op_info, op_attr) && parseEpsAttr(op_info, op_attr);
    }

    Operation* LayerNormalizationCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        LayerNormalizationOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        int32_t axis = op_attr.axis;
        float eps = op_attr.eps;
        return graph->CreateOperation<ops::LayerNormalization>(axis, eps).get();
    }

    REGISTER_OP_CREATOR(LayerNormalizationCreator, LayerNormalization);

} // namespace TimVX