/***********************************
******  activation_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/activations.h"
#include "timvx_ops/activation_op.h"

namespace TimVX
{

    bool ActivationCreator::parsePreluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_prelu";
        return parseValue<int>(op_info, full_op_name, "axis", op_attr.prelu.axis);
    }

    bool ActivationCreator::parseLeakyreluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_leakyrelu";
        return parseValue<float>(op_info, full_op_name, "ratio", op_attr.leakyrelu.ratio);
    }

    bool ActivationCreator::parseLinearAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_linear";
        return parseValue<float>(op_info, full_op_name, "a", op_attr.linear.a) && 
            parseValue<float>(op_info, full_op_name, "b", op_attr.linear.b, false);
    }

    bool ActivationCreator::parseGeluAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_gelu";
        return parseValue<bool>(op_info, full_op_name, "approximate", op_attr.gelu.approximate, false);
    }

    bool ActivationCreator::parseHardsigmoidAttr(const json& op_info, ActivationOpAttr& op_attr)
    {
        std::string full_op_name = m_op_name + "_hardsigmoid";
        return parseValue<float>(op_info, full_op_name, "alpha", op_attr.hardsigmoid.alpha) &&
            parseValue<float>(op_info, full_op_name, "beta", op_attr.hardsigmoid.beta);
    }

    bool ActivationCreator::parseOpAttr(std::string op_type, const json& op_info, ActivationOpAttr& op_attr)
    {
        op_attr.gelu.approximate = true;
        op_attr.linear.b = 0.0f;
        if ("Prelu" == op_type)
            return parsePreluAttr(op_info, op_attr);
        else if ("Leakyrelu" == op_type)
            return parseLeakyreluAttr(op_info, op_attr);
        else if ("Linear" == op_type)
            return parseLinearAttr(op_info, op_attr);
        else if ("Gelu" == op_type)
            return parseGeluAttr(op_info, op_attr);
        else if ("HardSigmoid" == op_type)
            return parseHardsigmoidAttr(op_info, op_attr);
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported activation op type: {}", op_type);
        return false;
    }

    Operation* ActivationCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        ActivationOpAttr op_attr;
        std::string activation_type;
        if (!parseValue<std::string>(op_info, m_op_name, "activation_type", activation_type))
            return nullptr;
        if (!parseOpAttr(activation_type, op_info, op_attr))
            return nullptr;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, activation_type);
        if ("Relu" == activation_type)
        {
            return graph->CreateOperation<ops::Relu>().get();
        }
        else if ("Relu1" == activation_type)
        {
            return graph->CreateOperation<ops::Relu1>().get();
        }
        else if ("Relu6" == activation_type)
        {
            return graph->CreateOperation<ops::Relu6>().get();
        }
        else if ("Elu" == activation_type)
        {
            return graph->CreateOperation<ops::Elu>().get();
        }
        else if ("Sigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::Sigmoid>().get();
        }
        else if ("Mish" == activation_type)
        {
            return graph->CreateOperation<ops::Mish>().get();
        }
        else if ("HardSigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::HardSigmoid>().get();
        }
        else if ("SoftRelu" == activation_type)
        {
            return graph->CreateOperation<ops::SoftRelu>().get();
        }
        else if ("HardSwish" == activation_type)
        {
            return graph->CreateOperation<ops::HardSwish>().get();
        }
        else if ("Prelu" == activation_type)
        {
            int axis = op_attr.prelu.axis;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, axis);
            return graph->CreateOperation<ops::Prelu>(axis).get();
        }
        else if ("Tanh" == activation_type)
        {
            return graph->CreateOperation<ops::Tanh>().get();
        }
        else if ("LeakyRelu" == activation_type)
        {
            float ratio = op_attr.leakyrelu.ratio;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, ratio);
            return graph->CreateOperation<ops::LeakyRelu>(ratio).get();
        }
        else if ("Linear" == activation_type)
        {
            float a = op_attr.linear.a;
            float b = op_attr.linear.b;
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, a);
            TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, b);
            return graph->CreateOperation<ops::Linear>(a, b).get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported activation op type: {}", activation_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(ActivationCreator, Activation);

} // namespace TimVX