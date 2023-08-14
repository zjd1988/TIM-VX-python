/***********************************
******  conv1d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/conv1d.h"
#include "timvx_ops/conv1d_op.h"

namespace TimVX
{

    bool Conv1dCreator::parseWeightsAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    bool Conv1dCreator::parsePaddingAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool Conv1dCreator::parseKsizeAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "ksize", op_attr.ksize, false);
    }

    bool Conv1dCreator::parseStrideAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Conv1dCreator::parseDilationAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool Conv1dCreator::parsePadAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool Conv1dCreator::parseMultiplierAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "multiplier", op_attr.multiplier, false);
    }

    bool Conv1dCreator::parseKernelLayoutAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "kernel_layout", op_attr.kernel_layout, false);
    }

    bool Conv1dCreator::parseOpAttr(const json& op_info, Conv1dOpAttr& op_attr)
    {
        op_attr.weights = 0;
        op_attr.padding = PadType::AUTO;
        op_attr.ksize = 0;
        op_attr.multiplier = 0;
        op_attr.pad = {0, 0};
        op_attr.input_layout = DataLayout::WHCN; // always set WHCN
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parseWeightsAttr(op_info, op_attr) && parsePaddingAttr(op_info, op_attr) && 
            parseKsizeAttr(op_info, op_attr) && parseStrideAttr(op_info, op_attr) && 
            parseDilationAttr(op_info, op_attr) && parsePadAttr(op_info, op_attr) && 
            parseMultiplierAttr(op_info, op_attr) && parseKernelLayoutAttr(op_info, op_attr);
    }

    Operation* Conv1dCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Conv1dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t                weights        = op_attr.weights;
        PadType                 padding        = op_attr.padding;
        uint32_t                ksize          = op_attr.ksize;
        uint32_t                stride         = op_attr.stride;
        uint32_t                dilation       = op_attr.dilation;
        std::array<uint32_t, 2> pad            = op_attr.pad;
        int32_t                 multiplier     = op_attr.multiplier;
        DataLayout              input_layout   = op_attr.input_layout;
        DataLayout              kernel_layout  = op_attr.kernel_layout;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "weights       : {}");
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "padding       : {}", (int)padding);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "ksize         : {}", ksize);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "stride        : {}", stride);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "dilation      : {}", dilation);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "pad           : {} {}", pad[0], pad[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "multiplier    : {}", multiplier);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "kernel_layout : {}", (int)kernel_layout);
        return graph->CreateOperation<ops::Conv1d>(weights, padding, ksize, stride, 
            dilation, pad, multiplier, input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(Conv1dCreator, Conv1d);

} // namespace TimVX