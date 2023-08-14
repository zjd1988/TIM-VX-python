/***********************************
******  conv2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/conv2d.h"
#include "timvx_ops/conv2d_op.h"

namespace TimVX
{

    bool Conv2dCreator::parseWeightsAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    bool Conv2dCreator::parsePaddingAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool Conv2dCreator::parseKsizeAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize, false);
    }

    bool Conv2dCreator::parseStrideAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Conv2dCreator::parseDilationAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool Conv2dCreator::parsePadAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool Conv2dCreator::parseMultiplierAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "multiplier", op_attr.multiplier, false);
    }

    bool Conv2dCreator::parseInputLayoutAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "input_layout", op_attr.input_layout, false);
    }
    
    bool Conv2dCreator::parseKernelLayoutAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "kernel_layout", op_attr.kernel_layout, false);
    }

    bool Conv2dCreator::parseOpAttr(const json& op_info, Conv2dOpAttr& op_attr)
    {
        op_attr.weights = 0;
        op_attr.padding = PadType::AUTO;
        op_attr.ksize = {0, 0};
        op_attr.multiplier = 0;
        op_attr.pad = {0, 0, 0, 0};
        op_attr.input_layout = DataLayout::WHCN;
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parseWeightsAttr(op_info, op_attr) && parsePaddingAttr(op_info, op_attr) && 
            parseKsizeAttr(op_info, op_attr) && parseStrideAttr(op_info, op_attr) && 
            parseDilationAttr(op_info, op_attr) && parsePadAttr(op_info, op_attr) && 
            parseMultiplierAttr(op_info, op_attr) && parseInputLayoutAttr(op_info, op_attr) && 
            parseKernelLayoutAttr(op_info, op_attr);
    }

    Operation* Conv2dCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Conv2dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t                weights        = op_attr.weights;
        PadType                 padding        = op_attr.padding;
        std::array<uint32_t, 2> ksize          = op_attr.ksize;
        std::array<uint32_t, 2> stride         = op_attr.stride;
        std::array<uint32_t, 2> dilation       = op_attr.dilation;
        std::array<uint32_t, 4> pad            = op_attr.pad;
        int32_t                 multiplier     = op_attr.multiplier;
        DataLayout              input_layout   = op_attr.input_layout;
        DataLayout              kernel_layout  = op_attr.kernel_layout;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "weights       : {}", weights);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "padding       : {}", (int)padding);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "ksize         : {} {}", ksize[0], ksize[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "stride        : {} {}", stride[0], stride[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "dilation      : {} {}", dilation[0], dilation[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "pad           : {} {} {} {}", pad[0], pad[1], pad[2], pad[3]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "multiplier    : {}", multiplier);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "input_layout  : {}", (int)input_layout);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "kernel_layout : {}", (int)kernel_layout);
        return graph->CreateOperation<ops::Conv2d>(weights, padding, ksize, stride, 
            dilation, pad, multiplier, input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(Conv2dCreator, Conv2d);

} // namespace TimVX