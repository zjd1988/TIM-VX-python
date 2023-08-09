/***********************************
******  deconv1d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/deconv1d.h"
#include "timvx_ops/deconv1d_op.h"

namespace TimVX
{

    bool DeConv1dCreator::parseOcCount(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "oc_count", op_attr.oc_count, false);
    }

    bool DeConv1dCreator::parsePadding(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "pad_type", op_attr.pad_type);
    }

    bool DeConv1dCreator::parseKsize(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "ksize", op_attr.ksize, false);
    }

    bool DeConv1dCreator::parseStride(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool DeConv1dCreator::parseOutputPadding(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "output_padding", op_attr.output_padding);
    }

    bool DeConv1dCreator::parsePad(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool DeConv1dCreator::parseGroup(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "group", op_attr.group, false);
    }

    bool DeConv1dCreator::parseKernelLayout(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "kernel_layout", op_attr.kernel_layout, false);
    }

    bool DeConv1dCreator::parseOpAttr(const json& op_info, DeConv1dOpAttr& op_attr)
    {
        op_attr.oc_count = 0;
        op_attr.pad_type = PadType::AUTO;
        op_attr.ksize = 0;
        op_attr.output_padding = 0;
        op_attr.pad = {0, 0};
        op_attr.group = 1;
        op_attr.input_layout = DataLayout::WHCN; // always set WHCN
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parseOcCount(op_info, op_attr) && parsePadding(op_info, op_attr) && 
            parseKsize(op_info, op_attr) && parseStride(op_info, op_attr) && 
            parseOutputPadding(op_info, op_attr) && parsePad(op_info, op_attr) && 
            parseGroup(op_info, op_attr) && parseKernelLayout(op_info, op_attr);
    }

    Operation* DeConv1dCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        DeConv1dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t                oc_count       = op_attr.oc_count;
        PadType                 pad_type       = op_attr.pad_type;
        uint32_t                ksize          = op_attr.ksize;
        uint32_t                stride         = op_attr.stride;
        uint32_t                output_padding = op_attr.output_padding;
        std::array<uint32_t, 2> pad            = op_attr.pad;
        uint32_t                group          = op_attr.group;
        DataLayout              input_layout   = op_attr.input_layout;
        DataLayout              kernel_layout  = op_attr.kernel_layout;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "oc_count       : {}");
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "pad_type       : {}", (int)pad_type);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "ksize          : {}", ksize);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "stride         : {}", stride);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "output_padding : {}", output_padding);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "pad            : {} {}", pad[0], pad[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "group          : {}", group);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "input_layout   : {}", (int)input_layout);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "kernel_layout  : {}", (int)kernel_layout);
        return graph->CreateOperation<ops::DeConv1d>(pad_type, stride, 
            output_padding, pad, group, input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(DeConv1dCreator, DeConv1d);

} // namespace TimVX