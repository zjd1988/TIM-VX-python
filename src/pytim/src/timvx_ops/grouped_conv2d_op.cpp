/***********************************
******  grouped_conv2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/groupedconv2d.h"
#include "timvx_ops/grouped_conv2d_op.h"

namespace TimVX
{

    bool GroupedConv2dCreator::parse_padding(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parsePadType(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool GroupedConv2dCreator::parse_stride(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool GroupedConv2dCreator::parse_dilation(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 2>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool GroupedConv2dCreator::parse_pad(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parseFixList<uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool GroupedConv2dCreator::parse_grouped_number(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "grouped_number", op_attr.grouped_number);
    }

    bool GroupedConv2dCreator::parse_input_layout(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "input_layout", op_attr.input_layout, false);
    }
    
    bool GroupedConv2dCreator::parse_kernel_layout(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "kernel_layout", op_attr.input_layout, false);
    }

    bool GroupedConv2dCreator::parseOpAttr(const json& op_info, GroupedConv2dOpAttr& op_attr)
    {
        op_attr.padding = PadType::AUTO;
        op_attr.pad = {0, 0, 0, 0};
        op_attr.input_layout = DataLayout::WHCN;
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parse_padding(op_info, op_attr) && parse_stride(op_info, op_attr)
            && parse_dilation(op_info, op_attr) && parse_pad(op_info, op_attr)
            && parse_grouped_number(op_info, op_attr) && parse_input_layout(op_info, op_attr) 
            && parse_kernel_layout(op_info, op_attr);
    }

    Operation* GroupedConv2dCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        GroupedConv2dOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        PadType                 padding            = op_attr.padding;
        std::array<uint32_t, 2> stride             = op_attr.stride;
        std::array<uint32_t, 2> dilation           = op_attr.dilation;
        std::array<uint32_t, 4> pad                = op_attr.pad;
        int32_t                 grouped_number     = op_attr.grouped_number;
        DataLayout              input_layout       = op_attr.input_layout;
        DataLayout              kernel_layout      = op_attr.kernel_layout;
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "padding: {}", (int)padding);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "stride: {} {}", stride[0], stride[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "dilation: {} {}", dilation[0], dilation[1]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "pad: {} {} {} {}", pad[0], pad[1], pad[2], pad[3]);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "grouped_number: {}", grouped_number);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "input_layout: {}", (int)input_layout);
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "kernel_layout: {}", (int)kernel_layout);
        if (0 != pad[0] || 0 != pad[2] || 0 != pad[2] || 0 != pad[3])
            return graph->CreateOperation<ops::GroupedConv2d>(pad, stride, dilation, grouped_number, 
                input_layout, kernel_layout).get();
        else
            return graph->CreateOperation<ops::GroupedConv2d>(padding, stride, dilation, grouped_number, 
                input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(GroupedConv2dCreator, GroupedConv2d);

} // namespace TimVX