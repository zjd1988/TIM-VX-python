/***********************************
******  grouped_conv2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/groupedconv2d.h"
#include "grouped_conv2d_op.h"


namespace TIMVXPY
{

    bool GroupedConv2dCreator::parse_padding(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_pad_type(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool GroupedConv2dCreator::parse_stride(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool GroupedConv2dCreator::parse_dilation(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool GroupedConv2dCreator::parse_pad(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool GroupedConv2dCreator::parse_grouped_number(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_value<py::int_, int32_t>(op_info, m_op_name, "grouped_number", op_attr.grouped_number);
    }

    bool GroupedConv2dCreator::parse_input_layout(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_data_layout_type(op_info, m_op_name, "input_layout", op_attr.input_layout, false);
    }
    
    bool GroupedConv2dCreator::parse_kernel_layout(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
    {
        return parse_data_layout_type(op_info, m_op_name, "kernel_layout", op_attr.input_layout, false);
    }

    bool GroupedConv2dCreator::parse_op_attr(const py::dict &op_info, GroupedConv2dOpAttr &op_attr)
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

    Operation* GroupedConv2dCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        GroupedConv2dOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        PadType                 padding            = op_attr.padding;
        std::array<uint32_t, 2> stride             = op_attr.stride;
        std::array<uint32_t, 2> dilation           = op_attr.dilation;
        std::array<uint32_t, 4> pad                = op_attr.pad;
        int32_t                 grouped_number     = op_attr.grouped_number;
        DataLayout              input_layout       = op_attr.input_layout;
        DataLayout              kernel_layout      = op_attr.kernel_layout;
        // std::cout << "padding: " << (int)padding << std::endl;
        // std::cout << "stride: " << stride[0] << " " << stride[1] << std::endl;
        // std::cout << "dilation: " << dilation[0] << " " << dilation[1] << std::endl;
        // std::cout << "pad: " << pad[0] << " " << pad[1] << " " << pad[2] << " " << pad[3] << std::endl;
        // std::cout << "grouped_number: " << grouped_number << std::endl;
        // std::cout << "input_layout: " << (int)input_layout << std::endl;
        // std::cout << "kernel_layout: " << (int)kernel_layout << std::endl;
        if (0 != pad[0] || 0 != pad[2] || 0 != pad[2] || 0 != pad[3])
            return graph->CreateOperation<ops::GroupedConv2d>(pad, stride, dilation, grouped_number, 
                input_layout, kernel_layout).get();
        else
            return graph->CreateOperation<ops::GroupedConv2d>(padding, stride, dilation, grouped_number, 
                input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(GroupedConv2dCreator, GroupedConv2d);
} // namespace TIMVXPY