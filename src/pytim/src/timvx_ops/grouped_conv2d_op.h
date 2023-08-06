/***********************************
******  grouped_conv2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{
    
    class GroupedConv2dCreator : public OpCreator
    {
    public:
        struct GroupedConv2dOpAttr
        {
            PadType                 padding;
            std::array<uint32_t, 2> stride;
            std::array<uint32_t, 2> dilation;
            std::array<uint32_t, 4> pad;
            int32_t                 grouped_number;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parse_padding(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parse_stride(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parse_dilation(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parse_pad(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parse_grouped_number(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parse_input_layout(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parse_kernel_layout(const json& op_info, GroupedConv2dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, GroupedConv2dOpAttr& op_attr);

    private:
        std::string m_op_name = "GroupedConv2d";
    };

} // namespace TimVX
