/***********************************
******  deconv2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{
    
    class DeConv2dCreator : public OpCreator
    {
    public:
        struct DeConv2dOpAttr
        {
            uint32_t                oc_count;
            PadType                 pad_type;
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            std::array<uint32_t, 2> output_padding;
            std::array<uint32_t, 4> pad;
            uint32_t                group;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOcCount(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parsePadding(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseKsize(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseStride(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseOutputPadding(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parsePad(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseGroup(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseKernelLayout(const json& op_info, DeConv2dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, DeConv2dOpAttr& op_attr);

    private:
        std::string m_op_name = "DeConv2d";
    };

} // namespace TimVX
