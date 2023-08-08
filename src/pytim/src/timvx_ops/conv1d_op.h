/***********************************
******  conv1d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class Conv1dCreator : public OpCreator
    {
    public:
        struct Conv1dOpAttr
        {
            uint32_t                weights;
            PadType                 padding;
            uint32_t                ksize;
            uint32_t                stride;
            uint32_t                dilation;
            std::array<uint32_t, 2> pad;
            int32_t                 multiplier;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseWeights(const json& op_info, Conv1dOpAttr& op_attr);
        bool parsePadding(const json& op_info, Conv1dOpAttr& op_attr);
        bool parseKsize(const json& op_info, Conv1dOpAttr& op_attr);
        bool parseStride(const json& op_info, Conv1dOpAttr& op_attr);
        bool parseDilation(const json& op_info, Conv1dOpAttr& op_attr);
        bool parsePad(const json& op_info, Conv1dOpAttr& op_attr);
        bool parseMultiplier(const json& op_info, Conv1dOpAttr& op_attr);
        bool parseKernelLayout(const json& op_info, Conv1dOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, Conv1dOpAttr& op_attr);

    private:
        std::string m_op_name = "Conv1d";
    };

} // namespace TimVX