/***********************************
******  l2normalization_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class L2NormalizationCreator : public OpCreator
    {
    public:
        struct L2NormalizationOpAttr
        {
            int32_t axis;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(const json& op_info, L2NormalizationOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, L2NormalizationOpAttr& op_attr);

    private:
        std::string m_op_name = "L2Normalization";
    };

} // namespace TimVX
