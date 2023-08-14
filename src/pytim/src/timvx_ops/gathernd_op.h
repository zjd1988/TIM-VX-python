/***********************************
******  gathernd_op.h
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class GatherNdCreator : public OpCreator
    {
    public:
        struct GatherNdOpAttr
        {
        };
    
        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseOpAttr(const json& op_info, GatherNdOpAttr& op_attr);

    private:
        std::string m_op_name = "GatherNd";
    };

} // namespace TimVX
