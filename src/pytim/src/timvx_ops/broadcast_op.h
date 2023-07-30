/***********************************
******  broadcast_op.h
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TIMVXPY
{

    class BroadcastCreator : public OpCreator
    {
    public:
        struct BroadcastOpAttr
        {
            std::vector<int32_t> shape;
            std::vector<int32_t> dimensions;
        };
    
        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool parse_op_attr(const py::dict &op_info, BroadcastOpAttr &op_attr);

    private:
        std::string m_op_name = "Broadcast";
    };

} // namespace TIMVXPY
