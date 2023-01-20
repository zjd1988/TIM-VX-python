/***********************************
******  broadcast_op.cpp
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#include "tim/vx/ops/broadcast.h"
#include "broadcast_op.h"

namespace TIMVXPY
{
    bool BroadcastCreator::parse_op_attr(const py::dict &op_info, BroadcastOpAttr &op_attr)
    {
        return parse_dynamic_list<py::int_, int32_t>(op_info, m_op_name, "shape", op_attr.shape) &&
            parse_dynamic_list<py::int_, int32_t>(op_info, m_op_name, "dimensions", op_attr.dimensions, false);
    }

    Operation* BroadcastCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        BroadcastOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> shape = op_attr.shape;
        std::vector<int32_t> dimensions = op_attr.dimensions;
        return graph->CreateOperation<ops::Broadcast>(shape, dimensions).get();
    }

    REGISTER_OP_CREATOR(BroadcastCreator, Broadcast);
} // namespace TIMVXPY