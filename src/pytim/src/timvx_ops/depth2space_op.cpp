/***********************************
******  depth2space_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/depth2space.h"
#include "timvx_ops/depth2space_op.h"

namespace TimVX
{

    bool Depth2SpaceCreator::parseBlockSize(const json& op_info, Depth2SpaceOpAttr& op_attr)
    {
        return parseValue<int32_t>(op_info, m_op_name, "block_size", op_attr.block_size);
    }

    bool Depth2SpaceCreator::parseLayout(const json& op_info, Depth2SpaceOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool Depth2SpaceCreator::parseOpAttr(const json& op_info, Depth2SpaceOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parseBlockSize(op_info, op_attr) && parseLayout(op_info, op_attr);
    }

    Operation* Depth2SpaceCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Depth2SpaceOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        int32_t block_size = op_attr.block_size;
        DataLayout layout = op_attr.layout;
        return graph->CreateOperation<ops::DepthToSpace>(block_size, layout).get();
    }

    REGISTER_OP_CREATOR(Depth2SpaceCreator, Depth2Space);

} // namespace TimVX