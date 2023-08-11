/***********************************
******  batch2space_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/batch2space.h"
#include "timvx_ops/batch2space_op.h"

namespace TimVX
{

    bool Batch2SpaceCreator::parseLayout(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        return parseDataLayoutType(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool Batch2SpaceCreator::parseBlockSize(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "block_size", op_attr.block_size);
    }

    bool Batch2SpaceCreator::parseCrop(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "crop", op_attr.crop);
    }

    bool Batch2SpaceCreator::parseOpAttr(const json& op_info, Batch2SpaceOpAttr& op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parseLayout(op_info, op_attr) && parseBlockSize(op_info, op_attr) && 
            parseCrop(op_info, op_attr);
    }

    Operation* Batch2SpaceCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        Batch2SpaceOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        std::vector<int32_t> block_size = op_attr.block_size;
        std::vector<int32_t> crop = op_attr.crop;
        DataLayout layout = op_attr.layout;
        return graph->CreateOperation<ops::Batch2Space>(block_size, crop, layout).get();
    }

    REGISTER_OP_CREATOR(Batch2SpaceCreator, Batch2Space);

} // namespace TimVX