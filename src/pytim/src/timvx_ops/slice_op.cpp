/***********************************
******  slice_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/slice.h"
#include "timvx_ops/slice_op.h"

namespace TimVX
{

    bool SliceCreator::parseDimsAttr(const json& op_info, SliceOpAttr& op_attr)
    {
        return parseValue<uint32_t>(op_info, m_op_name, "dims", op_attr.dims);
    }

    bool SliceCreator::parseStartAttr(const json& op_info, SliceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "start", op_attr.start);
    }

    bool SliceCreator::parseLengthAttr(const json& op_info, SliceOpAttr& op_attr)
    {
        return parseDynamicList<int32_t>(op_info, m_op_name, "length", op_attr.length);
    }

    bool SliceCreator::parseOpAttr(const json& op_info, SliceOpAttr& op_attr)
    {
        return parseDimsAttr(op_info, op_attr) && parseStartAttr(op_info, op_attr) && 
            parseLengthAttr(op_info, op_attr);
    }

    Operation* SliceCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        SliceOpAttr op_attr;
        if (!parseOpAttr(op_info, op_attr))
            return nullptr;

        uint32_t dims               = op_attr.dims;
        std::vector<int32_t> start  = op_attr.start;
        std::vector<int32_t> length = op_attr.length;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, dims);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, start);
        TIMVX_LOG_STL_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, length);
        return graph->CreateOperation<ops::Slice>(dims, start, length).get();
    }

    REGISTER_OP_CREATOR(SliceCreator, Slice);

} // namespace TimVX