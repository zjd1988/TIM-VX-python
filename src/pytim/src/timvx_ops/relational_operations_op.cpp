/***********************************
******  relational_operations_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/relational_operations.h"
#include "timvx_ops/relational_operations_op.h"

namespace TimVX
{

    bool RelationalOperationsCreator::parseGreaterAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsCreator::parseGreaterOrEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsCreator::parseLessAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsCreator::parseLessOrEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsCreator::parseNotEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsCreator::parseEqualAttr(const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        return true;
    }

    bool RelationalOperationsCreator::parseOpAttr(std::string relational_op_type, const json& op_info, RelationalOperationsOpAttr& op_attr)
    {
        if ("Greater" == relational_op_type)
            return parseGreaterAttr(op_info, op_attr);
        else if ("GreaterOrEqual" == relational_op_type)
            return parseGreaterOrEqualAttr(op_info, op_attr);
        else if ("Less" == relational_op_type)
            return parseLessAttr(op_info, op_attr);
        else if ("LessOrEqual" == relational_op_type)
            return parseLessOrEqualAttr(op_info, op_attr);
        else if ("NotEqual" == relational_op_type)
            return parseNotEqualAttr(op_info, op_attr);
        else if ("Equal" == relational_op_type)
            return parseEqualAttr(op_info, op_attr);
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported relational operations op type: {}", relational_op_type);
        return false;
    }

    Operation* RelationalOperationsCreator::onCreate(std::shared_ptr<Graph>& graph, const json& op_info)
    {
        RelationalOperationsOpAttr op_attr;
        std::string relational_op_type;
        if (!parseValue<std::string>(op_info, m_op_name, "relational_op_type", relational_op_type))
            return nullptr;
        if (!parseOpAttr(relational_op_type, op_info, op_attr))
            return nullptr;

        TIMVX_LOG_BASE_DATATYPE_ATTR(TIMVX_LEVEL_DEBUG, relational_op_type);
        if ("Greater" == relational_op_type)
        {
            return graph->CreateOperation<ops::Greater>().get();
        }
        else if ("GreaterOrEqual" == relational_op_type)
        {
            return graph->CreateOperation<ops::GreaterOrEqual>().get();
        }
        else if ("Less" == relational_op_type)
        {
            return graph->CreateOperation<ops::Less>().get();
        }
        else if ("LessOrEqual" == relational_op_type)
        {
            return graph->CreateOperation<ops::LessOrEqual>().get();
        }
        else if ("NotEqual" == relational_op_type)
        {
            return graph->CreateOperation<ops::NotEqual>().get();
        }
        else if ("Equal" == relational_op_type)
        {
            return graph->CreateOperation<ops::Equal>().get();
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported relational op type: {}", relational_op_type);
        return nullptr;
    }

    REGISTER_OP_CREATOR(RelationalOperationsCreator, RelationalOperations);

} // namespace TimVX