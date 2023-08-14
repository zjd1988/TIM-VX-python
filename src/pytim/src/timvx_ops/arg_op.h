/***********************************
******  arg_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ArgCreator : public OpCreator
    {
    public:
        struct ArgOpAttr
        {
            int32_t axis;
        };

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseAxisAttr(std::string op_type, const json& op_info, ArgOpAttr& op_attr);
        bool parseOpAttr(std::string op_type, const json& op_info, ArgOpAttr& op_attr);

    private:
        std::string m_op_name = "Arg";
    };

} // namespace TimVX
