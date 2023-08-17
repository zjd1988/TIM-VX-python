/***********************************
******  addn_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class AddNCreator : public OpCreator
    {
    public:
        struct AddNOpAttr
        {
            uint32_t num_inputs;
        };

        AddNCreator(std::string op_name) : OpCreator(op_name)
        {
        }

        virtual Operation* onCreate(std::shared_ptr<Graph>& graph, const json& op_info) override;

    private:
        bool parseNumInputAttr(const json& op_info, AddNOpAttr& op_attr);
        bool parseOpAttr(const json& op_info, AddNOpAttr& op_attr);

    };

} // namespace TimVX
