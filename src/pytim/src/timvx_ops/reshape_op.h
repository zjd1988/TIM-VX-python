/***********************************
******  reshape_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "timvx_ops/op_creator.h"
using namespace tim::vx;
using namespace std;

namespace TimVX
{

    class ReshapeCreator : public OpCreator
    {
    public:
        struct ReshapeOpAttr
        {
            std::vector<uint32_t> size;
        };

        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool parse_op_attr(const py::dict &op_info, ReshapeOpAttr &op_attr);

    private:
        std::string m_op_name = "Reshape";
    };

} // namespace TimVX
