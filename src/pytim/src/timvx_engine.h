/***********************************
******  timvx_engine.h
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#pragma once
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/operation.h"
#include "tim/transform/layout_inference.h"
#include "common/timvx_log.h"
#include "nlohmann/json.hpp"
using namespace tim::vx;
using namespace tim::transform;
using namespace nlohmann;

namespace TimVX
{

    class TimVXEngine 
    {
    public:
        TimVXEngine(const std::string& graph_name);
        ~TimVXEngine();

        // tensor utils
        size_t getTensorSize(const std::string& tensor_name);
        std::vector<uint32_t> getTensorDims(const std::string& tensor_name);
        Tensor* getTensor(const std::string& tensor_name);
        bool createTensor(const std::string& tensor_name, const json& tensor_info, 
            const char *weight_data = nullptr, const int weight_len = 0);
        bool copyDataFromTensor(const std::string& tensor_name, char* buffer_data, const int buffer_len);
        bool copyDataToTensor(const std::string& tensor_name, const char* buffer_data, const int buffer_len);

        // operation utils
        bool createOperation(const json& op_info);
        json getOpInfo(const std::string& op_name);
        bool bindInputs(const std::string& op_name, const std::vector<std::string>& input_list);
        bool bindOutputs(const std::string& op_name, const std::vector<std::string>& output_list);
        bool bindInput(const std::string& op_name, const std::string& input_name);
        bool bindOutput(const std::string& op_name, const std::string& output_name);

        // graph uitls
        bool createGraph();
        bool verifyGraph();
        bool compileGraph();
        bool runGraph();
        bool compileToBinary(std::vector<uint8_t>& nbg_buf, size_t& bin_size);
        std::string getGraphName();

        // util func
        uint32_t getTypeBits(DataType type);

    private:
        // operation func
        // tensor names
        std::vector<std::string>                                            m_input_tensor_names;
        std::vector<std::string>                                            m_output_tensor_names;
        // tensors
        std::map<std::string, std::shared_ptr<Tensor>>                      m_tensors;
        // std::map<std::string, TensorSpec>              m_tensors_spec;
        std::map<std::string, std::shared_ptr<char>>                        m_tensors_data;
        // operation
        std::map<std::string, Operation*>                                   m_operations;
        std::map<std::string, json>                                         m_op_info;
        // engine context/graph/name
        std::shared_ptr<Context>                                            m_context;
        std::shared_ptr<Graph>                                              m_graph;
        std::string                                                         m_graph_name;
        // call verify graph get a new graph
        std::pair<std::shared_ptr<Graph>, 
            std::map<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>     m_layout_infered;
    };

} //namespace TimVX
