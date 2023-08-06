/***********************************
******  timvx_engine.cpp
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#include <mutex>
#include <iostream>
#include "timvx_engine.h"
#include "tensor_info.h"
#include "timvx_ops/op_creator.h"

namespace TimVX
{

    #define BITS_PER_BYTE 8
    TimVXEngine::TimVXEngine(const std::string& graph_name)
    {
        m_context.reset();
        m_graph.reset();
        m_graph_name = graph_name;
    }

    TimVXEngine::~TimVXEngine()
    {
        m_tensors_data.clear();
        m_tensors.clear();
        m_graph.reset();
        m_context.reset();
    }

    Tensor* TimVXEngine::getTensor(const std::string& tensor_name)
    {
        if (m_tensors.find(tensor_name) == m_tensors.end())
            return nullptr;
        return m_tensors[tensor_name].get();
    }

    std::vector<uint32_t> TimVXEngine::getTensorDims(const std::string& tensor_name)
    {
        std::vector<uint32_t> tensor_dims;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return tensor_dims;
        }
        auto tensor = m_tensors[tensor_name];
        return tensor->GetShape();
    }

    uint32_t TimVXEngine::getTypeBits(DataType type)
    {   
        switch( type )
        {
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::BOOL8:
                return 8;
            case DataType::INT16:
            case DataType::UINT16:
            case DataType::FLOAT16:
                return 16;
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::FLOAT32:
                return 32;
            default:
                return 0;
        }
    }  

    size_t TimVXEngine::getTensorSize(const std::string& tensor_name)
    {
        size_t sz;
        size_t i;
        size_t bits_num;
        size_t dim_num;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return 0;
        }
        sz = 0;
        auto tensor = m_tensors[tensor_name];
        dim_num = tensor->GetShape().size();
        auto shape = tensor->GetShape();
        auto type = tensor->GetDataType();
        if(0 == dim_num)
        {
            return sz;
        }
        bits_num = getTypeBits( type );
        if( bits_num < BITS_PER_BYTE )
        {
            if(shape[0] % 2 == 0)
            {
                sz = shape[0] / 2;
            }
            else
            {
                sz = shape[0] / 2 + shape[0] % 2;
            }
        }
        else
        {
            sz = shape[0] * bits_num / BITS_PER_BYTE;
        }
        for( i = 1; i < dim_num; i ++ )
        {
            sz *= shape[i];
        }
        return sz;
    }

    bool TimVXEngine::createTensor(const std::string& tensor_name, const json& tensor_info,
        const char *weight_data, const int weight_len)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_tensors.find(tensor_name) != m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "duplicate tensor name {} is provided, please check again!", tensor_name);
            return false;
        }
        TensorSpec tensor_spec;
        if (!TensorSpecConstruct::constructTensorspec(tensor_info, tensor_name, tensor_spec))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "construct tensor {} spec fail, please check again!", tensor_name);
            return false;
        }
        std::shared_ptr<Tensor> tensor;
        if (!weight_data || 0 == weight_len)
            tensor = m_graph->CreateTensor(tensor_spec);
        else
        {
            std::shared_ptr<char> data_array_ptr(new char[weight_len], [](char* data_array_ptr){delete [] data_array_ptr;});
            m_tensors_data[tensor_name] = data_array_ptr;
            memcpy((void*)data_array_ptr.get(), weight_data, weight_len);
            tensor = m_graph->CreateTensor(tensor_spec, (void*)data_array_ptr.get());
        }
        if (nullptr == tensor.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "create tensor {} fail!", tensor_name);
            return false;
        }
        m_tensors[tensor_name] = tensor;
        return true;
    }    

    bool TimVXEngine::copyDataFromTensor(const std::string& tensor_name, char* buffer_data, const int buffer_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "dst buffer data ptr is nullptr, when copy from tensor {}", tensor_name);
            return false;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        size_t total_tensor_size = getTensorSize(tensor_name);
        if (total_tensor_size != buffer_len)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} not equal to buffer data size:{}", 
                tensor_name, total_tensor_size, buffer_len);
            return false;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} is invalid!", tensor_name, total_tensor_size);
            return false;
        }
        return tensor->CopyDataFromTensor(buffer_data);
    }

    bool TimVXEngine::copyDataToTensor(const std::string& tensor_name, const char* buffer_data, 
        const int buffer_len)
    {
        if (nullptr == buffer_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "src buffer data ptr is nullptr, when copy to tensor {}", tensor_name);
            return false;
        }
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        int total_tensor_size = getTensorSize(tensor_name);
        if (total_tensor_size != buffer_len)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} not equal to buffer data size:{}",
                tensor_name, total_tensor_size, buffer_len);
            return false;
        }
        if (total_tensor_size <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} size:{} is invalid!", tensor_name, total_tensor_size);
            return false;
        }
        return tensor->CopyDataToTensor(buffer_data, buffer_len);
    }

    bool TimVXEngine::createOperation(const json& op_info)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (!op_info.contains("op_name") || !op_info["op_name"].is_string())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op_name item is not contained, or op_name is not string!");
            return false;
        }
        std::string op_name = op_info.at("op_name");
        if (!op_info.contains("op_type") || !op_info["op_type"].is_string())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s op_type item is not contained, or op_type is not string", op_name);
            return false;
        }
        if (!op_info.contains("op_attr") || !op_info["op_attr"].is_object())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s op_attr item is not contained, or op_attr is not dict", op_name);
            return false;
        }
        if (op_info.contains("rounding_policy") && !op_info["rounding_policy"].is_object())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s rounding_policy item is contained, but rounding_policy is not dict", op_name);
            return false;
        }
        if (m_operations.find(op_name) != m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op_name {} is duplicate", op_name);
            return false;
        }

        std::string op_type = op_info.at("op_type");
        OpCreator* op_creator = TimVXOp::getInstance()->getOpCreator(op_type);
        if (nullptr == op_creator)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} creator not find!", op_type);
            return false;
        }
        auto op_node = op_creator->onCreate(m_graph, op_info["op_attr"]);
        if (nullptr != op_node && op_info.contains("rounding_policy"))
        {
            json rounding_policy = op_info["rounding_policy"];
            OverflowPolicy overflow_policy_type = OverflowPolicy::SATURATE;
            RoundingPolicy rounding_policy_type = RoundingPolicy::RTNE;
            RoundType      round_type           = RoundType::FLOOR;
            uint32_t       accumulator_bits     = 0;
            op_creator->parseOverflowPolicyType(rounding_policy, op_name, "overflow_policy", overflow_policy_type, -1);
            op_creator->parseRoundingPolicyType(rounding_policy, op_name, "rounding_policy", rounding_policy_type, -1);
            op_creator->parseRoundType(rounding_policy, op_name, "down_scale_size_rounding", round_type, -1);
            op_creator->parseValue<uint32_t>(rounding_policy, op_name, "accumulator_bits", accumulator_bits, -1);
            op_node->SetRoundingPolicy(overflow_policy_type, rounding_policy_type, round_type, accumulator_bits);
        }
        if (nullptr != op_node)
        {
            m_operations[op_name] = op_node;
            m_op_info[op_name] = op_info;
            return true;
        }        
        return false;
    }

    bool TimVXEngine::bindInputs(const std::string& op_name, const std::vector<std::string>& input_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (input_list.size() <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "bind input list is empty!");
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        for (int i = 0; i < input_list.size(); i++)
        {
            std::string tensor_name = input_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
                return false;
            }
            input_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindInputs(input_tensors);
        return true;
    }

    bool TimVXEngine::bindOutputs(const std::string& op_name, const std::vector<std::string>& output_list)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (output_list.size() <= 0)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "bind output list is empty!");
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> output_tensors;
        for (int i = 0; i < output_list.size(); i++)
        {
            std::string tensor_name = output_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", tensor_name);
                return false;
            }
            output_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindOutputs(output_tensors);
        return true;
    }

    bool TimVXEngine::bindInput(const std::string& op_name, const std::string& input_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (m_tensors.find(input_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", input_name);
            return false;
        }
        std::shared_ptr<Tensor> input_tensor = m_tensors[input_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindInput(input_tensor);
        return true;
    }

    bool TimVXEngine::bindOutput(const std::string& op_name, const std::string& output_name)
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return false;
        }
        if (m_tensors.find(output_name) == m_tensors.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "tensor {} not exists!", output_name);
            return false;
        }
        std::shared_ptr<Tensor> out_tensor = m_tensors[output_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindOutput(out_tensor);
        return true;
    }
    
    json TimVXEngine::getOpInfo(const std::string& op_name)
    {
        json op_json;
        if (m_op_info.find(op_name) != m_op_info.end())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "op {} not exists!", op_name);
            return op_json;
        }
        return m_op_info[op_name];
    }

    bool TimVXEngine::createGraph()
    {
        m_context = tim::vx::Context::Create();
        if (nullptr == m_context.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "create context fail!");
            return false;
        }
        m_graph = m_context->CreateGraph();
        if (nullptr == m_graph.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "create graph fail!");
            m_context.reset();
            return false;
        }
        return true;
    }

    bool TimVXEngine::verifyGraph()
    {
        if (nullptr == m_context.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "context is invalid, please create context first!");
            return false;
        }
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        m_layout_infered = LayoutInference(m_graph, m_context);
        if (nullptr == m_layout_infered.first.get() || 0 == m_layout_infered.second.size())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph layout inference fail, please check ori graph!");
            return false;
        }
        return true;
    }

    bool TimVXEngine::compileGraph()
    {
        if (nullptr == m_graph.get() && nullptr == m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        if (nullptr != m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "layout infered graph compile ...");
            return m_layout_infered.first->Compile();
        }
        TIMVX_LOG(TIMVX_LEVEL_INFO, "compile graph ...");
        return m_graph->Compile();
    }

    bool TimVXEngine::runGraph()
    {
        if (m_graph.get() == nullptr)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }
        return m_graph->Run();
    }

    bool TimVXEngine::compileToBinary(std::vector<uint8_t>& nbg_buf, size_t& bin_size)
    {
        Graph* graph = nullptr;
        if (nullptr == m_graph.get() && nullptr == m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph is invalid, please create graph first!");
            return false;
        }

        graph = m_graph.get();
        if (nullptr != m_layout_infered.first.get())
        {
            TIMVX_LOG(TIMVX_LEVEL_INFO, "use layout infered graph compile to binary buffer ...");
            graph = m_layout_infered.first.get();
        }

        // call compile to binary
        bin_size = -1;
        if (false == graph->CompileToBinary(nullptr, &bin_size) || 0 >= bin_size)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph compile to get binary buffer size fail ...");
            return false;
        }

        // generate binary graph does't require input data
        TIMVX_LOG(TIMVX_LEVEL_INFO, "compie binary file size is {}", bin_size);
        nbg_buf.resize(bin_size);
        if (false == graph->CompileToBinary(nbg_buf.data(), &bin_size))
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "graph compile to binary buffer fail ...");
            return false;
        }
        return true;
    }

    std::string TimVXEngine::getGraphName()
    {
        return m_graph_name;
    }

} //namespace TimVX