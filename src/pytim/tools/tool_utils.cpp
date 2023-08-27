/***********************************
******  tool_utils.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include <random>
#include <numeric>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "npy.hpp"
#include "tool_utils.h"
#include "common/timvx_log.h"

namespace TimVX
{

    int parseModelInferOption(int argc, char* argv[], CmdLineArgOption& arg_opt)
    {
        // 1 init arg options
        cxxopts::Options arg_options("model_infer", "model inference test");
        arg_options.add_options()
            // model weight file path
            ("weight", "Model weight file path", cxxopts::value<std::string>())
            // model para file path
            ("para", "Model para file path", cxxopts::value<std::string>())
            // model input data file
            ("input", "Input data file", cxxopts::value<std::string>()->default_value(""))
            // model output data file
            ("output", "Output tesnor data to file", cxxopts::value<bool>()->default_value("false"))
            // log level, default is info level
            ("log_level", "log level", cxxopts::value<int>()->default_value("2"))
            // help
            ("help", "Print usage");
        arg_options.allow_unrecognised_options();

        // 2 parse arg
        auto parse_result = arg_options.parse(argc, argv);

        // 3 check help arg
        arg_opt.help_flag = false;
        if (parse_result.count("help"))
        {
            arg_opt.help_flag = true;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }

        // 4 check unmatch arg
        const std::vector<std::string>& unmatch = parse_result.unmatched();
        if (parse_result.unmatched().size() > 0)
        {
            std::cout << "contain unsupported options:" << std::endl;
            for (int i = 0; i < unmatch.size(); i++)
                std::cout << unmatch[i] << std::endl;
            return -1;
        }

        // 5 chcek model/para file arg
        arg_opt.weight_file = "";
        if (0 == parse_result.count("weight"))
        {
            std::cout << "model weight file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.weight_file = parse_result["weight"].as<std::string>();

        arg_opt.para_file = "";
        {
            std::cout << "model para file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.para_file = parse_result["para"].as<std::string>();

        // 6 check input file and output flag arg
        arg_opt.input_file = "";
        arg_opt.output_flag = false;
        if (0 != parse_result.count("input"))
            arg_opt.input_file = parse_result["input"].as<std::string>();
        
        if (0 != parse_result.count("output"))
            arg_opt.output_flag = parse_result["output"].as<bool>();

        // 7 check log arg
        // LOG_LEVEL_DEBUG = 1,
        // LOG_LEVEL_INFO,
        // LOG_LEVEL_WARN,
        // LOG_LEVEL_ERROR
        arg_opt.log_level = 1;
        arg_opt.log_level = parse_result["log_level"].as<int>();

        return 0;
    }

    int parseModelBenchmarkOption(int argc, char* argv[], CmdLineArgOption& arg_opt)
    {
        // 1 init arg options
        cxxopts::Options arg_options("model_benchmark", "model benchmark test");
        arg_options.add_options()
            // model weight file path
            ("weight", "Model weight file path", cxxopts::value<std::string>())
            // model para file path
            ("para", "Model para file path", cxxopts::value<std::string>())
            // model input data file
            ("input", "Input data file", cxxopts::value<std::string>()->default_value(""))
            // log level, default is info level
            ("log_level", "log level", cxxopts::value<int>()->default_value("2"))
            // benchmark model infer times
            ("times", "Model benchmark infer times", cxxopts::value<int>()->default_value("1"))
            // help
            ("help", "Print usage");
        arg_options.allow_unrecognised_options();

        // 2 parse arg
        auto parse_result = arg_options.parse(argc, argv);

        // 3 check help arg
        arg_opt.help_flag = false;
        if (parse_result.count("help"))
        {
            arg_opt.help_flag = true;
            std::cout << arg_options.help() << std::endl;
            return 0;
        }

        // 4 check unmatch arg
        const std::vector<std::string>& unmatch = parse_result.unmatched();
        if (parse_result.unmatched().size() > 0)
        {
            std::cout << "contain unsupported options:" << std::endl;
            for (int i = 0; i < unmatch.size(); i++)
                std::cout << unmatch[i] << std::endl;
            return -1;
        }

        // 5 chcek model weight/para file arg
        arg_opt.weight_file = "";
        if (0 == parse_result.count("weight"))
        {
            std::cout << "model weight file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.weight_file = parse_result["weight"].as<std::string>();

        arg_opt.para_file = "";
        if (0 == parse_result.count("para"))
        {
            std::cout << "model para file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.para_file = parse_result["para"].as<std::string>();

        // 6 check input file arg
        arg_opt.input_file = "";
        if (0 == parse_result.count("input"))
        {
            std::cout << "need specify input files" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.input_file = parse_result["input"].as<std::string>();

        // 7 check log arg
        // LOG_LEVEL_DEBUG = 1,
        // LOG_LEVEL_INFO,
        // LOG_LEVEL_WARN,
        // LOG_LEVEL_ERROR
        arg_opt.log_level = 1;
        arg_opt.log_level = parse_result["log_level"].as<int>();

        // 8 benchmark flag and benchmark times
        if (parse_result.count("times"))
            arg_opt.benchmark_times = parse_result["times"].as<int>();

        return 0;
    }

    int parseModelCompileOption(int argc, char* argv[], CmdLineArgOption& arg_opt)
    {
        // 1 init arg options
        cxxopts::Options arg_options("model_compile", "model compile test");
        arg_options.add_options()
            // input model weight file path
            ("input_weight", "Input model weight file path", cxxopts::value<std::string>())
            // input model para file path
            ("input_para", "Input model para file path", cxxopts::value<std::string>())
            // output compiled model weight file path
            ("output_weight", "Output model weight file path", cxxopts::value<std::string>())
            // output compiled model para file path
            ("output_para", "Output model para file path", cxxopts::value<std::string>())
            // log level, default is info level
            ("log_level", "log level", cxxopts::value<int>()->default_value("2"))
            // help
            ("help", "Print usage");
        arg_options.allow_unrecognised_options();

        // 2 parse arg
        auto parse_result = arg_options.parse(argc, argv);

        // 3 check help arg
        arg_opt.help_flag = false;
        if (parse_result.count("help"))
        {
            arg_opt.help_flag = true;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }

        // 4 check unmatch arg
        const std::vector<std::string>& unmatch = parse_result.unmatched();
        if (parse_result.unmatched().size() > 0)
        {
            std::cout << "contain unsupported options:" << std::endl;
            for (int i = 0; i < unmatch.size(); i++)
                std::cout << unmatch[i] << std::endl;
            return -1;
        }

        // 5 check input model weight/para file arg
        arg_opt.weight_file = "";
        if (0 == parse_result.count("input_weight"))
        {
            std::cout << "input model file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.weight_file = parse_result["input_weight"].as<std::string>();

        arg_opt.para_file = "";
        if (0 == parse_result.count("input_para"))
        {
            std::cout << "input model para file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.para_file = parse_result["input_para"].as<std::string>();

        // 6 check output model weight/para file arg
        arg_opt.compile_weight_file = "";
        if (0 == parse_result.count("output_weight"))
        {
            std::cout << "output model weight file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.compile_weight_file = parse_result["output_weight"].as<std::string>();

        arg_opt.compile_para_file = "";
        if (0 == parse_result.count("output_para"))
        {
            std::cout << "output model para file should be set" << std::endl;
            std::cout << arg_options.help() << std::endl;
            return -1;
        }
        arg_opt.compile_para_file = parse_result["output_para"].as<std::string>();

        // 7 check log arg
        // LOG_LEVEL_DEBUG = 1,
        // LOG_LEVEL_INFO,
        // LOG_LEVEL_WARN,
        // LOG_LEVEL_ERROR
        arg_opt.log_level = 1;
        arg_opt.log_level = parse_result["log_level"].as<int>();

        return 0;
    }

    ModelTensorData::ModelTensorData(const char* file_name)
    {
        if (nullptr == file_name)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "file name is nullptr");
            return;
        }
        std::string file_path = file_name;
        std::string suffix_str = file_path.substr(file_path.find_last_of('.') + 1);
        int load_ret = -1;
        if (suffix_str == "jpg" || suffix_str == "bmp")
        {
            load_ret = loadDataFromStb(file_name);
        }
        else if (suffix_str == "npy" || suffix_str == "bin")
        {
            load_ret = loadDataFromNpy(file_name);
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "{}'s suffix {} not supported", file_path, suffix_str);
        m_tensor_valid = (0 == load_ret);
    }

    ModelTensorData::ModelTensorData(std::vector<int> shape, TimvxTensorType type, 
        TimvxTensorFormat format, bool random_init)
    {
        m_shape = shape;
        m_type = type;
        m_format = format;
        m_shape = shape;
        m_data_len = tensorElementCount() * tensorElementSize();
        if (m_data_len)
            m_data = (uint8_t*)(new char[m_data_len]);
        m_tensor_valid = (nullptr != m_data) ? true : false;
        if (random_init && nullptr != m_data)
            randomInitData();
    }

    ModelTensorData::~ModelTensorData()
    {
        if (nullptr != m_data)
            delete[] m_data;
    }

    void ModelTensorData::randomInitData()
    {
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<> distr_f(0.0f, 1.0f);
        std::uniform_int_distribution<> distr_i(0, 127);
        int element_count = tensorElementCount();
        if (TIMVX_TENSOR_FLOAT32 == m_type)
        {
            float* tensor_data = (float*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_f(eng);
            }
        }
        else if (TIMVX_TENSOR_INT8 == m_type)
        {
            int8_t* tensor_data = (int8_t*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_i(eng);
            }
        }
        else if (TIMVX_TENSOR_UINT8 == m_type)
        {
            uint8_t* tensor_data = (uint8_t*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_i(eng);
            }
        }
        else if (TIMVX_TENSOR_INT16 == m_type)
        {
            int16_t* tensor_data = (int16_t*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_i(eng);
            }
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_WARN, "not support random init {} type", getTypeString(m_type));
    }

    int ModelTensorData::tensorElementCount()
    {
        if (m_shape.size())
            return 0;
        else
            return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());
    }

    int ModelTensorData::tensorElementSize()
    {
        int element_size = 1;
        switch (m_type)
        {
            case TIMVX_TENSOR_FLOAT32:
                element_size = sizeof(float);
                break;
            case TIMVX_TENSOR_INT8:
                element_size = sizeof(char);
                break;
            case TIMVX_TENSOR_UINT8:
                element_size = sizeof(unsigned char);
                break;
            case TIMVX_TENSOR_INT16:
                element_size = sizeof(short);
                break;
            default:
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "cannot get {}'s element size", getTypeString(m_type));
                element_size = 0;
                break;
        }
        return element_size;
    }

    void* ModelTensorData::tensorData()
    {
        if (!m_tensor_valid)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "current tensor is invalid");
            return nullptr;
        }
        return (void*)m_data;
    }

    int ModelTensorData::loadDataFromStb(const char* file_name)
    {
        int height = 0;
        int width = 0;
        int channel = 0;

        unsigned char *image_data = stbi_load(file_name, &width, &height, &channel, 3);
        if (nullptr == image_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "stb load data from {} failed!", file_name);
            return -1;
        }
        TIMVX_LOG(TIMVX_LEVEL_INFO, "load image from {}, h*w*c={}*{}*{}", file_name, height, width, channel);
        // stb load image as rgb, need to convert rgb to bgr
        uint8_t* bgr_data = (uint8_t*)(new char[height * width * channel]);
        if (nullptr == bgr_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory for bgr data fail when load from {}", file_name);
            return -1;
        }
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < channel; k++)
                {
                    int src_index = i * width * channel + j * channel + k;
                    int dst_index = i * width * channel + j * channel + channel - k - 1;
                    bgr_data[dst_index] = image_data[src_index];
                }
            }
        }
        stbi_image_free(image_data);
        
        std::vector<int> image_shape = {1, height, width, channel};
        m_shape = image_shape;
        m_data = (uint8_t*)bgr_data;
        m_type = TIMVX_TENSOR_UINT8;
        m_format = TIMVX_TENSOR_NHWC;
        m_data_len = height * width * channel;
        return 0;
    }

    int ModelTensorData::loadDataFromNpy(const char* file_name)
    {
        std::ifstream stream(file_name, std::ifstream::binary);
        std::string header_str = npy::read_header(stream);
        npy::header_t npy_header = npy::parse_header(header_str);
        if (npy_header.fortran_order)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "currently not support fortran order npy file");
            return -1;
        }
        std::type_index type_index = std::type_index(typeid(unsigned char));
        std::string type_str = npy_header.dtype.str();
        int item_size = sizeof(char);
        for (auto it = npy::dtype_map.begin(); it != npy::dtype_map.end(); it++)
        {
            if (0 == it->second.str().compare(type_str))
            {
                type_index = it->first;
                item_size = it->second.itemsize;
            }
        }
        TimvxTensorType tensor_dtype = TIMVX_TENSOR_UINT8;
        if (std::type_index(typeid(float)) == type_index)
            tensor_dtype = TIMVX_TENSOR_FLOAT32;
        else if (std::type_index(typeid(short)) == type_index)
            tensor_dtype = TIMVX_TENSOR_INT16;
        else if (std::type_index(typeid(char)) == type_index)
            tensor_dtype = TIMVX_TENSOR_INT8;
        else if (std::type_index(typeid(unsigned char)) == type_index)
            tensor_dtype = TIMVX_TENSOR_UINT8;
        else
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported npy data type {}", type_str);
            return -1;
        }
        auto element_count = static_cast<size_t>(npy::comp_size(npy_header.shape));
        int tensor_len = item_size * element_count;
        uint8_t* tensor_data = (uint8_t*)(new char[tensor_len]);
        if (nullptr == tensor_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory for tensor data fail when load from {}", file_name);
            return -1;
        }
        // read the data
        stream.read((char*)tensor_data, tensor_len);
        stream.close();
        // compute the data size based on the shape
        std::vector<int> tensor_shape(npy_header.shape.begin(), npy_header.shape.end());
        m_shape = tensor_shape;
        m_type = tensor_dtype;
        m_format = TIMVX_TENSOR_NCHW;
        m_data = (uint8_t*)tensor_data;
        m_data_len = tensor_len;
        return 0;
    }

} // namespace TimVX