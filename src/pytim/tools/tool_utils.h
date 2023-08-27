/***********************************
******  tool_utils.h
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#pragma once
#include "cxxopts.hpp"
#include "common/non_copyable.h"
#include "timvx_c_api.h"

namespace TimVX
{

    typedef struct CmdLineArgOption
    {
        // model infer/compie
        std::string                    para_file;
        std::string                    weight_file;
        std::string                    input_file;
        bool                           output_flag;
        bool                           pass_through;
        bool                           want_float;
        // model compile
        std::string                    compile_para_file;
        std::string                    compile_weight_file;
        // model benchmark
        int                            benchmark_times;
        // common
        std::string                    log_path;
        int                            log_level;
        bool                           help_flag;
    } CmdLineArgOption;

    // command line arg option parse
    int parseModelInferOption(int argc, char* argv[], CmdLineArgOption& arg_opt);
    int parseModelBenchmarkOption(int argc, char* argv[], CmdLineArgOption& arg_opt);
    int parseModelCompileOption(int argc, char* argv[], CmdLineArgOption& arg_opt);

    class ModelTensorData : public NonCopyable
    {
    public:
        ModelTensorData(const char* file_name);
        ModelTensorData(std::vector<int> shape, TimvxTensorType type, 
            TimvxTensorFormat format, bool random_init=true);
        ~ModelTensorData();

    int tensorLength() { return m_data_len; };
    void* tensorData();
    int tensorElementCount();
    int tensorElementSize();
    TimvxTensorType tensorType() { return m_type; }
    TimvxTensorFormat tensorFormat() { return m_format; }

    private:
        // radom init data
        void randomInitData();
        // load data from image/npy
        int loadDataFromStb(const char* file_name);
        int loadDataFromNpy(const char* file_name);

        // save data to npy file
        int saveDataByNpy(const char* file_name);
    
    private:
        bool                           m_tensor_valid = false;
        bool                           m_own_flag = true;
        uint8_t*                       m_data = nullptr;
        int                            m_data_len = -1;
        TimvxTensorType                m_type;
        TimvxTensorFormat              m_format;
        std::vector<int>               m_shape;
    };

} // namespace TimVX
