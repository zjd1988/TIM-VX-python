/***********************************
******  model_benchmark.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include "tool_utils.h"
#include "timvx_model.h"

using namespace TimVX;

int main(int argc, char* argv[])
{
    CmdLineArgOption cmd_option;
    if (0 != parseModelBenchmarkOption(argc, argv, cmd_option))
        return -1;

    std::shared_ptr<TimVXModel> model(new TimVXModel(cmd_option));
    if (nullptr == model.get())
        return -1;
    
    return model->modelBenchmark();
}
