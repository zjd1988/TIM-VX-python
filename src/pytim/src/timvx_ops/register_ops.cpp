/***********************************
******  register_ops.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include <mutex>

namespace TimVX
{

    extern void registerActivationOpCreator();
    extern void registerEltwiseOpCreator();
    extern void registerConv1dOpCreator();
    extern void registerConv2dOpCreator();
    extern void registerDeConv1dOpCreator();
    extern void registerDeConv2dOpCreator();
    extern void registerGroupedConv2dOpCreator();
    extern void registerFullyConnectedOpCreator();
    extern void registerSoftmaxOpCreator();
    extern void registerPool2dOpCreator();
    extern void registerReshapeOpCreator();
    extern void registerResizeOpCreator();
    extern void registerTransposeOpCreator();
    extern void registerConcatOpCreator();
    extern void registerDataConvertOpCreator();
    extern void registerArgOpCreator();
    extern void registerNBGOpCreator();
    extern void registerBatch2SpaceOpCreator();
    extern void registerAddNOpCreator();
    extern void registerClipOpCreator();
    extern void registerDepth2SpaceOpCreator();

    static std::once_flag s_flag;
    void registerOps()
    {
        std::call_once(s_flag, [&]() 
        {
            registerActivationOpCreator();
            registerEltwiseOpCreator();
            registerConv1dOpCreator();
            registerConv2dOpCreator();
            registerDeConv1dOpCreator();
            registerDeConv2dOpCreator();
            registerGroupedConv2dOpCreator();
            registerFullyConnectedOpCreator();
            registerSoftmaxOpCreator();
            registerPool2dOpCreator();
            registerReshapeOpCreator();
            registerResizeOpCreator();
            registerTransposeOpCreator();
            registerConcatOpCreator();
            registerDataConvertOpCreator();
            registerArgOpCreator();
            registerNBGOpCreator();
            registerBatch2SpaceOpCreator();
            registerAddNOpCreator();
            registerClipOpCreator();
            registerDepth2SpaceOpCreator();
        });
    }

} // namespace TimVX