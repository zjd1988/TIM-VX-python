/***********************************
******  register_ops.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include <mutex>

namespace TimVX
{

    extern void registerActivationOpCreator();
    extern void registerAddNOpCreator();
    extern void registerArgOpCreator();
    extern void registerBatch2SpaceOpCreator();
    extern void registerBatchNormCreator();
    extern void registerClipOpCreator();
    extern void registerConcatOpCreator();
    extern void registerConv1dOpCreator();
    extern void registerConv2dOpCreator();
    extern void registerDataConvertOpCreator();
    extern void registerDeConv1dOpCreator();
    extern void registerDeConv2dOpCreator();
    extern void registerDepth2SpaceOpCreator();
    extern void registerDropoutOpCreator();
    extern void registerEltwiseOpCreator();
    extern void registerFullyConnectedOpCreator();
    extern void registerGatherOpCreator();
    extern void registerGatherNdOpCreator();
    extern void registerGroupedConv2dOpCreator();
    extern void registerInstanceNormalizationOpCreator();
    extern void registerL2NormalizationOpCreator();
    extern void registerLayerNormalizationOpCreator();
    extern void registerNBGOpCreator();
    extern void registerPool2dOpCreator();
    extern void registerReshapeOpCreator();
    extern void registerResizeOpCreator();
    extern void registerSoftmaxOpCreator();
    extern void registerTransposeOpCreator();

    static std::once_flag s_flag;
    void registerOps()
    {
        std::call_once(s_flag, [&]() 
        {
            registerActivationOpCreator();
            registerAddNOpCreator();
            registerArgOpCreator();
            registerBatch2SpaceOpCreator();
            registerBatchNormCreator();
            registerClipOpCreator();
            registerConcatOpCreator();
            registerConv1dOpCreator();
            registerConv2dOpCreator();
            registerDataConvertOpCreator();
            registerDeConv1dOpCreator();
            registerDeConv2dOpCreator();
            registerDepth2SpaceOpCreator();
            registerDropoutOpCreator();
            registerEltwiseOpCreator();
            registerFullyConnectedOpCreator();
            registerGatherOpCreator();
            registerGatherNdOpCreator();
            registerGroupedConv2dOpCreator();
            registerInstanceNormalizationOpCreator();
            registerL2NormalizationOpCreator();
            registerLayerNormalizationOpCreator();
            registerNBGOpCreator();
            registerPool2dOpCreator();
            registerReshapeOpCreator();
            registerResizeOpCreator();
            registerSoftmaxOpCreator();
            registerTransposeOpCreator();
        });
    }

} // namespace TimVX