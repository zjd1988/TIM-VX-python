/***********************************
******  timvx_c_api.h
******
******  Created by zhaojd on 2023/08/02.
***********************************/

#ifndef _TIMVX_C_API_H
#define _TIMVX_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifdef __arm__
typedef uint32_t TimvxContext;
#else
typedef uint64_t TimvxContext;
#endif

#define TIMVX_MAX_NUM_CHANNEL                    128     /* maximum channel number of graph input tensor. */
#define TIMVX_MAX_DIMS                           16      /* maximum dimension of tensor. */
#define TIMVX_MAX_NAME_LEN                       256     /* maximum name lenth of tensor. */


/*
    The query command for timvx_query
*/
typedef enum TimvxQueryCmd
{
    TIMVX_QUERY_IN_OUT_NUM = 0,                          /* query the number of input & output tensor. */
    TIMVX_QUERY_INPUT_ATTR,                              /* query the attribute of input tensor. */
    TIMVX_QUERY_OUTPUT_ATTR,                             /* query the attribute of output tensor. */

    TIMVX_QUERY_CMD_MAX
} TimvxQueryCmd;


/*
    the tensor data format.
*/
typedef enum TimvxTensorFormat 
{
    TIMVX_TENSOR_NCHW = 0,                               /* data format is NCHW. */
    TIMVX_TENSOR_NHWC,                                   /* data format is NHWC. */
    TIMVX_TENSOR_FORMAT_MAX
} TimvxTensorFormat;

/*
    the information for TIMVX_QUERY_IN_OUT_NUM.
*/
typedef struct TimvxInputOutputNum
{
    uint32_t n_input;                                   /* the number of input. */
    uint32_t n_output;                                  /* the number of output. */
} TimvxInputOutputNum;

/*
    the tensor data type.
*/
typedef enum TimvxTensorType
{
    TIMVX_TENSOR_FLOAT32 = 0,                            /* data type is float32. */
    TIMVX_TENSOR_FLOAT16,                                /* data type is float16. */
    TIMVX_TENSOR_INT8,                                   /* data type is int8. */
    TIMVX_TENSOR_UINT8,                                  /* data type is uint8. */
    TIMVX_TENSOR_INT16,                                  /* data type is int16. */

    TIMVX_TENSOR_TYPE_MAX
} TimvxTensorType;

/*
    the quantitative type.
*/
typedef enum TimvxTensorQntType
{
    TIMVX_TENSOR_QNT_NONE = 0,                           /* none. */
    TIMVX_TENSOR_QNT_DFP,                                /* dynamic fixed point. */
    TIMVX_TENSOR_QNT_AFFINE_ASYMMETRIC,                  /* asymmetric affine. */

    TIMVX_TENSOR_QNT_MAX
} TimvxTensorQntType;

/*
    the information for TIMVX_QUERY_INPUT_ATTR / TIMVX_QUERY_OUTPUT_ATTR.
*/
typedef struct TimvxTensorAttr
{
    uint32_t index;                                     /* input parameter, the index of input/output tensor */

    uint32_t n_dims;                                    /* the number of dimensions. */
    uint32_t dims[TIMVX_MAX_DIMS];                      /* the dimensions array. */
    char name[TIMVX_MAX_NAME_LEN];                      /* the name of tensor. */

    uint32_t n_elems;                                   /* the number of elements. */
    uint32_t size;                                      /* the bytes size of tensor. */

    TimvxTensorFormat fmt;                              /* the data format of tensor. */
    TimvxTensorType type;                               /* the data type of tensor. */
    TimvxTensorQntType qnt_type;                        /* the quantitative type of tensor. */
    uint32_t zp;                                        /* zero point for TIMVX_TENSOR_QNT_AFFINE_ASYMMETRIC. */
    float scale;                                        /* scale for TIMVX_TENSOR_QNT_AFFINE_ASYMMETRIC. */
} TimvxTensorAttr;

/*
    the input information for timvx_input_set.
*/
typedef struct TimvxInput 
{
    uint32_t index;                                     /* the input index. */
    void* buf;                                          /* the input buf for index. */
    uint32_t size;                                      /* the size of input buf. */
    uint8_t pass_through;                               /* pass through mode.
                                                        if TRUE, the buf data is passed directly to the input node of the timvx model
                                                                    without any conversion. the following variables do not need to be set.
                                                        if FALSE, the buf data is converted into an input consistent with the model
                                                                    according to the following type and fmt. so the following variables
                                                                    need to be set.*/
    TimvxTensorType type;                              /* the data type of input buf. */
    TimvxTensorFormat fmt;                             /* the data format of input buf.
                                                        currently the internal input format of NPU is NCHW by default.
                                                        so entering NCHW data can avoid the format conversion in the driver. */
} TimvxInput;

/*
    the output information for timvx_outputs_get.
*/
typedef struct TimvxOutput
{
    uint8_t want_float;                                 /* want transfer output data to float */
    uint8_t is_prealloc;                                /* whether buf is pre-allocated.
                                                        if TRUE, the following variables need to be set.
                                                        if FALSE, the following variables do not need to be set. */
    uint32_t index;                                     /* the output index. */
    void* buf;                                          /* the output buf for index.
                                                        when is_prealloc = FALSE and timvx_outputs_release called,
                                                        this buf pointer will be free and don't use it anymore. */
    uint32_t size;                                      /* the size of output buf. */
} TimvxOutput;

/*  timvx_init

    initial the context and load the timvx model.

    input:
        TimvxContext* context           the pointer of context handle.
        const char* model_para_path     model para file path.
        const char* model_weight_path   model weight file path.
        uint32_t flag                   extend flag, see the define of TIMVX_FLAG_XXX_XXX.
    return:
        int                             error code.
*/
int timvxInit(TimvxContext* context, const char* model_para_path, const char* model_weight_path, uint32_t flag);


/*  timvxDestroy

    unload the timvx model and destroy the context.

    input:
        TimvxContext context        the handle of context.
    return:
        int                         error code.
*/
int timvxDestroy(TimvxContext context);


/*  timvx_query

    query the information about model or others. see timvx_query_cmd.

    input:
        TimvxContext context        the handle of context.
        timvx_query_cmd cmd          the command of query.
        void* info                  the buffer point of information.
        uint32_t size               the size of information.
    return:
        int                         error code.
*/
int timvxQuery(TimvxContext context, TimvxQueryCmd cmd, void* info, uint32_t size);


/*  timvx_inputs_set

    set inputs information by input index of timvx model.
    inputs information see TimvxInput.

    input:
        TimvxContext context        the handle of context.
        uint32_t n_inputs           the number of inputs.
        TimvxInput inputs[]         the arrays of inputs information, see TimvxInput.
    return:
        int                         error code
*/
int timvxInputsSet(TimvxContext context, uint32_t n_inputs, TimvxInput inputs[]);


/*  timvx_run

    run the model to execute inference.

    input:
        TimvxContext context        the handle of context.
        timvx_run_extend* extend     the extend information of run.
    return:
        int                         error code.
*/
int timvxRun(TimvxContext context);


/*  timvx_outputs_get

    wait the inference to finish and get the outputs.
    this function will block until inference finish.
    the results will set to outputs[].

    input:
        TimvxContext context        the handle of context.
        uint32_t n_outputs          the number of outputs.
        TimvxOutput outputs[]       the arrays of output, see TimvxOutput.
    return:
        int                         error code.
*/
int timvxOutputsGet(TimvxContext context, uint32_t n_outputs, TimvxOutput outputs[]);


/*  timvx_outputs_release

    release the outputs that get by timvx_outputs_get.
    after called, the TimvxOutput[x].buf get from timvx_outputs_get will
    also be free when TimvxOutput[x].is_prealloc = FALSE.

    input:
        TimvxContext context        the handle of context.
        uint32_t n_ouputs           the number of outputs.
        TimvxOutput outputs[]       the arrays of output.
    return:
        int                         error code
*/
int timvxOutputsRelease(TimvxContext context, uint32_t n_ouputs, TimvxOutput outputs[]);


#ifdef __cplusplus
} //extern "C"
#endif

#endif  //_TIMVX_C_API_H