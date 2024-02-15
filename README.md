# SqueezeLLM-Modified

## Introduction
This repository implements a modified version of SqueezeLLM. To be more specific, it implements row-based quantization of weights, instead of the previous column-based quantization method.

## Explanation of modified code

The following sections give a brief explanation about my contribution to the modification of SqueezeLLM. For more information, please take a look at the github commit logs as every single line of code that I've touched on is recorded there.

### quantization/nuq.py

This file is responsible for the quantization of LLM weight matrices. To be more specific, the following for loop in lines 88 ~ 126 performs kmeans clustering on each of the columns in each of the weight matrices.

```python
for name in tqdm(get_module_names(model_type)):
    g = gradient_layer[name].float()

    config_per_col = []
    module_weight = model_layer[name]
    _weights_np = module_weight.float()

    n_cluster = 2 ** args.bit

    # iterate over columns
    for i in (range(module_weight.shape[1])):
        config_per_group = []
        weights_np_temp = _weights_np[:, i]
        weights_np = weights_np_temp.reshape(-1, 1)

        weight_mask = weights_np_temp != 0
        sample_weight = g[:, i]
        sample_weight = sample_weight * weight_mask

        if np.sum(sample_weight.numpy()) == 0:
            sample_weight = torch.from_numpy(np.ones_like(sample_weight))

        kmeans = KMeans(
            n_clusters=n_cluster, 
            random_state=0, 
            n_init="auto", 
            max_iter=50,
        ).fit(
            weights_np, 
            sample_weight=sample_weight,
        )

        config_per_group.append(
            (kmeans.cluster_centers_.reshape(-1), np.cast['byte'](kmeans.labels_))
        )

        config_per_col.append(config_per_group)

    config_per_layer[name] = config_per_col
```
Note that in the GPU, matrix multiplication happens in the form of y=xW^T. Therefore, the following column-based quantization is equivalent to row-based quantization of W^T.

For a more detailed comparison between the previous and current versions, check out the [original source code](https://github.com/losif63/SqueezeLLM_modified/blob/9e7aaed0f64becdec8e5eec7916efc9d2ae87f00/quantization/nuq.py).


### squeezellm/quant_cuda_kernel.cu

This is the cuda kernel file that is responsible for the matrix multiplication between quantized weight matrices and activations. As of current, FP32 and FP16 precisions are natively supported by CUDA itself, and a custom type is implemented by myself. 

The cuda kernel definitions can be found in [lines 50~306](https://github.com/losif63/SqueezeLLM_modified/blob/main/squeezellm/quant_cuda_kernel.cu?plain=1#L50-L306).

For the FP32 precision, the kernel implementations are located in [lines 920~1354](https://github.com/losif63/SqueezeLLM_modified/blob/main/squeezellm/quant_cuda_kernel.cu?plain=1#L920-L1354). Note that for these kernels, the only difference between these kernels and the original ones is that the lookup table ```deq2``` is set & accessed according to the modified row-based quantization method. 

For the FP16 precision, the kernel implementations are located in [lines 1967~2404](https://github.com/losif63/SqueezeLLM_modified/blob/main/squeezellm/quant_cuda_kernel.cu?plain=1#L1967-L2404). Multiplication is done in FP16 precision, and accumulation is done in FP32 precision. Note that the MAC instruction is handled by the custom macro `HMUL_FLOATS`, created by me.

For the custom precision, the kernel implementations are located in [lines 3017~3975](https://github.com/losif63/SqueezeLLM_modified/blob/main/squeezellm/quant_cuda_kernel.cu?plain=1#L3017-L3975). Note that round-to-even, conversion to custom type, and fadd / fmul operations between custom types were all newly implemented by me. You can manually adjust the desired custom precision by changing the macro definitions in lines 3523 ~ 3527. The macros EXP1 and MAN1 are the number of exponent and mantissa bits used for multiplication, and the macros EXP2 and MAN2 are the number of exponent and mantissa bits used for accumulation. Please modify these numbers freely according to your needs. 

However, please note that after chaning the number of exponent & mantissa bits, you must re-compile the cuda code. Otherwise, your changes will not be built into the executable file. Compilation can be easily done by the following set of commands.

```bash
cd squeezellm
python setup_cuda.py install
cd ..
```

### How to use

Since SqueezeLLM isn't exactly optimized for row-based quantization, the modified code is much slower than its original counterpart. Therefore, we use [princeton-nlp/Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B), which is a version of LLaMA small enough to use in this experiment. Evaluation of the perplexity values can be done by using the following command.

```bash
python llama.py princeton-nlp/Sheared-LLaMA-1.3B c4 --wbits 4 --load /home/sslunder7/SqueezeLLM-params/PACKED_CKPT_PATH_NEW/sheared-llama-1b-w4-s0.pt --precision fp16 --eval
```

For the `--precision` parameter, the possible options are "fp32", "fp16", or "custom". If the `--precision` parameter is not specified, then the fp32 precision will be automatically selected.

As mentioned before, when using the custom type, please make sure to re-compile the cuda code.
