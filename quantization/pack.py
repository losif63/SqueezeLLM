import time
import pickle

import torch
import torch.nn as nn

import transformers
from squeezellm.modelutils import *
from squeezellm.quant import *

import json
import os

@torch.no_grad()
def llama_sequential(model, folder, include_sparse, updated_format):
    layers = model.model.layers

    if not updated_format and include_sparse:
        # list of num layers, each item is a dictionary with the keys ["q", "k", "v", "o", "gate", "up", "down"]
        outlier_list = pickle.load(open(f"{folder}/outliers.pkl", "rb"))

    quantizers = {}
    for i in range(len(layers)):
        with open(f"{folder}/lut/l{i}.pkl", "rb") as f:
            # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
            # each of which is a list of #group lists, (here, it's always 1)
            # and each of them are a tuple (centroids, indices)
            lut_layer = pickle.load(f)
            # lut_layer is dictionary from nuq.py

        if not updated_format and include_sparse:
            outlier_list_layer = outlier_list[i]
        elif include_sparse:
            with open(f"{folder}/outliers/l{i}.pkl", "rb") as f:
                # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
                # each of which is a list of #group lists, (here, it's always 1)
                # and each of them are a tuple (centroids, indices)
                outlier_list_layer = pickle.load(f)

        sequential_lut = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
        sequential_lut_real_name = {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'gate': 'mlp.gate_proj',
            'up': 'mlp.up_proj',
            'down': 'mlp.down_proj'
        }

        outlier_index = {
            'q': 0,
            'k': 1,
            'v': 2,
            'o': 3,
            'gate': 4,
            'up': 5,
            'down': 6
        }

        for s in sequential_lut:
            lut = lut_layer[s]
            # lut_layer is a dictionary
            # lut is a list of (LUT, labels)!!
            if not updated_format and include_sparse:
                outliers = outlier_list_layer[s]
            elif include_sparse:
                idx = outlier_index[s]
                outliers = outlier_list_layer[idx]
            else:
                outliers = None
            name = sequential_lut_real_name[s]
            quantizers['model.layers.%d.%s' % (i, name)] = [lut,outliers]
            # lut, the list of (LUT, lables) is stored in quantizers dictionary!!
            # outliers is None because no include_sparse yet

    return quantizers


def llama_pack(model, quantizers, wbits, include_sparse):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant_lut(model, quantizers, wbits, include_sparse=include_sparse)

    qlayers = find_layers(model, [QuantLinearLUT])
    print('Packing ...')
    sparsedict = {}

    for name in qlayers:
        print(name)
        lookup_table = quantizers[name]
        layers[name].cpu()
        qlayers[name].pack(layers[name], lookup_table, include_sparse)
        if include_sparse:
            sparsedict[name] = qlayers[name].vals.shape[-1]

    print('Packing Done.')
    return model, sparsedict


if __name__ == '__main__':
    import argparse
    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--save', type=str, required=True,
        help='Save quantized checkpoint under this name.'
    )

    #sparse args
    parser.add_argument(
        '--folder', type=str, default='',
        help='Path to folder containing luts and outliers.'
    )
    parser.add_argument(
        '--include_sparse', action='store_true',
        help='Whether loaded checkpoint has sparse matrix.'
    )
    parser.add_argument(
        '--updated-format', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )

    args = parser.parse_args()
    assert not args.include_sparse, "Sparse not supported yet"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype='auto'
    )
    model.eval()

    print('Running llama_sequential')
    tick = time.time()
    quantizers = llama_sequential(
        model=model, 
        folder=args.folder, 
        include_sparse=args.include_sparse, 
        updated_format=True,
    )
    print("llama_sequential Done:", time.time() - tick)

    print("Running llama_pack")
    tick = time.time()
    model, numvals = llama_pack(
        model=model, 
        quantizers=quantizers, 
        wbits=args.wbits, 
        include_sparse=args.include_sparse, 
    )
    print("llama_pack Done:", time.time() - tick)

    model_dict = model.state_dict()

    if args.include_sparse:
        #need to merge in sparse dict
        for k,v in numvals.items():
          model_dict['sparse_threshold.'+k] = v

    #save model
    torch.save(model_dict, args.save)

    #get directory to save quant_config
    directory = os.path.dirname(args.save)
    data = {
        "wbits": args.wbits
    }
    output_fn = os.path.join(directory, "quant_config.json")

    #save quant_config
    with open(output_fn, 'w') as f:
        json.dump(data, f, indent = 4)
