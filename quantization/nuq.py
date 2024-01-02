import os
import torch
import pickle
import argparse
import numpy as np
from sklearn.cluster import KMeans

from tqdm import tqdm
from transformers import LlamaForCausalLM

from squeezellm.model_parse import parse_model, get_module_names

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str,
    help='model weights to load', required=True
)
parser.add_argument(
    '--model_type', type=str, default=None,
    help='model type', choices=['llama', 'opt']
)
parser.add_argument(
    '--gradient', type=str,
    help='model gradients to load', required=True
)
parser.add_argument(
    '--bit', type=int, default=3,
    help='bitwidth', choices=[3, 4],
)
parser.add_argument(
    '--range', type=str, default=None,
    help='range of layers to quantize'
)
parser.add_argument(
    '--output_folder', type=str, required=None,
    help='path to dump the output'
)

def do_kmeans_plus(n_clusters: int, random_state: int, max_iter: int, X: torch.Tensor, sample_weight: torch.Tensor):
    DEV = torch.device('cuda:0')
    X = X.to(DEV)
    sample_weight = sample_weight.to(DEV)
    n_samples, n_features = X.shape
    torch.manual_seed(random_state)
    
    def dist(x, y):
        return (x - y).pow(2).sum(-1)
    
    # Kmeans++ Initialization
    centroids = torch.zeros((n_clusters, n_features)).to(DEV)
    centroids[0, :] = X[torch.randint(0, n_samples - 1, (1, 1), device=DEV), :]
    for i in range(1, n_clusters):
        next_centroid = torch.argmax(torch.min(dist(X.view(n_samples, 1, n_features), centroids[0:i, :].view(1, i, n_features)), dim=-1).values.view(n_samples)).view(1)
        centroids[i, :] = X[next_centroid, :]
    
    # Kmeans Algorithm
    for i in range(max_iter):
        assignment = torch.argmin(dist(X.view(n_samples, 1, n_features), centroids.view(1, n_clusters, n_features)), dim=-1).view(n_samples)
        for j in range(n_clusters):
            temp_weight = sample_weight[assignment == j]
            cluster_j = X[assignment == j, :] * temp_weight[:, None]
            if cluster_j.shape[0] > 0:
                centroids[j, :] = cluster_j.mean(0)
                centroids[j, :] /= torch.sum(temp_weight[:, None]) 
    
    assignments = torch.argmin(dist(X.view(n_samples, 1, n_features), centroids.view(1, n_clusters, n_features)), dim=-1).view(n_samples)
    return assignments.cpu().numpy(), centroids.cpu().numpy()

if __name__ == "__main__":
    args = parser.parse_args()

    # if model type is not explicitly given, infer from the model name
    model_type = args.model_type or parse_model(args.model)

    lut_folder = f"{args.output_folder}/lut"
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    if args.range:
        ranges = args.range.split(",")
        ranges = [int(r) for r in ranges]
        ran = list(range(ranges[0], ranges[1]))
    else:
        # Count number of layers based on the chunk item count in the model folder
        # You should not add/delete anything in the folder to make this work
        nlayers = len([f for f in os.listdir(args.model)])
        ran = list(range(nlayers))

    print(f"Quantizing layers {ran}")

    for l in ran:
        if ran is not None and l not in ran:
            print(f"Skipping layer {l}")
            continue

        lut_file_name = f"{lut_folder}/l{l}.pkl"
        print(lut_file_name)
        
        if os.path.exists(lut_file_name):
            print(f"Skipping layer {l}, file already exists at {lut_file_name}")
            continue

        print(f"Quantizing layer {l}")

        try:
            gradient_layer = torch.load(f"{args.gradient}/layer_{l}.pt")
        except:
            raise Exception(f"Needs chunked gradient file at {gradient_layer}")
            
        try:
            model_layer = torch.load(f"./{args.model}/layer_{l}.pt")
        except:
            raise Exception(f"Needs chunked model weight file at {model_layer}")

        config_per_layer = {}

        for name in tqdm(get_module_names(model_type)):
            g = gradient_layer[name].float()

            config_per_row = []
            module_weight = model_layer[name]
            _weights_np = module_weight.float()

            n_cluster = 2 ** args.bit

            # iterate over row
            for i in (range(module_weight.shape[0])):
                config_per_group = []
                weights_np_temp = _weights_np[i, :]
                weights_np = weights_np_temp.reshape(-1, 1)

                weight_mask = weights_np_temp != 0
                sample_weight = g[i, :]
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

                config_per_row.append(config_per_group)

                # assignments, centroids = do_kmeans_plus(
                #     n_clusters = n_cluster,
                #     random_state = 0,
                #     max_iter = 50,
                #     X = weights_np,
                #     sample_weight = sample_weight
                # )
                # print(assignments)

                # config_per_group.append(
                #     (centroids.reshape(-1), np.cast['byte'](assignments))
                # )
                # config_per_row.append(config_per_group)

            config_per_layer[name] = config_per_row

        # save parts
        with open(lut_file_name, "wb") as f:
            print(f"Saving layer lut to {lut_folder}/l{l}.pkl")
            pickle.dump(config_per_layer, f)
