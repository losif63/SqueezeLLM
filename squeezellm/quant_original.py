import numpy as np
import torch
import torch.nn as nn
import math
import quant_cuda

def round_to_nearest_pole_sim(w, poles):
    """
    w: weight values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    for i, c in enumerate(poles):
        aug += (idx == i) * c
    return aug


# drop-in layer replacement class
class QuantLinearLUT(nn.Module):
    def __init__(self, bits, infeatures, outfeatures, bias, include_sparse=False, numvals=0, topX=10):
        super().__init__()
        if bits not in [3,4]:
            raise NotImplementedError("Only 3 and 4 bits is supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        # dtype is int32, but we only need infeatures * self.bits amount of bits!
        # Therefore, the number of int32 is infeatures * self.bits // 32!!
        # Let N be infeatures, and M be outfeatures
        # The multiplication order is X * W^T!!
        # Therefore, qweight is stored in dimensionality (N, M)
        # However, normally the weights are stored as (M, N)
        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        if bias:
            self.include_bias = True
            self.register_buffer('bias', torch.zeros((outfeatures)))
        else:
            self.include_bias = False
            self.bias = None
        # Note that it is 2^self.bits, not 2 * self.bits
        # For each row (there are outfeature rows), there are k=2^self.bits cluster centers
        # Basically, this stores the cluster centers for each row!!
        self.register_buffer('lookup_table', torch.zeros((outfeatures, 2**self.bits), dtype=torch.float32))

        self.include_sparse = include_sparse
        self.numvals = numvals
        self.topX = topX
        # Note for N input features and M output features, the matrix stored is (M, N)
        # M (outfeatures) rows and N (infeatures) columns!
        if numvals > 0:
            self.register_buffer('rows', torch.zeros(outfeatures+1, dtype=torch.int32))
            self.register_buffer('cols', torch.zeros(numvals, dtype=torch.int32))
            self.register_buffer('vals', torch.zeros(numvals, dtype=torch.float32))
        if topX > 0:
            self.register_buffer('full_rows', torch.zeros((infeatures, topX), dtype=torch.float32))
            self.register_buffer('full_row_indices', torch.zeros(topX, dtype=torch.int32))


    def pack(self, linear, lookup_table, include_sparse):
        if self.include_bias: #linear.bias is not None:
            self.bias = linear.bias.clone() #todo: check this condition

        # self.lookup_table = lookup_table.float()
        lut,outliers = lookup_table
        # lut is the list of (LUT, labels)!!

        # handle dense matrix
        intweight = linear.weight.data.clone()
        # Note that the linear.weight dimensionality will also be (M, N) for N input perceptrons and M output perceptrons!!

        if include_sparse:
            outliers = outliers.to_dense()

        #get zero mapping
        num_channels = len(lut)
        # lut is array of (row LUT, label) --> length is M!!
        for channel in range(num_channels):
            centroid, indices = lut[channel][0] # last 0 is for group 0
            intweight[channel] = torch.from_numpy(indices)
            self.lookup_table[channel] = torch.from_numpy(centroid)
            # now, intweight stores the indices for each row & self.lookup_table stores the cluster centroids for each row

            if include_sparse:
                zero_mapping = round_to_nearest_pole_sim(torch.zeros(1), centroid)
                nonzero_vals = torch.nonzero(outliers[channel])

                outliers_channel = outliers[channel]
                outliers_channel[nonzero_vals] -= zero_mapping
                outliers[channel] = outliers_channel

        if include_sparse:
            outliers = outliers.to_sparse(layout=torch.sparse_csr)

            # save sparse matrix (already in CSR)
            self.register_buffer('rows', outliers.crow_indices().to(torch.int32))
            self.register_buffer('cols', outliers.col_indices().to(torch.int32))
            self.register_buffer('vals', outliers.values().to(torch.float32))

        # Intweight is transposed here!!!!
        # Therefore, previously, intweight's dimensionality is (outfeature, infeature)!!
        intweight = intweight.to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        # qweight = quantized weight
        # The following code compresses the intweight matrix into a quantized format!!
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)


    #replacement forward pass
    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            if self.bias is not None:
                y = self.bias.clone()
                outshape[-1] = self.bias.numel()
            else:
                y = torch.zeros((self.outfeatures), device='cuda', dtype=torch.float32)
                outshape[-1] = self.outfeatures
            dtype = x.dtype

            if self.bits == 3:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant3matmul_spmv_hybrid_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant3matmul_spmv_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant3matmul_nuq_perchannel(
                        x, 
                        self.qweight, 
                        y, 
                        self.lookup_table,
                    )
            elif self.bits == 4:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant4matmul_spmv_hybrid_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant4matmul_spmv_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant4matmul_nuq_perchannel(x, self.qweight, y, self.lookup_table)
            y = y.to(dtype)
            return y.reshape(outshape)
        else:
            out_shape = x.shape[:-1] + (self.outfeatures, )
            x = x.reshape(-1,x.shape[-1])
            out = torch.zeros((x.shape[0], self.outfeatures), device='cuda', dtype=torch.float32)
            dtype = x.dtype
            if self.bits == 3:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant3matmul_spmv_hybrid_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant3matmul_spmv_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant3matmul_nuq_perchannel_batched(
                        x, 
                        self.qweight, 
                        out, 
                        self.lookup_table,
                    )
            elif self.bits == 4:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant4matmul_spmv_hybrid_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant4matmul_spmv_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant4matmul_nuq_perchannel_batched(
                        x, 
                        self.qweight, 
                        out, 
                        self.lookup_table,
                    )
            out = out.to(dtype)
            out = out.reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out

# function to iterate through model layers and replace with our LUT-based layer
def make_quant_lut(module, names, bits, name='', include_sparse=False, numvals=None, topX=0):
    if isinstance(module, QuantLinearLUT):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 not in names:
            continue
        num = 0
        if numvals:
            num = getattr(numvals[name1])
        delattr(module, attr)
        setattr(
            module, 
            attr, 
            QuantLinearLUT(
                bits, 
                tmp.in_features, 
                tmp.out_features, 
                tmp.bias is not None, 
                include_sparse=include_sparse, 
                numvals=num, 
                topX=topX,
            ),
        )

    for name1, child in module.named_children():
        make_quant_lut(
            child, 
            names, 
            bits, 
            name + '.' + name1 if name != '' else name1, 
            include_sparse=include_sparse, 
            numvals=numvals, 
            topX=topX,
        )
