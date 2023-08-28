## Pytorch的量化框架具有的特点

1. PyTorch has data types corresponding to quantized tensors, which share many of the features of tensors.具有量化张量类型，与普通张量具有很多相同的特征

2. One can write kernels with quantized tensors, much like kernels for floating point tensors to customize their implementation.
   
   - torch.nn.quantized
   
   - torch.nn.quantized.dynamic

3. Quantization is compatible with the rest of PyTorch
   
   量化与PyTorch的其他模块兼容
   
   One can easily mix quantized and floating point operations in a model.

4. 从浮点张量到量化张量的映射可以用户自定义，PyTorch定义了常用的默认实现.

## 1.Dynamic Quantization

- torch.quantization.quantize_dynamic

- 不仅将权重量化为int8，并且即时的把激活函数转换到int8.所以叫动态量化.

- [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)

## 2.Post-Training Static Quantization

## 3.Quantization Aware Training

- 训练期间的所有权重调整都是在“意识到”模型最终将被量化的事实的情况下进行的


