# Training Mixed-Precision Quantized Neural Networks for microcontroller deployments

### Description
This project targets quantization-aware training methodologies on Pytorch for microcontroller deployment of quantized neural networks. The featured mixed-precision quantization techniques aim at byte or sub-byte quantization, i.e. INT8, INT4, INT2. The generated network for deployment supports integer arithmetic only. Optionally, the selection of individual per-tensor bit precision is driven by the device memory constraints.

### Reference
Please, cite this paper [arXiv](https://arxiv.org/abs/1905.13082) when using the code.
```
@article{rusci2019memory,
  title={Memory-Driven Mixed Low Precision Quantization For Enabling Deep Network Inference On Microcontrollers},
  author={Rusci, Manuele and Capotondi, Alessandro and Benini, Luca},
  journal={arXiv preprint arXiv:1905.13082},
  year={2019}
}

```

### Questions
For any question just drop me an [email](mailto:manuele.rusci@unibo.it). 

## Getting Started

### Prerequisites
- The code is tested with PyTorch 0.4.1 and Python 3.5
- Tensorflow package is needed to load pretrained tensorflow model weights

### Setup
Set the correct dataset paths inside `data.py` . As an example:
```
_IMAGENET_MAIN_PATH = '/home/user/ImagenetDataset/'
_DATASETS_MAIN_PATH = './datasets/'
```
To download pretrained mobilenet weights:
```
$ cd models/mobilenet_tf/
$ source download_pretrained_mobilenet.sh
```

### Quickstart
For quantization-aware retraining of a 8-bit integer only mobilenet model type:
```
$ python3 main_binary.py -a mobilenet --mobilenet_width 1.0 --mobilenet_input 224 --save Imagenet/mobilenet_224_1.0_w8a8 --dataset imagenet --type_quant 'PerLayerAsymPACT' --weight_bits 8 --activ_bits 8 --activ_type learned --gpus 0,1,2,3 -j 8 --epochs 12 -b 128 --save_check --quantizer --batch_fold_delay 1 --batch_fold_type folding_weights
```

### Quantization Options

- *quantizer*: enables quantization when True 
- *type_quant*: type of weight uantization method to apply (see below)
- *weight_bits*: number of bits for weights quantization
- *activ_bits*: number of activation bits
- *activ_type*: type of quantized activation layers
- *batch_fold_delay*: number of epochs before freezing batch norm parameters 
- *batch_fold_type*: how to deal with folding of batch norm parameters (or any other scalar params). \[Supported: 'folding_weights' | 'ICN'\] 
- *quant_add_config*: optinal list of per-layer configuration, which overwrite previous settings on a per-layer basis
- *mobilenet_width*: Mobilenet width multiplier ( default=1.0; supported \[ 0.25, 0.5, 0.75, 1.0 \] ) 
- *mobilenet_input*: Mobilenet resolution input size ( default=224; supported \[ 128, 160, 192, 224 \] ) 
- *mem_constraint*: Memory contraints of the target device. Must be provided as a string '\[ROM_SIZE,RAM_SIZE\]'
- *mixed_prec_quant*: Mixed Per-Layer ('MixPL') or mixed per-channel ('MixPC') 


### Reproducing paper results
For any given mobilenet model, run the script with:

- memory constraints 512kB of RAM and 2MB of FLASH *--mem_constraint \[2048000,512000\]*
- mixed precision per-layer or per-channel  *--mixed_prec_quant MixPL* (or MixPC)

As an example:
```
$ python3 main_binary.py --model mobilenet --save Imagenet_ARM/mobilenet_128_0.75_quant_auto_tt --mobilenet_width 0.75 --mobilenet_input 128 --dataset imagenet -j 32 --epochs 10 -b 128 --save_check --gpus 0,1,2,3 --type_quant PerLayerAsymPACT --activ_type learned --quantizer --batch_fold_delay 1 --batch_fold_type folding_weights --mem_constraint [2048000,512000] --mixed_prec_quant MixPL
```


## Quantization Strategy Guide

### Overview

The quantization functions are located into `quantization/quantop.py`. The operator `QuantOp` wraps the full-precision model to handle weight quantization. As a usage example:
```
import quantization 
quantizer = quantization.QuantOp(model, type_quant, weight_bits, \
            batch_fold_type=args.batch_fold_type, batch_fold_delay=batch_fold_delay, \
            act_bits=activ_bits, add_config = quant_add_config )
```

The operator *QuantOp* after wrapping a full-precision model:

- generates the deployment  integer-only graph **quantizer.deployment_model**, based on the full-precision graph *model*.
- updates the quantized parameters of the deployment model based on the actual full-precision graph parameters **quantizer.generate_deployment_model()**
- provides methods to support quantization-aware retraining of the full-precision model

At training time, the quantizer works in combination with the optimizer: 
```
  # weight quantization before the forward pass
  quantizer.store_and_quantize() # copy the real-value weights and quantize the actual ones
   
  # forward pass
  output = model(input) # compute output
  loss = criterion(output, target) # compute loss

  if training:
      # backward pass
      optimizer.zero_grad()
      loss.backward()

      quantizer.restore_real_value()  # restore real value parameters          
      quantizer.backprop_quant_gradients() # compute gradients wrt to real-value weights      

      optimizer.step() # update the values
      
  else:
      quantizer.restore_real_value() # restore real-value weights after forward pass
```


### Weight Quantization
Currently, the following quantization schemes are supported:

- *PerLayerAsymPACT*: per-layer asymmetric quantization, quantization range is learned with PACT method
- *PerChannelsAsymMinMax*: per-channel asymmetric quantization, quantization range is defined by min/max range of the weight-channel tensor 
- *PerLayerAsymMinMax*: per-layer asymmetric quantization, quantization range is defined by min/max range of the weight tensor (not fully tested)


### Activation Quantization
At the present stage, the quantized activation layers must be part of the model definition itself. This is why the input model is already a fake-quantized model. See 'models/mobilenet.py' as an example. This part will be improved with automatic graph analysis and parsing, to turn a full-precision input model into a fake-quantized one.

### Limitations
This project does not include any graph analysis tools. Hence, the graph parser (see \_\_init\_\_ of *QuantOp* operator) is specific for the tested model 'models/mobilenet.py', which already includes quantized activation layers. A rework of this part may be necessary to apply the implemented techniques on any other models.