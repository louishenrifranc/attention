# Implementation of Attention is All You Need in Sonnet/Tensorflow
Architecture:
![](https://camo.githubusercontent.com/88e8f36ce61dedfd2491885b8df2f68c4d1f92f5/687474703a2f2f696d6775722e636f6d2f316b72463252362e706e67)

Paper:
https://arxiv.org/abs/1706.03762

## Usage
1. Install requirements
```
pip install -r requirements.txt
```
2. Install [sonnet](https://github.com/deepmind/sonnet)
3. Run ```run_micro_services.sh```

## Organisation of the repository
Transformer is very modular. I try to decouple it as much as possible:
* ```attention/algorithms/transformers```: Create an ```tf.contrib.learn.Experiment```, and the ```tf.contrib.data.Dataset```
* ```attention/modules/cores```: Implementation of the core blocks of Transformer such as ```MultiHeadAttention```, ```PointWiseFeedForward```
* ```attention/modules/decoders```: Implementation of the Decoder block
* ```attention/modules/encoders```: Implementation of the Encoder blok
* ```attention/models```: Implementation of a Transformer Module
* ```attention/services```: Micro Services that create the dataset, or train the model
* ```attention/utils```: Classes uses as utility (recursive namespace, mocking object)
* ```attention/*/tests/```:  Test of the Module/Algorithm/MicroService

### Training Task implemented
- [X] Copy inputs
- [ ] Dialogue generation



## Road Map
- [X] Code modules
- [X] Test modules
- [X] Construct input function
- [X] Build Estimator
- [X] Run estimator
- [X] Plug into a workflow
- [X] Add validation queue
- [ ] Iterate over model improvements

