## NasNet Mobile graph.pb generation

Google provides slim checkpoints for NasNet (mobile and full versions) and other models. [Tensorflow Models Github repo](https://github.com/tensorflow/models/tree/d07447a3e34bc66acd9ba7267437ebe9d15b45c0/research/slim), tested version is here as git submodule in tfmodels directory.

Necessary steps:
- Download the pretrained model checkpoint
- Save the graph definition
- Freeze the graph (graph.pb will include the graph definition and the learned weights)

NasNet requires the usage of slim scripts, the solution is to use the slim guide from the [Tensorflow Models Github repo](https://github.com/tensorflow/models/tree/d07447a3e34bc66acd9ba7267437ebe9d15b45c0/research/slim).
Tested also with Inception V3.

#### Download and extract the model

Download the NasNet Mobile model checkpoint from [Tensorflow Models Github repo](https://github.com/tensorflow/models/tree/d07447a3e34bc66acd9ba7267437ebe9d15b45c0/research/slim).
Extract the files in this directory.

#### Graph Def Save
Use the *export_inference_graph.py* script from slim repository (placed in *tfmodels/research/slim*).

```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=nasnet_mobile \
  --image_size=224 \
  --output_file=../../../baseModels/nasnet_mobile/nasnet_mobile_graph_def.pb
```

#### Freezed Graph
To freeze the graph, we need the checkpoint for the correct model retrieved from [Tensorflow Models Github repo](https://github.com/tensorflow/models/tree/d07447a3e34bc66acd9ba7267437ebe9d15b45c0/research/slim).
The official slim tutorial suggest to use the c++ API, but it needs to compile the whole tensorflow library, so I prefer the python API.

Here you can find a custom *freezeGraph.py* created by only for MobileNet and easily adaptable to other models.

```
python freezeGraph.py \
--modelDir=./
--outputDir=./
```

Now you a have a graph.pb file containing all the trained weights and you can load the model with training script.

#### Final notes

Please notice that NasNet pretrained ImageNet models are trained on 1001 classes, the FIRST class in the synset is a background class.
Furthermore NasNet, MobileNet and Inception needs a specific preprocessing of the input images: each image needs to be RGB and normalized from -1.0 to 1.0.
Other models like SqueezeNet, Alexnet, VGG, etc... use a different preprocessing (0 - 255 range, mean subtraction, some models converted from caffe uses BGR).