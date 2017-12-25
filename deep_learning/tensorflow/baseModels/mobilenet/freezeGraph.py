import argparse

from tensorflow.python.tools.freeze_graph import freeze_graph

#TODO: Add a guide and a reference to official way to export tf slim model
# First step to export graph_def is needed

#python export_inference_graph.py   --alsologtostderr   --model_name=mobilenet_v1   --image_size=224
# --output_file=../../../mobilenet/mobilenet_v1_224_graph_def.pb

def doParsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='Freeze MobileNet graph.pb script')
    parser.add_argument('--modelDir', required=True, help="Model directory with graph def pb and checkpoint")
    parser.add_argument('--outputDir', required=False, default="./output", help="Output directory for graph.pb")
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    freeze_graph(input_graph=args.modelDir + "/mobilenet_v1_224_graph_def.pb", input_saver="", input_binary=True,
                 input_checkpoint=args.modelDir + "/mobilenet_v1_1.0_224.ckpt",
                 output_node_names="MobilenetV1/Predictions/Reshape_1",
                 restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                 output_graph=args.outputDir + "/mobilenet_v1_224_graph.pb", clear_devices=True, initializer_nodes="")


if __name__ == '__main__':
    main()
