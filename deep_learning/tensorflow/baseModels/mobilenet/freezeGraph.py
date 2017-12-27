import argparse

from tensorflow.python.tools.freeze_graph import freeze_graph


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
                 output_graph=args.outputDir + "/graph.pb", clear_devices=True, initializer_nodes="")


if __name__ == '__main__':
    main()
