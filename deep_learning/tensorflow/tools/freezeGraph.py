import argparse
import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Script to generate a batch sample from a trained model")
    parser.add_argument("--modelDir", required=True, type=str, help="Checkpoint directory")
    parser.add_argument("--frozenModelDir", required=False, type=str, default="./export",
                        help="Frozen graph file")
    parser.add_argument("--outputName", required=False, type=str, default="softmax", help="Name of the output layer")
    args = parser.parse_args()
    return args


def main():

    args = doParsing()
    print(args)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        metagraphPath = os.path.join(args.modelDir, "model.meta")
        checkpointPath = os.path.join(args.modelDir, "model")

        print("Loading metagraph")
        saver = tf.train.import_meta_graph(metagraphPath)
        print("Restoring model")
        restored = saver.restore(sess, checkpointPath)
        print("Checkpoint loaded")

        # Save metagraph
        if os.path.exists(os.path.join(args.modelDir, "model_graph.pb")) is False:
            tf.train.write_graph(sess.graph.as_graph_def(), "", args.modelDir + "/model_graph.pb", False)
            print("Metagraph saved")

    # Freeze graph (graphdef plus parameters),
    # this includes in the graph only the layers needed to provide the output_node_names
    print("Freezing graph...")
    freeze_graph(input_graph=args.modelDir + "/model_graph.pb", input_saver="", input_binary=True,
                 input_checkpoint=checkpointPath, output_node_names=args.outputName,
                 restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                 output_graph=args.frozenModelDir + "/graph.pb", clear_devices=True, initializer_nodes="")


if __name__ == "__main__":
    main()