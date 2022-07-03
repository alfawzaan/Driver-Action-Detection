# This is a sample Python script.

from argparse import ArgumentParser

import cv2
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from inference_d import NetworkDecoder
from inference_e import NetworkEncoder


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("-ec_model", "--encoder_model_path", required=True,
                        help="path to the Encoder Model IR xml file")
    parser.add_argument("-dc_model", "--decoder_model_path", required=True,
                        help="path to the Decoder Model IR xml file")
    parser.add_argument("-i", "--input_type", required=True,
                        help="Input type (CAM, VIDEO or IMAGE)")
    parser.add_argument("-source", "--input_path", required=False,
                        help="path to the input file (Required for Video and Image Files)")
    return parser


def app(args):
    action_list = ["Safe driving", "Texting left", "Texting right", "Talking phone left", "Talking phone right",
                   "Operating radio", "Drinking eating", "Reaching behind", "Hair and makeup"]
    encoder_net = args.encoder_model_path
    network_encoder = NetworkEncoder(encoder_net)

    decoder_net = args.decoder_model_path
    network_decoder = NetworkDecoder(decoder_net)

    network_encoder.load_model()
    network_decoder.load_model()
    n_e, c_e, h_e, w_e = network_encoder.get_input_shape()

    asd = network_decoder.get_input_shape()  # n_d, c_d, h_d, w_d
    # print(asd)

    input_device = args.input_type

    if input_device == "CAM":
        input_device = 0
    else:
        input_device = args.input_path
    print(f"INPUT TYPE: {input_device} SOURCE: {args.input_path}")
    cap = cv2.VideoCapture(input_device)

    # cap.open(input_device)

    embedding_count = 0
    embedded_encode = []
    message_infer = ""
    while True:  # cap.isOpened():

        ### frames should be feed in every 60 seconds
        flag, frame = cap.read()
        key_pressed = cv2.waitKey(10)  # 60
        # print(f"FRAME: {frame}")
        # cv2.imshow("Visualize", frame)

        if not flag:
            break
        pre_pro_frame = cv2.resize(frame, (w_e, h_e))
        pre_pro_frame = pre_pro_frame.transpose((2, 0, 1))
        pre_pro_frame = pre_pro_frame.reshape((n_e, c_e, h_e, w_e))  # n, c,

        network_encoder.exec_net(pre_pro_frame)
        if network_encoder.wait() == 0:
            result_encode = network_encoder.get_output()
            # print(f"ENCODED RESULT {np.shape(np.array(result_encode).flatten().tolist())}")
            embedded_encode.append(np.array(result_encode).flatten().tolist())
            if len(embedded_encode) == 16:
                print("Decoding")
                ###DECODE
                embed = []
                embed.append(embedded_encode)
                network_decoder.exec_net(embed)

                if network_decoder.wait() == 0:
                    result_decode = network_decoder.get_output()
                    probs = softmax(result_decode - np.max(result_decode)).flatten().tolist()
                    print(f"RESULT: {result_decode}")
                    print(f"PROBABILITY: {probs}")
                    embedded_encode = []
                    message_infer = action_list[probs.index(max(probs))]
        cv2.putText(frame, f"{message_infer}", (15, 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.imshow("Visualize", frame)
        if key_pressed == 27:
            break



def softmax(x, axis=None):
    """Normalizes logits to get confidence values along specified axis"""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argument = parse_argument().parse_args()
    app(argument)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
