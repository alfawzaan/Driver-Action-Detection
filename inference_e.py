import sys
from logging import log

from openvino.inference_engine import IECore

from intel.helpers import encoder_path
from argparse import ArgumentParser


class NetworkEncoder:
    def __init__(self, encoder_net):
        self.network = None
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_req = None
        self.encoder_net = encoder_net

    def load_model(self, device="CPU", cpu_extension=None):
        model_weight = self.encoder_net[:-3] + "bin"
        self.plugin = IECore()
        print(f"ENCODER PATH:{model_weight}")
        self.network = self.plugin.read_network(
            model=self.encoder_net, weights=model_weight)  # IENetwork(model=encoder_path, weights=model_weight)

        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        self.exec_network = self.plugin.load_network(self.network, device)

        if "CPU" in device:
            supported_layer = self.plugin.query_network(self.network, "CPU")
            unsupported_layer = [l for l in self.network.layers.keys() if l not in supported_layer]

            if len(unsupported_layer) > 0:
                log.error("Following layers are not supported by the plugin for the specified device {}:\n {}".
                          format(device, ', '.join(unsupported_layer)))

                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)

        self.input_blob = next(iter(self.network.input_info))
        self.output_blob = next(iter(self.network.outputs))

        print(self.input_blob)
        print(self.output_blob)

        return self.exec_network

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, inf_input):
        result = self.exec_network.start_async(request_id=0, inputs={self.input_blob: inf_input})
        return result

    def get_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]

    def wait(self):
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status
