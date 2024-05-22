import onnx
import onnxruntime
import torch

from core import to_numpy
from core.model import Model


class ONNXModel:
    def __init__(self, torch_model: Model):
        self.onnx_path = f"{torch_model.name}.onnx"
        self.torch_model = torch_model
        self.onnx_program = torch.onnx.dynamo_export(
            torch_model, torch_model.example_input[0]
        )
        self.onnx_program.save(self.onnx_path)
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )

    def load(self) -> onnx.ModelProto:
        onnx_model = onnx.load(f"{self.torch_model.name}.onnx")
        onnx.checker.check_model(onnx_model)
        return onnx_model

    def convert_input(self, torch_input):
        onnx_input = self.onnx_program.adapt_torch_inputs_to_onnx(torch_input)
        onnxruntime_input = {
            k.name: to_numpy(v)
            for k, v in zip(self.ort_session.get_inputs(), onnx_input)
        }
        return onnxruntime_input

    def __call__(self, torch_input):
        onnxruntime_input = self.convert_input(torch_input)
        onnxruntime_output = self.ort_session.run(None, onnxruntime_input)
        return onnxruntime_output
