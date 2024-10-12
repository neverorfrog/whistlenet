import onnx
import onnxruntime
import torch
from core.model import Model

from whistlenet.core.utils import to_numpy


class ONNX:
    def __init__(self, torch_model: Model):
        self.onnx_path = f"{torch_model.name}.onnx"
        self.torch_model = torch_model
        torch_model.load()
        torch_model.eval()
        torch_input = torch_model.example_input[0]

        torch.onnx.export(
            torch_model,
            torch_input,
            f"{torch_model.name}.onnx",
            verbose=True,
            input_names=["input"],
            output_names=["output"],
        )

        self.ort_session = onnxruntime.InferenceSession(
            f"{torch_model.name}.onnx", providers=["CPUExecutionProvider"]
        )

    def get_model(self) -> onnx.ModelProto:
        onnx_model = onnx.load(f"{self.torch_model.name}.onnx")
        onnx.checker.check_model(onnx_model)
        return onnx_model

    def __call__(self, torch_input):
        ort_input = {
            self.ort_session.get_inputs()[0].name: to_numpy(torch_input)
        }
        return self.ort_session.run(None, ort_input)
