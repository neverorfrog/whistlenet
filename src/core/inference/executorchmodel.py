from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_edge_compile_config,
)
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)
from executorch.exir.backend.backend_api import to_backend
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.export import ExportedProgram, export

from core.model import Model


class ExecutorchModel:
    def __init__(self, model: Model, debug: bool = False):
        self.path = f"{model.name}.pte"
        self.model = model.eval()
        quantized_model = self.quantize(model)
        traced_model: ExportedProgram = export(
            quantized_model, model.example_input
        )
        executorch_program = self.lower_model(traced_model)
        with open(self.path, "wb") as file:
            file.write(executorch_program.buffer)

    def lower_model(self, model: ExportedProgram) -> ExecutorchProgramManager:
        edge_config = get_xnnpack_edge_compile_config()
        edge_program: EdgeProgramManager = to_edge(
            model, compile_config=edge_config
        )
        edge_program = edge_program.to_backend(XnnpackPartitioner())
        executorch_program: ExecutorchProgramManager = (
            edge_program.to_executorch()
        )
        print(executorch_program.exported_program().graph.print_tabular())
        return executorch_program

    def quantize(self, model: Model):
        xnnpack_quant_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        xnnpack_quantizer = XNNPACKQuantizer()
        xnnpack_quantizer.set_global(xnnpack_quant_config)
        m = capture_pre_autograd_graph(model, model.example_input)

        # Annotate the model for quantization. This prepares the model for calibration.
        m = prepare_pt2e(m, xnnpack_quantizer)

        # Calibrate the model using representative inputs. This allows the quantization
        # logic to determine the expected range of values in each tensor.
        m(*model.example_input)
        # Perform the actual quantization.
        m = convert_pt2e(m, fold_quantize=False)
        DuplicateDynamicQuantChainPass()(m)
        return m
