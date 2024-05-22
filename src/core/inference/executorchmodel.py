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
from torch.export import ExportedProgram, export

from core.model import Model


class ExecutorchModel:
    def __init__(self, model: Model, debug: bool = False):
        self.path = f"{model.name}.pte"
        self.model = model.eval()

        aten_dialect: ExportedProgram = export(model, model.example_input)
        if debug:
            aten_dialect.graph.print_tabular()

        edge_config = get_xnnpack_edge_compile_config()
        edge_program: EdgeProgramManager = to_edge(
            aten_dialect, compile_config=edge_config
        )
        edge_program = edge_program.to_backend(XnnpackPartitioner())
        print(edge_program.exported_program().graph_module)

        self.executorch_program: ExecutorchProgramManager = (
            edge_program.to_executorch()
        )
        if debug:
            print(
                self.executorch_program.exported_program().graph.print_tabular()
            )

        with open(self.path, "wb") as file:
            file.write(self.executorch_program.buffer)
