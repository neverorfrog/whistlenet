#include <torch/torch.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/util/util.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/extension/evalue_util/print_evalue.h>

#include <iostream>
#include <string>
#include <vector>

using torch::data::datasets::MNIST;
using torch::data::Example;
using std::cout, std::endl, std::cerr;
using namespace torch::executor;
using torch::executor::util::FileDataLoader;
static constexpr LogLevel Info = torch::executor::LogLevel::Info;
static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

static TensorImpl fc_tensor() {
  torch::Tensor tensor = torch::randn({1,4});
  Tensor::SizesType sizes[] = {1,4};
  Tensor::DimOrderType dim_order[] = {0,1};
  TensorImpl impl(ScalarType::Float,2,sizes,&tensor,dim_order);
  return impl;
}

static TensorImpl cnn_tensor() {
  const char* kDataRoot = "../python/data/MyMNIST/raw";
  auto test_dataset = MNIST(kDataRoot, MNIST::Mode::kTest);
  Example example = test_dataset.get(1);
  torch::Tensor tensor = example.data;
  // torch::Tensor tensor = torch::randn({1,1,28,28});
  Tensor::SizesType sizes[] = {1,1,28,28};
  Tensor::DimOrderType dim_order[] = {0,1,2,3};
  TensorImpl impl(ScalarType::Float,4,sizes,&tensor,dim_order);
  return impl;
}

static TensorImpl input_tensor(){
  return cnn_tensor();
}

int main() {
  runtime_init();

  // model loader
  Result<FileDataLoader> loader = FileDataLoader::from("cnn/model.pte");
  assert(loader.ok());
  Result<Program> program = Program::load(&loader.get());
  assert(program.ok());

  /*Method names map back to Python nn.Module method names.*/
  const char* method_name = "forward";

  /* MethodMeta is a lightweight structure that lets us gather metadata
  information about a specific method. In this case we are looking to
  get the required size of the memory planned buffers for the method "forward".*/
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  assert(method_meta.ok());

  /* One of the principles of ExecuTorch is giving users control 
  over where the memory used by the runtime comes from. 
  We need to define two different allocators.

  First allocator: MemoryAllocator used to allocate 
  runtime structures at Method load time. 
  Things like Tensor metadata, the internal chain of instructions, 
  and other runtime state come from this.
  
  This allocator is only used during loading a method of the program, 
  which will return an error if there was not enough memory.

  The amount of memory required depends on the loaded method and the runtime
  code itself. The amount of memory here is usually determined by running the
  method and seeing how much memory is actually used, though it's possible to
  subclass MemoryAllocator so that it calls malloc() under the hood (see
  MallocMemoryAllocator).

  In this example we use a statically allocated memory pool.*/
  MemoryAllocator method_allocator{MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  /* The memory-planned buffers will back the mutable tensors used by the
  method. The sizes of these buffers were determined ahead of time during the
  memory-planning pasees.

  Each buffer typically corresponds to a different hardware memory bank. Most
  mobile environments will only have a single buffer. Some embedded
  environments may have more than one for, e.g., slow/large DRAM and
  fast/small SRAM, or for memory associated with particular cores. */
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.\n", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }

  /*Second Allocator: Planned Memory: A HierarchicalAllocator containing 1 or more memory spans 
  where internal mutable tensor data buffers are placed. At Method load time 
  internal tensors have their data pointers assigned to various offsets within. 
  The positions of those offsets and the sizes of the arenas are 
  determined by memory planning ahead of time.*/
  HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});

  // Assemble all of the allocators into the MemoryManager that the Executor will use.
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  /*Load the method from the program, using the provided allocators. Running
  the method can mutate the memory-planned buffers, so the method should only
  be used by a single thread at at time, but it can be reused.*/
  Result<Method> method =  program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      method.error()
  );
  printf("Method loaded.\n");

  /*Getting input for specific model: if you use one, you have to comment the other
  Need to find a better solution than this zozzata, to test more easily*/ 
  TensorImpl impl = input_tensor();
  Tensor t(&impl);
  Error set_input_error = method->set_input(t, 0);
  assert(set_input_error == Error::Ok);

  // Run the model.
  Error status = method->execute();
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)status);
  printf("Model executed successfully.\n");

  // Print outputs
  std::vector<EValue> outputs(method->outputs_size());
  printf("%zu outputs: ", outputs.size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  cout << torch::executor::util::evalue_edge_items(10);
  for (int i = 0; i < outputs.size(); ++i) {
    std::cout << "Output " << i << ": " << outputs[i] << std::endl;
  }
} 
