import torch
from torch.profiler import profile, record_function, ProfilerActivity
from agents.alphazero.network import Connect4Net

# Initialize model and dummy input
model = Connect4Net()
# Detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profile_dir = "./cuda_profile_logs"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    activities = [ProfilerActivity.CPU, ProfilerActivity.MPS]
    profile_dir = "./mps_profile_logs"
else:
    device = torch.device("cpu")
    activities = [ProfilerActivity.CPU]
    profile_dir = "./cpu_profile_logs"

model.to(device)

# Adjust the input size based on the model's requirements
dummy_input = torch.randn(1, 3, 6, 7).to(device)  # Batch size 1, 3 channels, 6x7 board

# GPU profiling
with profile(
    activities=activities,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    record_shapes=True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        output = model(dummy_input)

# Write profiling results to a file
log_file_path = f"{device}_profile_results.log"
with open(log_file_path, "w") as log_file:
    log_file.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
