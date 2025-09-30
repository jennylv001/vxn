import torch
import onnxruntime as ort

def check_torch():
    print("=== PyTorch ===")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

def check_onnxruntime():
    print("\n=== ONNX Runtime ===")
    providers = ort.get_available_providers()
    print("Available providers:", providers)
    using_cuda = "CUDAExecutionProvider" in providers
    print("CUDAExecutionProvider available:", using_cuda)
    if using_cuda:
        print("Tip: set providers=['CUDAExecutionProvider'] and provider_options={'device_id': <gpu_id>} when creating sessions.")

if __name__ == "__main__":
    check_torch()
    check_onnxruntime()
