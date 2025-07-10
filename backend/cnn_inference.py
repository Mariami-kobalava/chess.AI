import os
import torch
import sys

def init_model():
    from cnn import CNN
    # გამოიყენე CNN აქ


sys.path.append(os.path.dirname(__file__))
from cnn import CNN

# from cnn import CNN

import torch
from cnn import CNN  # make sure this import is correct
import torch
from cnn import CNN  # Adjust this import based on your project structure

def init_model(checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(in_planes=26, num_blocks=2)
    model.to(device)

    if checkpoint_path is not None:
        # Load checkpoint directly to the correct device
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)

    # DEBUG: print device for each parameter to catch errors
    for name, param in model.named_parameters():
        print(f"{name} → {param.device}")

    model.eval()  # important: disables dropout/batchnorm training mode
    return model, device

# def init_model(checkpoint_path: str = None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = CNN(in_planes=26, num_blocks=2)
#     model.to(device)

#     if checkpoint_path is not None:
#         # Load checkpoint directly to the correct device
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(ckpt)

#     # DEBUG: print device for each parameter to catch errors
#     for name, param in model.named_parameters():
#         print(f"{name} → {param.device}")

#     model.eval()  # important: disables dropout/batchnorm training mode
#     return model, device

# def init_model(checkpoint_path: str = None):
#     device = 'cuda'

#     model = CNN(in_planes=26, num_blocks=2)
#     model.to(device)

#     if checkpoint_path is not None:
#         # assumes you saved with torch.save(model.state_dict(), …)
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(ckpt)

#     model.eval()    # important: turn off dropout/batchnorm training
#     return model, device


# def evaluate(model: CNN, device: torch.device, board_tensor: torch.Tensor):
#     # ensure we have a batch-dim
#     print('saaaaaad',board_tensor.device)
#     print("dddddd", next(model.parameters()).device)
#     x = board_tensor.to(device)

#     if x.dim() == 3:
#         x = x.unsqueeze(0)

#     with torch.no_grad():
#         log_p, v = model(x)
#         p = log_p.exp()                        # convert log-probs → probs

#     # move back to CPU + numpy
#     p_np = p.cpu().numpy()
#     v_np = v.cpu().numpy()

#     # if batch size = 1, return 1D arrays + scalar value
#     if p_np.shape[0] == 1:
#         return p_np[0], float(v_np[0])
#     return p_np, v_np

# def evaluate(model: CNN, device: torch.device, board_tensor: torch.Tensor):
#     print('Input tensor device before to(device):', board_tensor.device)
#     x = board_tensor.to(device)  # move tensor to device
#     print('Tensor device after to(device):', x.device)

#     if x.dim() == 3:
#         x = x.unsqueeze(0)
#     print('Tensor device after unsqueeze:', x.device)

#     print('Model parameter device:', next(model.parameters()).device)

#     with torch.no_grad():
#         log_p, v = model(x)
#         p = log_p.exp()

#     p_np = p.cpu().numpy()
#     v_np = v.cpu().numpy()

#     if p_np.shape[0] == 1:
#         return p_np[0], float(v_np[0])
#     return p_np, v_np

# if __name__ == "__main__":
#     model_path = r"C:\Users\Admin\Desktop\My_project\chess_model.pth"
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.abspath(os.path.join(base_dir, '..', 'chess_model.pth'))

#     model, device = init_model(model_path)

#     # dummy board tensor
#     dummy = torch.randn(26, 8, 8)

#     policy, value = evaluate(model, device, dummy)

#     print("Policy shape:", policy.shape)    # (4672,)
#     print("Value:", value)
import os
import torch
from cnn import CNN  # adjust import if needed


def evaluate(model: CNN, device: torch.device, board_tensor: torch.Tensor):
    print('Input tensor device before to(device):', board_tensor.device)
    x = board_tensor.to('cuda')  # move tensor to device
    print('Tensor device after to(device):', x.device)

    if x.dim() == 3:
        x = x.unsqueeze(0)
    print('Tensor device after unsqueeze:', x.device)

    print('Model parameter device:', next(model.parameters()).device)

    with torch.no_grad():
        log_p, v  = model(x)
        p = log_p.exp()

    p_np = p.cpu().numpy()
    v_np = v.cpu().numpy()

    if p_np.shape[0] == 1:
        return p_np[0], float(v_np[0]) 
    
    return v_np, p_np


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(base_dir, '..', 'chess_model.pth'))

    model, device = init_model(model_path)

    # Create dummy input with batch dimension (1, 26, 8, 8)
    dummy = torch.randn(1, 26, 8, 8)

    policy, value = evaluate(model, device, dummy)

    print("Policy shape:", policy.shape)    # (4672,) or your POLICY_SIZE
    print("Value:", value)
