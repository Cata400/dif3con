import torch
from torch import nn
import torch.nn.functional as F


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def mse_loss(output, target, labels=1.0):
    return F.mse_loss(output, target)


def weighted_mse_loss(tensor1, tensor2, labels, use_noise=False, max_loss=0.0, forget_alpha=0.25):
    retain_loss = F.mse_loss(tensor1 * (labels[:, None, None, None] + 1.0) * 0.5, 
                             tensor2 * (labels[:, None, None, None] + 1.0) * 0.5) 
    if use_noise:
        tensor2= torch.randn_like(tensor2) * torch.std(tensor2) + torch.mean(tensor2)
    forget_loss = (1 - 2.0 * max_loss) * F.mse_loss(
                tensor1 * (1.0 - labels[:, None, None, None]) * 0.5, 
                tensor2 * (1.0 - labels[:, None, None, None]) * 0.5
            ) 
    return retain_loss + forget_alpha * forget_loss


def cosine_distance(x1, x2):
    """
    Computes the cosine distance between two \ell_2 normalized tensors of shape (B, *).
    
    Assumption: ||x1||_2 = ||x2||_2 = 1
    """
    x1_flat = x1.view(x1.size(0), x1.size(1), -1)
    x2_flat = x2.view(x2.size(0), x2.size(1), -1)

    x1_normalized = x1_flat / torch.norm(x1_flat, dim=-1, keepdim=True)
    x2_normalized = x2_flat / torch.norm(x2_flat, dim=-1, keepdim=True)

    return 1 - (x1_normalized * x2_normalized).sum(dim=-1)


# TODO: complete for the other training methods: max_loss, use_noise, etc.
def weighted_cosine_loss(tensor1, tensor2, labels, use_noise=False, max_loss=0.0, forget_alpha=0.25):
    retain_loss = cosine_distance(
        tensor1 * (labels[:, None, None, None] + 1.0) * 0.5, 
        tensor2 * (labels[:, None, None, None] + 1.0) * 0.5
    )

    forget_loss = cosine_distance(
        tensor1 * (1.0 - labels[:, None, None, None]) * 0.5, 
        tensor2 * (1.0 - labels[:, None, None, None]) * 0.5  
    )
    
    return retain_loss + forget_alpha * forget_loss


def weighted_others_loss(noise_hat, teacher_feat, updated_feat, labels, use_noise=False, max_loss=0.0, forget_alpha=0.25):
    retain_loss = F.mse_loss(
        noise_hat * (labels[:, None, None, None] + 1.0) * 0.5, 
        teacher_feat * (labels[:, None, None, None] + 1.0) * 0.5
    ) 
    forget_loss = (1 - 2.0 * max_loss) * F.mse_loss(
                noise_hat * (1.0 - labels[:, None, None, None]) * 0.5, 
                updated_feat * (1.0 - labels[:, None, None, None]) * 0.5
            ) 
    return retain_loss + forget_alpha * forget_loss


def retain_forget_loss(tensor1, tensor2, labels):
    retain_loss = F.mse_loss(tensor1 * (labels[:, None, None, None] + 1.0) * 0.5, 
                             tensor2 * (labels[:, None, None, None] + 1.0) * 0.5) 

    forget_loss = F.mse_loss(
                tensor1 * (1.0 - labels[:, None, None, None]) * 0.5, 
                tensor2 * (1.0 - labels[:, None, None, None]) * 0.5
            ) 
    return retain_loss, forget_loss


class GradientHarmonizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, retain_loss, forget_loss):
        ctx.save_for_backward(retain_loss, forget_loss)
        return retain_loss + forget_loss

    @staticmethod
    def backward(ctx, grad_output):
        retain_loss, forget_loss = ctx.saved_tensors
        
        # Get the two gradients
        retain_grad = torch.autograd.grad(retain_loss, retain_loss, grad_output, retain_graph=True)[0]
        forget_grad = torch.autograd.grad(forget_loss, forget_loss, grad_output)[0]

        ### Modify gradient of forget / retain to match better the direction of the other
        cosine = torch.dot(retain_grad.view(-1), forget_grad.view(-1)) / (torch.norm(retain_grad) * torch.norm(forget_grad))
        
        ### For forget 
        forget_grad = forget_grad if cosine >= 0 else forget_grad - cosine * retain_grad
        
        ### For retain
        # retain_grad = retain_grad if cosine >= 0 else retain_grad - cosine * forget_grad

        # Final gradient computation
        final_grad = retain_grad + forget_grad

        return final_grad, None, None

def gradient_harm_loss(tensor1, tensor2, labels, forget_alpha=0.25, max_loss=0.0):
    """
    Performs custom gradient computation given the two non-weighted losses: retain and forget.
    
    Args:
        tensor1 (_type_): _description_
        tensor2 (_type_): _description_
        labels (_type_): _description_
        forget_alpha (float, optional): Only added for compatibility.
        max_loss (float, optional): Only added for compatibility.
    """

    retain_loss = F.mse_loss(tensor1 * (labels[:, None, None, None] + 1.0) * 0.5,
                            tensor2 * (labels[:, None, None, None] + 1.0) * 0.5)
    forget_loss = F.mse_loss(tensor1 * (1.0 - labels[:, None, None, None]) * 0.5,
                            tensor2 * (1.0 - labels[:, None, None, None]) * 0.5)
    
    return GradientHarmonizer.apply(retain_loss, forget_loss)


def l2_regularizer(model, model_prev, lambda_reg, opt_prev=None):
    regularization_loss = 0.0
    
    if opt_prev is not None:
        optimizer_state = opt_prev.state_dict()["state"]
    
    for i, (param, param_prev) in enumerate(zip(model.parameters(), model_prev.parameters())):
        layer_sq_diff = (param - param_prev) ** 2
        
        if opt_prev is not None:
            # Adjust the regularization based on the optimizer state
            if optimizer_state.get(i):
                if 'exp_avg_sq_epoch' in optimizer_state[i]:
                    exp_avg_sq = optimizer_state[i]['exp_avg_sq_epoch']
                else:
                    exp_avg_sq = optimizer_state[i]['exp_avg_sq']
                exp_avg_sq = exp_avg_sq / exp_avg_sq.max()
                layer_sq_diff = layer_sq_diff * exp_avg_sq
                
        regularization_loss += lambda_reg * torch.sum(layer_sq_diff)
    return regularization_loss

def l2_salun_regularizer(model, model_prev, lambda_reg, opt_prev=None):
    regularization_loss = 0.0
    
    if opt_prev is not None:
        optimizer_state = opt_prev.state_dict()["state"]
        if 'exp_avg_sq_epoch' in optimizer_state[i]:
            exp_avg_sq_list = [optimizer_state[i]['exp_avg_sq_epoch'] for i in optimizer_state.keys()]
        else:
            exp_avg_sq_list = [optimizer_state[i]['exp_avg_sq'] for i in optimizer_state.keys()]
        threshold = torch.median(torch.stack([torch.median(exp_avg_sq) for exp_avg_sq in exp_avg_sq_list]))
    
    for i, (param, param_prev) in enumerate(zip(model.parameters(), model_prev.parameters())):
        layer_sq_diff = (param - param_prev) ** 2
        
        if opt_prev is not None:
            # Adjust the regularization based on the optimizer state
            if optimizer_state.get(i):
                if 'exp_avg_sq_epoch' in optimizer_state[i]:
                    exp_avg_sq = optimizer_state[i]['exp_avg_sq_epoch']
                else:
                    exp_avg_sq = optimizer_state[i]['exp_avg_sq']
                exp_avg_sq_binary = (exp_avg_sq >= threshold).float()
                layer_sq_diff = layer_sq_diff * exp_avg_sq_binary
                
        # regularization_loss += lambda_reg * torch.mean(layer_sq_diff)
        regularization_loss += lambda_reg * torch.sum(layer_sq_diff)
    return regularization_loss