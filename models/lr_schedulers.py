def get_linear_warmup_lr_lambda(num_warmup_steps: int):
    def linear_warmup_lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps
        else:
            return 1.0
    return linear_warmup_lr_lambda