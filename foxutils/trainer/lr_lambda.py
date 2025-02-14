import math

#NOTE: the range of the lambda output should be [0,1]                   
def get_cosine_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the cosine schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    def cosine_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return (idx_epoch+1) / (warmup_epoch+1)
        else:
            return 1-(1-(math.cos((idx_epoch-warmup_epoch)/(epochs-warmup_epoch)*math.pi)+1)/2)*(1-final_lr/initial_lr)
    return cosine_lambda
    
def get_linear_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the linear schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    def linear_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return (idx_epoch+1) / (warmup_epoch+1)
        else:
            return 1-((idx_epoch-warmup_epoch)/(epochs-warmup_epoch))*(1-final_lr/initial_lr)
    return linear_lambda

def get_constant_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the constant schedule.

    Args:
        initial_lr (float): Just a placeholder, no actual use.
        final_lr (float): Just a placeholder, no actual use.
        epochs (int): Just a placeholder, no actual use.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    def constant_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return (idx_epoch+1) / (warmup_epoch+1)
        else:
            return 1
    return constant_lambda
