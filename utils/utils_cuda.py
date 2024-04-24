import random
import torch ,os
import numpy as np


##########################
### SETTINGS
##########################




def get_device():
    """ 
    get the device to run the model on : GPU or CPU
    """     
    if torch.cuda.is_available():
        cuda_id = torch.cuda.current_device()
        CUDA_DEVICE_NUM = cuda_id
        DEVICE = torch.device(f"cuda:{CUDA_DEVICE_NUM}")
        print("Device:", DEVICE)
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    else:
        DEVICE = torch.device( "cpu")
        
    return DEVICE

def seed_everything(seed=42):
    #os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        

import gc

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        