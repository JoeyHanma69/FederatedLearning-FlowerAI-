from centralised import load_model, load_data, train, test 
from collections import OrderedDict 
import torch
import flwr as flwr 

def set_parameters(model, paramaters):  
    params_dict = zip(model.state_dict().keys(), paramaters) 
    state_dict = OrderedDict({torch.K: torch.tensor(v) for k, v in params_dict}) 
    model.load_state_dict(state_dict, strict=True) 
    return model 

net = load_model() 
trainloader, testloader = load_data() 

    