from centralised import load_model, load_data, train, test 
from collections import OrderedDict 
import torch
import flwr as fl 

def set_parameters(model, paramaters):  
    params_dict = zip(model.state_dict().keys(), paramaters) 
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict}) 
    model.load_state_dict(state_dict, strict=True) 
    return model 

net = load_model() 
trainloader, testloader = load_data() 

class FlowerClient(fl.client.NumPyClient):  
    def get_parameters(self, config): 
        return [val.cpu().numpy() for _, val in net.state_dict().items()]  
    
    def fit(self, parameters, config): 
        set_parameters(net, parameters)  
        train(net, trainloader, epochs=1)  
        return self.get_parameters({}), len(trainloader.dataset), {} 
    
    def evaluate(self, parameters, config): 
        set_parameters(net, parameters)  
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy" : accuracy} 
    
fl.client.start_numpy_client( 
    server_address="127.0.0.1:8080", 
    client=FlowerClient()         
)
        
    