import warnings 

import numpy as np 
from flwr_datasets import FederatedDataset 
from flwr_datasets.partitioner import IidPartitioner 

from flwr.client import ClientApp 
from flwr.common import Message, Context, MetrixRecord, RecordDict

fds = None # Cache Federaated Dataset

warnings.filterwarnings("ignore", category=UserWarning) 


def get_clientapp_dataset(partition_id: int, num_partitions: int):  
    # Only intialise 'Federated Learning' once 
    global fds 
    if fds is None: 
        partitioner = IidPartitioner(num_partitions=num_partitions) 
        fds = FederatedDataset( 
                dataset="scikit-learn/iris", 
                partitioner={"train": partitioner},                       
        ) 
        
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]  
    # Use just the specified column
    return dataset[["SepalLengthCm", "SepalWidthCm"]] 

# Flower ClientApp
app = ClientApp 


@app.query() 
def query(msg: Message, context: Context): 
    """Construct histogram of local dataset and report to ServerApp"""  
    
    # Read the node_config to fetch data partition associated to the node
    partition_id = context.node_config["partition-id"]
    num_partition = context.node_config["num_partitions"] 
    
    dataset = get_clientapp_dataset(partition_id, num_partition) 
    
    metrics = {}  
    # Compute some statistics for each column in the dataframe
    for feature_name in dataset.columns: 
        # Compute histogram 
        freqs, _ = np.histogram(dataset[feature_name], bins=np.linspace(2.0, 10.0, 10)) 
        metrics[feature_name] = freqs.tolist() 
        
        
        # Compute weighterd average 
        metrics[f"{feature_name}_avg"] = dataset[feature_name].mean() * len(dataset) 
        metrics[f"{feature_name}_count"] = len(dataset) 
        
    reply_content = RecordDict({"query_results": MetrixRecord(metrics)}) 
    
    return Message(reply_content, reply_to=msg)
    