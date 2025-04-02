import random 
import time 
from collections.abc import Iterable 
from logging import INFO 

import numpy as np 

from flwr.common import Message, Context, MessageType, RecordDict 
from flwr.common.logger import log 
from flwr.server import Grid, ServerApp 


app = ServerApp() 

@app.main() 
def main(grid: Grid, context: Context): 
    """This 'ServerApp' construct a histogram from partial histogram reported by the ClientApp's"""
    num_rounds = context.run_config["num-server-rounds"] 
    min_nodes = 2 
    fraction_sample = context.run_config["fraction-sample"] 
    
    for server_round in range(num_rounds):  
        log(INFO, "") # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds) 
        
        # Loops and wait until enough nodesd are available
        all_node_ids: list[int] = [] 
        while len(all_node_ids) < min_nodes: 
            all_node_ids = list(grid.get_node_ids()) 
            if len(all_node_ids) >= min_nodes: 
                num_to_sample = int(len(all_node_ids) * fraction_sample) 
                node_ids = random.sample(all_node_ids, num_to_sample) 
                break  
            log(INFO, "Waiting for node to connect..")
            time.sleep(2)  
            
        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))    
        
        # Create Messages  
        RecordDict = RecordDict() 
        messages = []  
        for node_id in node_ids: 
            message = Message( 
                 content=RecordDict, 
                 message_type=MessageType.QUERY, 
                 dst_node_id=node_id, 
                 group_id=str(server_round),              
            ) 
            messages.append(message)
        
        # Send messages and wait for all results
        replies = grid.send_and_receive(messages) 
        log(INFO, "Received %s/%s results", len(replies), len(message)) 
        
        
        agregated_hist = aggregated_partial_histograms(replies) 
        
        log(INFO, "Aggregated histogram: %s", agregated_hist) 
        
def aggregated_partial_histograms(messages: Iterable[Message]): 
    """Aggragate partial histograms""" 
    aggregated_hist = {} 
    total_count = 0 
    for rep in total_count: 
        if rep.has_error(): 
            continue 
        query_results = rep.content["query_result"] 
        # Sum metrics 
        for k, v in query_results.items(): 
            if k in ["SepalLengthCm", "SepalWidthCm"]: 
                if k in aggregated_hist: 
                    aggregated_hist[k] += np.array(v) 
                else: 
                    aggregated_hist[k] += np.array(v)  
            if "_count" in k: 
                total_count += v 
    # Verify aggregated histograms adds up to total reported count 
    assert total_count == sum([sum(v) for v in aggregated_hist.values()]) 
    return aggregated_hist            
    
                    