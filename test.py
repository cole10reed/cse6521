import torch
import numpy
from segment_anything import sam_model_registry
    
def main():
    model_type = 'vit_h'
    sam = sam_model_registry[model_type](checkpoint='tuned_models/model_15.pth')
    sam_tuned = sam_model_registry[model_type](checkpoint='tuned_models/model_4.pth')#.to(device = gpu_device)

    #store params
    paramdic_base = {}
    paramdic_tuned = {}
    print('Storing params', flush=True)

    #sam has an iterator to iterate over params. Save them in corresponding dic
    for (param_name, param) in sam.mask_decoder.named_parameters():
        paramdic_base[param_name] = param
        # print('Model 15:', param_name)
    
    for(param_name, param) in sam_tuned.mask_decoder.named_parameters():
        paramdic_tuned[param_name] = param
        # print('Model 4:', param_name)

    print('Initialization complete.', flush=True)
    #### look for changes in parameters
    weights_updated = False
    weight_diff = {}
    for (param_name, param) in paramdic_base.items():
        # print('Base: ', paramdic_base[param_name])
        # print('Tuned:', paramdic_tuned[param_name])
        if not paramdic_tuned[param_name].equal(param): # tensor.equal returns true if tensor is exactly the same (values and all other elements of tensor)
            weights_updated = True
            print(paramdic_base[param_name])
            print(paramdic_tuned[param_name])
            weight_diff[param_name] = paramdic_tuned[param_name].eq(param) # this goes element by element and returns a tensor of true and falses corresponding to which exact elements have changed (i.e are different)
    
    if not weights_updated: # not one change in parameter detected
        print('Model weights are exactly the same')
    else:
        print('WOOHOO!!!! Different Weights!!')

if __name__ == '__main__':
    
    main()