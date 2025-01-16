hod
    def from_pretrained(cls, model_type: str):
        """
        Load parameters from pre-trained model.
        """
        
        assert model_type in ['gpt2'], "Model type must be 'gpt2'"
        
        config = GPTConfig() 
        model = GPT2(config) 
        
        sd = model.state_dict()
        
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()
        hf_sd_keys = list(hf_sd.keys())
        
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        
        for k, v in sd.items(): 
            if k in hf_sd_keys: 
                # if the key is in the hf model, we can just copy it over
                # if the key in transposed, we need to transpose the weight.
                if any(k.endswith(w) for w in transposed):
                    # check shape. 
                    assert v.shape == hf_sd[k].T.shape, f"Shape mismatch key transposed {k}: {v.shape} and {hf_sd[k].T.shape}"
                    with torch.no_grad():
                        sd[k].copy_(hf_sd[k].T)
                else:
                    assert v.shape == hf_sd[k].shape, f"Shape mismatch key {k}: {v.shape} and {hf_sd[k].shape}"
                    with torch.no_grad():
                        sd[k].copy_(hf_sd[k])
            else: 
                print(f"Key {k} not found in hf model")
        
        model.load_state_dict(sd)
        print("Model loaded successfully")
        return model