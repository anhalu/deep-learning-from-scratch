import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model import init_model
from data import DataProcessor, Config
import matplotlib.pyplot as plt
from tqdm import tqdm
from tokenizer import BytePairEncoding, RegexBytePairEncoding



def train_script(model, train_loader, test_loader, loss_fn, optimizer, n_epochs, device, vocab_len): 
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(n_epochs), desc='Epochs'): 
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc='Batches'): 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x) 
            loss = loss_fn(y_pred.view(-1, vocab_len), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))
        
        # evaluate the model on test data
        if (epoch + 1) % 10 == 0: 
            test_loss = eval_script(model, test_loader, loss_fn, device, vocab_len)
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}, Test Loss: {test_loss}")
        
            # if test loss is not decreasing, stop training
            if len(test_losses) > 1 and test_losses[-1] >= test_losses[-2]: 
                print("Test loss is not decreasing anymore. Stop training.")
                break     
            
        else: 
            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")
    
    # Plot the training and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(range(9, n_epochs, 10), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return model

def eval_script(model, test_loader, loss_fn, device, vocab_len): 
    model.eval()
    total_loss = 0
    with torch.no_grad(): 
        for x, y in test_loader: 
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred.view(-1, vocab_len), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(test_loader)


def generate_text(model, start_text, max_len, device, encoder): 
    model.eval()
    start_text = torch.tensor(encoder(start_text), dtype=torch.long).unsqueeze(0).to(device)
    model = model.to(device)
    with torch.no_grad(): 
        for _ in range(max_len): 
            y_pred = model(start_text)
            y_pred = y_pred[:, -1, :]
            next_char = torch.argmax(y_pred, dim=-1).unsqueeze(0)
            start_text = torch.cat([start_text, next_char], dim=1)

            # check max length. 
            if len(start_text[0]) + 1 >= max_len:
                break
    return start_text

def train_script_ddp(rank, world_size, model, train_loader, test_loader, loss_fn, optimizer, n_epochs, device, tokenizer, patience=3):
    # Setup DDP
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # Wrap model
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    # Training setup
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    counter = 0
    
    # Early stopping tensor
    stop_signal = torch.zeros(1, device=device)
    
    for epoch in tqdm(range(n_epochs), desc=f'Epochs Rank {rank}'):
        model.train()
        epoch_loss = 0
        
        # Training loop
        for x, y in tqdm(train_loader, desc=f'Batches Rank {rank}', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation and early stopping logic on rank 0
        if rank == 0:
            if (epoch + 1) % 2 == 0:
                test_loss = eval_script(model.module, test_loader, loss_fn, device, tokenizer.vocab_size)
                test_losses.append(test_loss)
                print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
                
                # Early stopping check
                if test_loss < best_loss:
                    best_loss = test_loss
                    counter = 0
                else:
                    counter += 1
                    
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    stop_signal = torch.ones(1, device=device)
            
            print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        
        # Broadcast stop signal to all processes
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break
    
    # Cleanup
    dist.destroy_process_group()
    
def main(): 
    # init tokenizer. 
    tokenizer = RegexBytePairEncoding(vocab_size=500)
    # tokenizer.load_tokenizer("tokenizer_regex.json")
    
    # load data and create dataloader
    data_processor = DataProcessor(tokenizer=tokenizer) 
    data = data_processor.load_data()
    
    # create tokenizer 
    # tokenizer.train(data)
    tokenizer.save_tokenizer("tokenizer_regex_vocab_size_500.json")
    
    train_loader, test_loader = data_processor.get_dataloaders()   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nano_gpt = init_model(vocab_len=tokenizer.vocab_size)
    
    lr = 1e-4
    n_epochs = 50
    optimizer = torch.optim.Adam(nano_gpt.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    world_size = torch.cuda.device_count()  # Number of GPUs
    print("Number of GPUs: ", world_size)
    mp.spawn(
        train_script_ddp,
        args=(world_size, nano_gpt, train_loader, test_loader, loss_fn, optimizer, n_epochs, device, tokenizer),
        nprocs=world_size,
        join=True
    )
    

    # save model 
    torch.save(nano_gpt.state_dict(), "nanoGPT_multi_gpu_layer_6_head_8.pth")
    
    # load model 
    nano_gpt.load_state_dict(torch.load("nanoGPT_multi_gpu_layer_6_head_8.pth"))
    
    # infer test
    start_text = '16. Thúy Kiều là chị, em là Thúy Vân.' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated_text = generate_text(nano_gpt, start_text, Config.MAX_LEN, device, encoder=tokenizer.encode)
    generated_text = generated_text.squeeze(0).cpu().numpy()
    generated_text = tokenizer.decode(generated_text)
    print(generated_text)
    
if __name__ == '__main__':
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main()