from model import nanoGPT
from data import test_loader, train_loader, vocab_len, max_len, char2idx, idx2char, encoder, decoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def train_script(model, train_loader, loss_fn, optimizer, n_epochs, device): 
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
            test_loss = eval_script(model, test_loader, loss_fn, device)
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

def eval_script(model, test_loader, loss_fn, device): 
    model.eval()
    total_loss = 0
    with torch.no_grad(): 
        for x, y in test_loader: 
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred.view(-1, vocab_len), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(test_loader)


def generate_text(model, start_text, max_len, device): 
    model.eval()
    start_text = torch.tensor(encoder(start_text), dtype=torch.long).unsqueeze(0).to(device)

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

# hyperparameters
d_model = 512   # dimension of model : embedding size
n_layers = 3   # number of layers
n_heads = 4   # number of heads in multihead attention
d_ff = 4*d_model     # dimension of feedforward network | 4 times d_model
dropout = 0.1   # dropout rate
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 100

# create model and training script 
nano_gpt = nanoGPT(vocab_len, d_model, n_layers, n_heads, d_ff, dropout)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nano_gpt.parameters(), lr=lr)


nano_gpt = train_script(nano_gpt, train_loader, loss_fn, optimizer, n_epochs, device)

# save model 
torch.save(nano_gpt.state_dict(), "nanoGPT.pth")


start_text = '16. Thúy Kiều là chị, em là Thúy Vân.' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generated_text = generate_text(nano_gpt, start_text, max_len, device)
generated_text = generated_text.squeeze(0).cpu().numpy()
generated_text = decoder(generated_text)
print(generated_text)