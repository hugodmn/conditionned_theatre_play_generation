import torch 
import torch.nn as nn 
from tqdm import tqdm 
#from dataset import ShakespeareDataset
from dataset import OpenWebText, TheatreDataset, DialogueDataset
from torch.utils.data import DataLoader, Subset
from model_V2 import LLM, Config
import torch.nn.functional as F
import wandb



eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200



learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

############################################################################################################

def step(model, optimizer, scheduler, train_loader, test_dataset,  device, epoch, path, best_loss,eval_steps = 50, result_dict = None):
    model.train()
    total_acc = 0
    total_loss = 0
    sample_nb = 0
    loss_for_n_steps = 0
    

    with tqdm(range(len(train_loader))) as pbar :
        for idx, (context,targets) in enumerate(train_loader):
            context, targets = context.to(device), targets.to(device)
  
            logits = model(context)
            B,T,C = logits.shape
            #B,T = targets.shape
            logits= logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
           
            #loss.backward()
            scaler.scale(loss).backward()
            loss_for_n_steps += loss.item()
            total_acc += (logits.argmax(-1) == targets).sum().item()
            sample_nb += B
            total_loss += loss.item()

            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            pbar.update(1)

            if idx % eval_steps == 0 and idx != 0:
                print(f'loss for step {idx} : {loss_for_n_steps/eval_steps}')
                wandb.log({"train_accuracy": total_acc*100/(sample_nb*T),"train_loss": loss_for_n_steps/eval_steps, "step": idx})
                loss_for_n_steps = 0
                test_loss, best_loss = test_for_n_steps(model, test_dataset, device, path, best_loss, epoch, n_steps = 200, step = idx)
                total_acc = 0
                input = test_dataset.encode("begining\n")
                #input = input.split(' ')
                input = torch.Tensor(input).type(torch.int32).to(device).unsqueeze(0)
                output = model.generate(input, 200).squeeze(0).cpu()
                
                
                
                decoded_output = test_dataset.decode(output)
                decoded_output = ''.join(decoded_output)

                print(decoded_output)

                # final_output = ""

                scheduler.step()

           
              
            
            
            
            
            


    #print(f'[TRAIN EPOCH {epoch}] Accuracy : {total_acc*100/(sample_nb*T)}% Train Loss : {total_loss/len(train_loader)}')
    

def test_for_n_steps(model, test_dataset, device, path, best_loss, epoch, n_steps, step, batch_size=64):
        # Create a subset of the CIFAR10 dataset
    subset_indices = torch.randperm(len(test_dataset))[:n_steps*batch_size]

    subset_dataset = Subset(test_dataset, subset_indices)


    # Create the subset DataLoader
    subset_dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    model.eval()

    # Rest of your code
    with torch.no_grad():
        total_acc_test = 0 
        total_loss_test = 0
        sample_nb_test = 0
        
        with tqdm(range(n_steps)) as pbar :
            for idx,(context, targets) in enumerate(subset_dataloader):
                #context, targets = next(iter(test_loader))
                context, targets = context.to(device), targets.to(device)
                logits = model(context)
                B,T,C = logits.shape
                #B,T = targets.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

                sample_nb_test += B

                total_acc_test += (logits.argmax(-1) == targets).sum().item()
                total_loss_test += loss.item()

      
                pbar.update(1)
            if total_loss_test/n_steps < best_loss :
                best_loss = total_loss_test/n_steps
                torch.save(model.state_dict(), path+'model.pt')
        print(f'[TEST EPOCH {epoch}] Accuracy : {total_acc_test*100/(sample_nb_test*T)}% Test Loss : {total_loss_test/n_steps} Best Loss : {best_loss}')
        wandb.log({"test_accuracy": total_acc_test*100/(sample_nb_test*T), "test_loss": total_loss_test/n_steps, 'step': step})
        return total_loss_test/n_steps, best_loss
    


if __name__ == '__main__' : 

    pretraining = False


    epochs = 200
    device = 'mps'
    print_all_vocab = False
    block_size = 512 # -> context length : char : 256, word : 64

    emb_size = 384
    head_nb = 6
    block_nb = 6

    lr = 1e-3 if pretraining else 1e-4

    beta1 = 0.9
    beta2 = 0.95


# eval_interval = 250 # keep frequent because we'll overfit
# eval_iters = 200

# n_layer = 6
# n_head = 6
# n_embd = 384

# #For pretraining 0.0 then 0.1
# dropout = 0.0

# batch_size = 64
# block_size = 512 # context of up to 256 previous characters


# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small




    train_dataset = DialogueDataset(dataset_path = 'dailydialogue_dataset', split_str='train')
    test_dataset = DialogueDataset(dataset_path = 'dailydialogue_dataset', split_str='test')



    LLM_config = Config(
        vocab_size = train_dataset.vocab_size,
        emb_size = emb_size,
        head_nb = head_nb,
        block_nb = block_nb,
        block_size = block_size,
        dropout=0.0 if pretraining else 0.1,
        )

    run = wandb.init(
    # Set the project where this run will be logged
    project="llm_finetuning",
    # Track hyperparameters and run metadata
    config={
        "device": device,
        "learning_rate": lr,
        "head_nb": head_nb,
        "block_nb": block_nb,
        "emb_size": emb_size,
        "block_size": block_size,   
    },
)
    
    dtype = 'bfloat16'
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    if pretraining:
        batch_sz = 64
    else :
        batch_sz = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=True, num_workers=0)

    vocab_size = train_dataset.vocab_size#len(train_dataset.stoi)
    LLM_config.vocab_size = vocab_size
    best_loss = 10000
    print(f'Vocab size : {vocab_size}')

    #vocab_size, emb_size : int, head_nb : int, block_nb : int, block_size : int, tokenizer_type : str, train_test_split : float = 0.9):

 
    model = LLM(LLM_config).to(device) 
    model.load_state_dict(torch.load("checkpoints/best_model.pt"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas = (beta1, beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
    #Save config attributes in a json file
    path = 'checkpoints_ft/'

    print('---------------------Starting training---------------------')
    for epoch in range(epochs):
        step(model, optimizer, scheduler, train_loader, test_dataset, device, epoch, path, best_loss = 1000)
        #best_loss = test(model, test_loader, device, best_loss, epoch)