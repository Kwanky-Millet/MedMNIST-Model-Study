from medmnist import INFO, Evaluator

import torch

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.01, mode='min', path='checkpoints/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = float('inf')
        self.val_score_max = -float('inf')
        self.delta = delta
        self.mode = mode
        self.path = path

    def __call__(self, val_score, model):
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'Validation Metric Improved. Saving Model...')
        torch.save(model.state_dict(), self.path)

def train_model(model, train_loader, num_epochs, criterion, optimizer, scheduler, data_flag, val_loader=None, patience=5, mode='max'):
    early_stopping = EarlyStopping(patience=patience, mode=mode, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)

            loss.backward(retain_graph=True)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
            
        if val_loader:
            metrics = test_model(model, val_loader, 'val', data_flag)
            val_score = metrics[0]
            
            early_stopping(val_score, model)
            
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training...")
                break

def test_model(model, data_loader, split, data_flag):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
        
        split = "Validation" if split == 'val' else "Test"
        print('%s  AUC: %.3f  Accuracy:%.3f' % (split, *metrics))
        
        return metrics