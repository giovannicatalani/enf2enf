

class ReduceLROnPlateau:
    """Learning rate scheduler that reduces LR when metric plateaus."""
    def __init__(self, 
                 initial_lr: float,
                 factor: float = 0.1,
                 patience: int = 10,
                 min_lr: float = 1e-6,
                 min_delta: float = 1e-4):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.wait = 0
        
    def should_reduce(self, current_loss: float) -> bool:
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        
        self.wait += 1
        if self.wait >= self.patience:
            if self.current_lr * self.factor >= self.min_lr:
                self.current_lr *= self.factor
                self.wait = 0
                return True
        return False
        
    def get_lr(self) -> float:
        return self.current_lr