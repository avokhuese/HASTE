"""
HASTE: Hierarchical Attention-based Stacked Ensemble for Fraud Detection
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleAttentionHASTE(nn.Module):
    """
    Simplified HASTE model that works reliably
    Uses attention mechanism to weight base models based on input features
    """
    
    def __init__(self, n_base_models: int, input_size: int, 
                 hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.n_base_models = n_base_models
        
        # Attention network: learns which base models to trust based on input features
        self.attention_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_base_models),
            nn.Softmax(dim=-1)
        )
        
        # Feature extractor for additional context
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Extract feature dimension
        feature_dim = hidden_size // 2
        
        # Final classifier that combines attended base predictions with extracted features
        # attended_predictions adds 1 dimension, extracted_features adds feature_dim dimensions
        classifier_input_size = 1 + feature_dim + n_base_models  # Include original base predictions too
        
        print(f"DEBUG: n_base_models={n_base_models}, input_size={input_size}, hidden_size={hidden_size}")
        print(f"DEBUG: feature_dim={feature_dim}, classifier_input_size={classifier_input_size}")
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, base_logits: torch.Tensor, 
                input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            base_logits: Base model predictions of shape (batch_size, n_base_models)
            input_features: Input features of shape (batch_size, input_size)
        
        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size = base_logits.shape[0]
        
        # Compute attention weights based on input features
        attention_weights = self.attention_network(input_features)
        
        # Apply attention to base model predictions
        attended_predictions = (attention_weights * base_logits).sum(dim=1, keepdim=True)
        
        # Extract additional features from input
        extracted_features = self.feature_extractor(input_features)
        
        # Combine attended predictions with extracted features AND original base logits
        combined = torch.cat([attended_predictions, extracted_features, base_logits], dim=1)
        
        # Debug shape information
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"DEBUG forward shapes:")
            print(f"  base_logits: {base_logits.shape}")
            print(f"  input_features: {input_features.shape}")
            print(f"  attention_weights: {attention_weights.shape}")
            print(f"  attended_predictions: {attended_predictions.shape}")
            print(f"  extracted_features: {extracted_features.shape}")
            print(f"  combined: {combined.shape}")
            print(f"  Expected classifier input: {self.classifier[0].in_features}")
            self._debug_printed = True
        
        # Final prediction
        predictions = self.classifier(combined)
        
        return predictions.squeeze(), attention_weights
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention weights for regularization"""
        epsilon = 1e-10
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + epsilon), dim=-1)
        return entropy.mean()


class HASTETrainer:
    """Trainer for HASTE model"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        
        # Set debug flag
        self.model._debug_printed = False
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            #verbose=True
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def train_epoch(self, train_loader, lambda_entropy: float = 0.01):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (base_logits, features, targets, _) in enumerate(train_loader):
            # Move to device
            base_logits = base_logits.to(self.device).float()
            features = features.to(self.device).float()
            targets = targets.to(self.device).float()
            
            # Forward pass
            predictions, attention_weights = self.model(base_logits, features)
            
            # Compute classification loss
            classification_loss = self.criterion(predictions, targets)
            
            # Compute attention entropy for regularization
            entropy_loss = self.model.compute_attention_entropy(attention_weights)
            
            # Total loss with entropy regularization
            loss = classification_loss - lambda_entropy * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, np.array(all_predictions), np.array(all_targets)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for base_logits, features, targets, _ in val_loader:
                # Move to device
                base_logits = base_logits.to(self.device).float()
                features = features.to(self.device).float()
                targets = targets.to(self.device).float()
                
                # Forward pass
                predictions, attention_weights = self.model(base_logits, features)
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, np.array(all_predictions), np.array(all_targets)
    
    def train(self, train_loader, val_loader, num_epochs: int = 20, 
             patience: int = 5, save_path: str = 'models/haste_best.pt'):
        """Main training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Training HASTE model...")
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train F1':<12} {'Val F1':<12}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_preds, train_targets = self.train_epoch(train_loader)
            train_f1 = self._calculate_f1(train_preds, train_targets)
            
            # Validate
            val_loss, val_preds, val_targets = self.validate(val_loader)
            val_f1 = self._calculate_f1(val_preds, val_targets)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"{epoch+1:<8} {train_loss:<12.6f} {val_loss:<12.6f} "
                      f"{train_f1:<12.4f} {val_f1:<12.4f}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                    'history': self.history
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        try:
            checkpoint = torch.load(save_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        except:
            print("\nTraining completed. Using final model.")
        
        return self.history
    
    def _calculate_f1(self, predictions: np.ndarray, targets: np.ndarray, 
                     threshold: float = 0.5) -> float:
        """Calculate F1 score"""
        from sklearn.metrics import f1_score
        preds_binary = (predictions > threshold).astype(int)
        return f1_score(targets, preds_binary, zero_division=0)
    
    def predict(self, test_loader, return_attention: bool = False):
        """Make predictions"""
        self.model.eval()
        all_predictions = []
        all_attention = [] if return_attention else None
        
        with torch.no_grad():
            for base_logits, features, _, _ in test_loader:
                # Move to device
                base_logits = base_logits.to(self.device).float()
                features = features.to(self.device).float()
                
                # Forward pass
                predictions, attention_weights = self.model(base_logits, features)
                
                all_predictions.extend(predictions.cpu().numpy())
                
                if return_attention:
                    all_attention.append(attention_weights.cpu().numpy())
        
        predictions_array = np.array(all_predictions)
        
        if return_attention:
            attention_array = np.concatenate(all_attention, axis=0) if all_attention else None
            return predictions_array, attention_array
        else:
            return predictions_array


def create_haste_model(n_base_models: int, input_size: int, 
                       hidden_size: int = 128, dropout: float = 0.3,
                       device: str = 'cpu'):
    """Create a HASTE model with appropriate configuration"""
    model = SimpleAttentionHASTE(
        n_base_models=n_base_models,
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout
    )
    
    # Print model architecture for debugging
    print(f"\nHASTE Model Architecture:")
    print(f"  Number of base models: {n_base_models}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Feature extractor output: {hidden_size // 2}")
    print(f"  Classifier input size: {1 + (hidden_size // 2) + n_base_models}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model.to(device)


# Alias for backward compatibility
HASTE = SimpleAttentionHASTE
SimpleHASTE = SimpleAttentionHASTE