import torch
import torch.nn as nn
from torchvision import models

class ModelList:

    def get_model(name:str):
        '''
        This function is used to return the model object as per the given name.

        Args:
            
            name (str): The name of the model

        Returns:

            model (nn.Module): The model object corresponding to the given name
        '''
        if name == 'resnet50':
            return ResNet50LSTMClassifier()
        else:
            return None

class ResNet50LSTMClassifier(nn.Module):
    def __init__(self, num_classes=2, hidden_size=512, num_layers=1, dropout=0.5):
        super(ResNet50LSTMClassifier, self).__init__()
        
        # Load pre-trained ResNet-50 model
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final fully connected layer and average pooling
        self.resnet50_backbone = nn.Sequential(*list(resnet50.children())[:-2])
        
        # Define a new fully connected layer to produce the correct feature size
        self.fc = nn.Linear(2048, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape input for ResNet-50
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features using ResNet-50
        features = self.resnet50_backbone(x)
        features = torch.mean(features, dim=[2, 3])  # Global average pooling
        
        # Reshape features to fit the fully connected layer
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, num_features)
        features = self.fc(features)  # (batch_size, seq_len, hidden_size)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, hidden_size)
        
        # Use the output from the final timestep for classification
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification
        out = self.classifier(lstm_out)  # (batch_size, num_classes)
        
        return out