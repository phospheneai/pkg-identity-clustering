import torch
import torch.nn as nn
from torchvision import models
from abc import abstractmethod, ABC

class ModelList:

    def get_model(name:str, model_path:str, device:str):
        '''
        This function is used to return the model object as per the given name.

        Args:
            
            name (str): The name of the model

        Returns:

            model (nn.Module): The model object corresponding to the given name
        '''
        if name == 'resnet50_lstm_num_layer2':
            model = ResNet50LSTMClassifier(num_layers=2)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            return model
        if name == 'resnet50_lstm_num_layer1':
            model = ResNet50LSTMClassifier(num_layers=1)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            return model
        else:
            return None
        
class BaseModel(ABC):

    @abstractmethod
    def get_transform(self):
        pass


class ResNet50LSTMClassifier(nn.Module):
    def __init__(self, num_classes=1,hidden_size=512, num_layers=2, dropout=0.5):
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
    
    def forward(self, x,hidden=None):
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
        lstm_out, hidden = self.lstm(features,hidden)  # (batch_size, seq_len, hidden_size)
        
        # Use the output from the final timestep for classification
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification
        out = self.classifier(lstm_out)  # (batch_size, 1)
        
        return out,hidden 

    def init_hidden(self, batch_size, device):
        # Initialize hidden states (h0, c0) for LSTM
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)
