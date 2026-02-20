"""Model definitions for Driver Activity Recognition pipeline."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CNNFeatureExtractor(nn.Module):
    """Frozen ResNet-18 for spatial feature extraction.

    Removes the final FC layer and returns 512-dim feature vectors.
    All parameters are frozen (no gradient computation).
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove final FC layer, keep everything up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W) input images

        Returns:
            (batch, 512) feature vectors
        """
        with torch.no_grad():
            features = self.features(x)
            features = features.squeeze(-1).squeeze(-1)  # (batch, 512)
        return features


class ActivityLSTM(nn.Module):
    """LSTM-based temporal model for activity classification.

    Takes a sequence of CNN features and outputs class predictions.
    Uses the hidden state from the last timestep for classification.
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2,
                 num_classes=34, lstm_dropout=0.3, fc_dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) feature sequences

        Returns:
            (batch, num_classes) logits
        """
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


class CNNLSTMModel(nn.Module):
    """End-to-end CNN+LSTM model for inference/demo.

    Combines frozen ResNet-18 feature extraction with LSTM classifier.
    """

    def __init__(self, num_classes=34, hidden_dim=256, num_layers=2,
                 lstm_dropout=0.3, fc_dropout=0.5):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = ActivityLSTM(
            input_dim=512,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            lstm_dropout=lstm_dropout,
            fc_dropout=fc_dropout,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, T, C, H, W) video frames

        Returns:
            (batch, num_classes) logits
        """
        batch_size, T = x.shape[0], x.shape[1]

        # Extract features for each frame
        # Reshape to (batch*T, C, H, W)
        x_flat = x.view(batch_size * T, *x.shape[2:])
        features = self.cnn(x_flat)  # (batch*T, 512)

        # Reshape back to (batch, T, 512)
        features = features.view(batch_size, T, -1)

        # Pass through LSTM
        logits = self.lstm(features)
        return logits
