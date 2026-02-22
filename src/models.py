"""Model definitions for Driver Activity Recognition pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TemporalAttentionPool(nn.Module):
    """Attention-based temporal pooling over LSTM hidden states.

    Learns a query vector to compute attention weights over timesteps,
    then returns the weighted sum of hidden states.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, return_attention=False):
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
            return_attention: if True, also return attention weights

        Returns:
            context: (batch, hidden_dim)
            attn_weights: (batch, seq_len) — only if return_attention=True
        """
        # (batch, seq_len, 1)
        scores = self.query(lstm_out)
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden_dim)

        if return_attention:
            return context, attn_weights.squeeze(-1)
        return context


class ActivityLSTM(nn.Module):
    """LSTM-based temporal model for activity classification.

    Supports bidirectional LSTM, LayerNorm, attention pooling,
    and Gaussian noise augmentation for regularization.
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2,
                 num_classes=34, lstm_dropout=0.3, fc_dropout=0.5,
                 use_layernorm=False, bidirectional=False,
                 pooling="last", noise_std=0.0):
        super().__init__()

        self.pooling = pooling
        self.noise_std = noise_std
        self.bidirectional = bidirectional

        # Optional LayerNorm on input features
        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Effective hidden dim doubles for bidirectional
        effective_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Temporal pooling
        if pooling == "attention":
            self.attention = TemporalAttentionPool(effective_dim)

        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(effective_dim, num_classes)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, input_dim) feature sequences
            return_attention: if True, also return attention weights

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, seq_len) — only if return_attention=True
        """
        # Gaussian noise augmentation during training
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # LayerNorm on input
        x = self.layernorm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, effective_dim)

        # Temporal pooling
        attn_weights = None
        if self.pooling == "attention":
            pooled, attn_weights = self.attention(lstm_out, return_attention=True)
        elif self.pooling == "mean":
            pooled = lstm_out.mean(dim=1)
        else:  # "last"
            pooled = lstm_out[:, -1, :]

        out = self.dropout(pooled)
        logits = self.fc(out)

        if return_attention and attn_weights is not None:
            return logits, attn_weights
        return logits


class CNNLSTMModel(nn.Module):
    """End-to-end CNN+LSTM model for inference/demo.

    Combines frozen ResNet-18 feature extraction with LSTM classifier.
    """

    def __init__(self, num_classes=34, hidden_dim=256, num_layers=2,
                 lstm_dropout=0.3, fc_dropout=0.5,
                 use_layernorm=False, bidirectional=False,
                 pooling="last", noise_std=0.0):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = ActivityLSTM(
            input_dim=512,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            lstm_dropout=lstm_dropout,
            fc_dropout=fc_dropout,
            use_layernorm=use_layernorm,
            bidirectional=bidirectional,
            pooling=pooling,
            noise_std=noise_std,
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
