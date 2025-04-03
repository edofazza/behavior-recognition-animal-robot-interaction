import torch
import torch.nn as nn
import torch.nn.functional as F



class MultilabelDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super(MultilabelDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        # Distillation loss: Use sigmoid activation (no softmax in multilabel problems)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        student_probs = torch.sigmoid(student_logits / self.temperature)

        # Use mean squared error (MSE) as a proxy for KL divergence in multilabel distillation
        distill_loss = F.mse_loss(student_probs, teacher_probs)

        # Binary cross-entropy loss for the true labels
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Total loss combining distillation and hard-label loss
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss
        return total_loss


class MultilabelDistillationLossWithTemperatureAnnealing(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        super(MultilabelDistillationLossWithTemperatureAnnealing, self).__init__()
        self.starting_temperature = temperature
        self.final_temperature = 1.0
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels, epoch, num_epochs):
        temperature = self.starting_temperature - (self.starting_temperature - self.final_temperature) * (
                    epoch / num_epochs)
        # Distillation loss: Use sigmoid activation (no softmax in multilabel problems)
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        student_probs = torch.sigmoid(student_logits / temperature)

        # Use mean squared error (MSE) as a proxy for KL divergence in multilabel distillation
        distill_loss = F.mse_loss(student_probs, teacher_probs)

        # Binary cross-entropy loss for the true labels
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Total loss combining distillation and hard-label loss
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss
        return total_loss



def dynamic_temperature(student_logits, teacher_logits, base_temperature=5.0, min_temperature=1.0):
    """
    Adjust temperature dynamically based on the discrepancy between student and teacher logits.

    Args:
    - student_logits (torch.Tensor): Logits from the student model.
    - teacher_logits (torch.Tensor): Logits from the teacher model.
    - base_temperature (float): Initial temperature.
    - min_temperature (float): Minimum allowed temperature.

    Returns:
    - temperature (float): Adjusted temperature for distillation.
    """
    # Compute the mean absolute error (MAE) between student and teacher logits
    discrepancy = torch.mean(torch.abs(teacher_logits - student_logits)).item()

    # Dynamically adjust temperature based on discrepancy
    temperature = base_temperature / (1 + discrepancy)

    # Ensure the temperature doesn't fall below a minimum threshold
    temperature = max(temperature, min_temperature)

    return temperature


class MultilabelDistillationLossWithDynamicTemperature(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        super(MultilabelDistillationLossWithDynamicTemperature, self).__init__()
        self.base_temperature = temperature
        self.min_temperature = 1.0
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        temperature = dynamic_temperature(student_logits, teacher_logits,
                                          base_temperature=self.base_temperature, min_temperature=self.min_temperature)
        # Distillation loss: Use sigmoid activation (no softmax in multilabel problems)
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        student_probs = torch.sigmoid(student_logits / temperature)

        # Use mean squared error (MSE) as a proxy for KL divergence in multilabel distillation
        distill_loss = F.mse_loss(student_probs, teacher_probs)

        # Binary cross-entropy loss for the true labels
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Total loss combining distillation and hard-label loss
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss
        return total_loss


class DistillationWithGaussianNoiseAndDynamicTemperature(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5, noise_std=0.1):
        super(DistillationWithGaussianNoiseAndDynamicTemperature, self).__init__()
        self.base_temperature = temperature
        self.min_temperature = 1.0
        self.alpha = alpha
        self.noise_std = noise_std
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        # Inject Gaussian noise into the teacher logits
        noise = torch.randn_like(teacher_logits) * self.noise_std
        noisy_teacher_logits = teacher_logits + noise

        temperature = dynamic_temperature(student_logits, noisy_teacher_logits,
                                          base_temperature=self.base_temperature, min_temperature=self.min_temperature)
        # Distillation loss: Soft predictions from noisy teacher
        teacher_probs = torch.sigmoid(noisy_teacher_logits / temperature)
        student_probs = torch.sigmoid(student_logits / temperature)
        distill_loss = F.mse_loss(student_probs, teacher_probs)

        # Hard-label loss (supervised)
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Combine the two losses
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss
        return total_loss


class DistillationWithGaussianNoiseAndTemperatureAnnealing(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5, noise_std=0.1):
        super(DistillationWithGaussianNoiseAndTemperatureAnnealing, self).__init__()
        self.starting_temperature = temperature
        self.final_temperature = 1.0
        self.alpha = alpha
        self.noise_std = noise_std
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels, epoch, num_epochs):
        # Inject Gaussian noise into the teacher logits
        noise = torch.randn_like(teacher_logits) * self.noise_std
        noisy_teacher_logits = teacher_logits + noise

        temperature = self.starting_temperature - (self.starting_temperature - self.final_temperature) * (
                    epoch / num_epochs)
        # Distillation loss: Soft predictions from noisy teacher
        teacher_probs = torch.sigmoid(noisy_teacher_logits / temperature)
        student_probs = torch.sigmoid(student_logits / temperature)
        distill_loss = F.mse_loss(student_probs, teacher_probs)

        # Hard-label loss (supervised)
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Combine the two losses
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss
        return total_loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: Features from the student model [batch_size, feature_dim].
            teacher_features: Features from the teacher model [batch_size, feature_dim].
        """
        device = student_features.device

        # Normalize features
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        # Concatenate student and teacher features
        features = torch.cat([student_features, teacher_features], dim=0)

        # Compute cosine similarities
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels - positive pairs are (i, i + batch_size)
        batch_size = student_features.size(0)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(device)

        # Mask to remove self-similarities
        mask = (~torch.eye(2 * batch_size, device=device).bool()).float()

        # Apply mask to remove diagonal (self-similarity)
        similarity_matrix = similarity_matrix * mask

        # Compute loss
        logits = similarity_matrix / self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss


class DistillationWithContrastiveNTXent(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        super(DistillationWithContrastiveNTXent, self).__init__()
        self.starting_temperature = temperature
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.nt_xent_loss = NTXentLoss(temperature=0.07)

    def forward(self, student_logits, teacher_logits, true_labels):
        ntxent_loss = self.nt_xent_loss(student_logits, teacher_logits)

        # Hard-label loss (supervised)
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Combine the two losses
        total_loss = self.alpha * ntxent_loss + (1.0 - self.alpha) * ce_loss
        return total_loss


class NTXentLossWithTemperatureAnnealing(nn.Module):
    def __init__(self, initial_temperature=0.07, min_temperature=0.01, total_epochs=300):
        super(NTXentLossWithTemperatureAnnealing, self).__init__()
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.total_epochs = total_epochs

    def anneal_temperature(self, epoch):
        # Reduce temperature according to the annealing rate
        self.temperature = self.temperature - (self.temperature - self.min_temperature) * (epoch / self.total_epochs)

    #def exponential_annealing_temperature(epoch, T0=0.07, T_final=0.01, total_epochs=300):
    #    return T0 * (T_final / T0) ** (epoch / total_epochs)

    def forward(self, student_features, teacher_features, epoch):
        self.anneal_temperature(epoch)

        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        # Concatenate student and teacher features
        features = torch.cat([student_features, teacher_features], dim=0)

        # Cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels
        batch_size = student_features.size(0)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(
            student_features.device)

        # Mask for removing self-similarities
        mask = (~torch.eye(2 * batch_size, device=student_features.device).bool()).float()
        similarity_matrix = similarity_matrix * mask

        logits = similarity_matrix / self.temperature
        loss = F.cross_entropy(logits, labels)

        return loss


class DistillationWithContrastiveNTXentAndTemperatureAnnealing(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        super(DistillationWithContrastiveNTXentAndTemperatureAnnealing, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.nt_xent_loss = NTXentLossWithTemperatureAnnealing()

    def forward(self, student_logits, teacher_logits, true_labels, epoch):
        ntxent_loss = self.nt_xent_loss(student_logits, teacher_logits, epoch)

        # Hard-label loss (supervised)
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Combine the two losses
        total_loss = self.alpha * ntxent_loss + (1.0 - self.alpha) * ce_loss
        return total_loss


class NTXentLossWithDynamicTemperature(nn.Module):
    def __init__(self, initial_temperature=0.07, min_temperature=0.01):
        super(NTXentLossWithDynamicTemperature, self).__init__()
        self.base_temperature = initial_temperature
        self.min_temperature = min_temperature

    def forward(self, student_features, teacher_features):
        temperature = dynamic_temperature(student_features, teacher_features,
                                          base_temperature=self.base_temperature,
                                          min_temperature=self.min_temperature)

        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        # Concatenate student and teacher features
        features = torch.cat([student_features, teacher_features], dim=0)

        # Cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # Create labels
        batch_size = student_features.size(0)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(
            student_features.device)

        # Mask for removing self-similarities
        mask = (~torch.eye(2 * batch_size, device=student_features.device).bool()).float()
        similarity_matrix = similarity_matrix * mask

        logits = similarity_matrix / temperature
        loss = F.cross_entropy(logits, labels)

        return loss


class DistillationWithContrastiveNTXentAndDynamicTemperature(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        super(DistillationWithContrastiveNTXentAndDynamicTemperature, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.nt_xent_loss = NTXentLossWithDynamicTemperature()

    def forward(self, student_logits, teacher_logits, true_labels):
        ntxent_loss = self.nt_xent_loss(student_logits, teacher_logits)

        # Hard-label loss (supervised)
        ce_loss = self.bce_loss(student_logits, true_labels.float())

        # Combine the two losses
        total_loss = self.alpha * ntxent_loss + (1.0 - self.alpha) * ce_loss
        return total_loss
