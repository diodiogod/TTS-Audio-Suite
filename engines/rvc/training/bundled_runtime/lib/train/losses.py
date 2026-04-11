from collections import namedtuple
from typing import Callable, List, Literal, Optional
import librosa
import numpy as np
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

from .mel_processing import mel_spectrogram_torch, spectral_de_normalize_torch

from ..utils import gc_collect
from ..infer_pack.commons import compute_correlation, median_pool1d, minmax_scale

class LossBalancer:
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, initial_weights: dict={}, historical_losses: dict={}, ema_weights: dict={}, epsilon=1e-8, weights_decay=0., loss_decay=.0, active=True, use_pareto=True, use_norm=False):
        """
        Initializes the LossBalancer with optional initial weights, partial gradient computation, and EMA.
        
        Args:
            model (torch.nn.Module): The model for which gradients will be computed.
            initial_weights (dict): Initial weights for each loss term.
            epsilon (float): A small value to avoid division by zero.
            partial (bool): Whether to compute gradients only for the output layer.
        """
        self.model = model
        self.epsilon = epsilon
        self.weights_decay = weights_decay
        self.loss_decay = loss_decay
        self.initial_weights = initial_weights
        self.ema_weights = ema_weights
        self.historical_losses = historical_losses
        self.active = active
        self.use_pareto = use_pareto # apply the 20/80 rule to loss balancing
        self.use_norm = use_norm

    def to_dict(self):
        return dict(
            epsilon = self.epsilon,
            weights_decay = self.weights_decay,
            loss_decay = self.loss_decay,
            ema_weights = self.ema_weights,
            initial_weights = self.initial_weights,
            historical_losses = self.historical_losses,
            active = self.active,
            use_pareto = self.use_pareto, # apply the 20/80 rule to loss balancing
            use_norm = self.use_norm
        )

    def update_ema_weights(self, new_weights: dict):
        """Updates the EMA of the weights."""
        if not self.ema_weights: self.ema_weights = dict(**new_weights)
        else: self.ema_weights = {
                k: np.nan_to_num(
                    self.weights_decay * self.ema_weights.get(k, 1.) + (1 - self.weights_decay) * new_weights[k],nan=self.epsilon)
                for k in new_weights
            }
        return dict(**self.ema_weights)
    
    def update_historical_losses(self, new_losses: dict):
        """Updates the EMA of the weights."""
        if not self.historical_losses: self.historical_losses = dict(**new_losses)
        else: 
            for k,v in new_losses.items():
                self.historical_losses[k]=np.nan_to_num(
                    self.loss_decay*self.historical_losses.get(k, v) + (1 - self.loss_decay)*v,nan=self.epsilon)
        return dict(**self.historical_losses)

    def calculate_loss_slope(self, key: str, current_loss: torch.Tensor):
        """Calculates the slope of the loss using the current loss and its EMA."""
        ema_loss = self.historical_losses.get(key, current_loss)+self.epsilon
        slope = (current_loss-ema_loss)/ema_loss  # relative loss change
        return slope.abs()
    
    def calculate_gradients(self, key: str, current_loss: torch.Tensor, input: torch.Tensor):
        """Calculates the gradient norm of the current loss wrt the model params."""
        self.model.zero_grad()  # Clear previous gradients
        input.requires_grad_(True)
        current_loss.requires_grad_(True)

        # Backward pass to output layer
        output_params = torch.autograd.grad(current_loss, [input], retain_graph=True, allow_unused=True, materialize_grads=True)[0].nan_to_num(self.epsilon)

        # Compute L2 gradient norm
        if output_params.ndim>1: grad_norm = output_params.view(output_params.size(0), -1).norm(2, dim=-1).mean()
        else: grad_norm = output_params.norm(2)

        if grad_norm<=self.epsilon: #use slope as fallback
            grad_norm = self.calculate_loss_slope(key, current_loss)

        return grad_norm

    def pareto_normalizer(self, loss_dict: dict, weight=.8):
        """
        Normalize losses based on the Pareto Principle (80/20 rule).
        
        Parameters:
        loss_dict (dict): Dictionary of loss values with keys as identifiers.
        
        Returns:
        dict: Dictionary of normalized loss values.
        """
        # Extract keys and values from the dictionary
        keys = list(loss_dict.keys())
        losses = np.array(list(loss_dict.values()))
        
        # Calculate the total loss
        total_loss = np.sum(losses)
        
        # Calculate the contribution of each loss
        contributions = losses / total_loss
        
        # Sort contributions in descending order
        sorted_indices = np.argsort(contributions)[::-1]
        sorted_contributions = contributions[sorted_indices]
        
        # Calculate cumulative contributions
        cumulative_contributions = np.cumsum(sorted_contributions)
        
        # Identify the top 20% contributors
        top_20_percent_index = np.argmax(cumulative_contributions >= weight)
        
        # Create a weight array based on the Pareto Principle
        weights = np.ones_like(losses)
        weights[sorted_indices[:top_20_percent_index + 1]] = len(losses) # Give more weight to the top 20%
        
        # Normalize the losses
        normalized_losses = losses * weights
        normalized_losses /= np.sum(normalized_losses) + self.epsilon
        
        # Create a dictionary of normalized losses
        normalized_loss_dict = {keys[i]: normalized_losses[i] for i in range(len(keys))}
        
        return normalized_loss_dict
    
    def redistribute_weights(self, gradients: dict):
        if self.use_pareto: pareto_weights = self.pareto_normalizer(self.historical_losses)
        else: pareto_weights = {}

        inverse_total_gradient = 1./(sum(gradients.values()) + self.epsilon)
        total_initial_weight = sum(self.initial_weights.values())-len(gradients)
        if total_initial_weight<0: return {k: 1. for k in gradients}

        normalized_weights = {}

        # First pass: sum of Pareto contributions (if any) and weight ratios
        for k, g in gradients.items():

            # Compute the weight ratio based on the gradient
            w_ratio = g * inverse_total_gradient

            # average weights
            smoothed_ratio = pareto_weights.get(k, w_ratio)*.5 + w_ratio*.5

            # Store the smoothed ratio for further calculations
            normalized_weights[k] = 1. + total_initial_weight * (smoothed_ratio)

        return normalized_weights

    def on_train_batch_start(self, losses: dict, input: Optional[torch.Tensor]=None):
        """
        Balances the loss terms based on their gradients before each training batch.
        
        Args:
            losses (dict): Dictionary of individual loss terms (scalar tensors).
        
        Returns:
            balanced_loss (torch.Tensor): Weighted sum of losses.
            new_weights (dict): Updated weights for each loss term.
        """
        
        if len(losses)==0: return 0. # no losses to balance
        if not self.initial_weights: self.initial_weights = {k: 1. for k in losses} # initialize weights
        if not self.ema_weights: self.ema_weights = {k: 1. for k in losses} # initialize ema weights
        if not self.active:
            self.update_historical_losses({k: v.item()*self.initial_weights.get(k,1.) for k,v in losses.items() if v>0})
            return sum(v*self.initial_weights.get(k,1.) for k,v in losses.items())

        gradients = {}
        valid_losses = {}

        # Process each loss, skip if the corresponding weight is 0
        for key, loss in losses.items():
            weight = self.initial_weights.get(key, 1.)
            if weight == 0 or loss == 0 or not isinstance(loss,torch.Tensor): continue  # Skip loss with weight 0 or constants
            weighted_loss = loss * weight

            if self.use_norm and input is not None:
                # Compute L2 gradient norm
                grad_norm = self.calculate_gradients(key, weighted_loss, input)
                gradients[key] = max(grad_norm.item(), self.epsilon)
            else:
                # Use loss plateau detection with EMA
                loss_slope = self.calculate_loss_slope(key, weighted_loss)
                gradients[key] = max(loss_slope.item(), self.epsilon)
            valid_losses[key] = loss.nan_to_num(self.epsilon)

        if not valid_losses or not gradients: return torch.tensor(0.0)  # If all losses are skipped

        # update historical losses
        self.update_historical_losses({k: loss.item() for k, loss in valid_losses.items()})

        # Calculate loss weights based on gradient magnitudes
        if len(valid_losses)>1: normalized_weights = self.redistribute_weights(gradients)
        else: normalized_weights = {k: self.initial_weights.get(k,1.) for k in valid_losses.keys()}

        # Update EMA weights
        normalized_weights = self.update_ema_weights(normalized_weights)
        
        # Balance losses
        balanced_loss = 0
        for k, loss in valid_losses.items():
            balanced_loss += normalized_weights.get(k, 1.) * loss
        
        return balanced_loss

    def on_epoch_end(self, weights_decay=None, loss_decay=None):
        """
        Optional method to call at the end of each epoch for logging or further adjustments.
        """
        if weights_decay is not None: self.weights_decay=weights_decay
        if loss_decay is not None: self.loss_decay=loss_decay
        if self.active:
            weights = dict(sorted(self.ema_weights.items(),key=lambda x:x[1],reverse=True))
            losses = dict(sorted(self.historical_losses.items(),key=lambda x:x[1],reverse=True))
            top_weights = {k: round(v, 3) for k, v in list(weights.items())[:3]}
            top_losses = {k: round(v, 3) for k, v in list(losses.items())[:3]}
            print(
                f"|| balancer ema={self.weighted_ema_loss:.3f} "
                f"weights={top_weights} losses={top_losses}"
            )
        gc_collect()

    @property
    def weighted_ema_loss(self):
        return sum(v*self.ema_weights.get(k,1.) for k,v in self.historical_losses.items())

def compute_tsi_loss(original_log_magnitude: torch.Tensor, generated_log_magnitude: torch.Tensor, dim=-1, eps=1e-8):
    """
    Computes the correlation loss between the original and generated log-magnitude spectrograms.
    :param original_envelope: Original log magnitude spectrogram (batch, time)
    :param generated_envelope: Generated log magnitude spectrogram (batch, time)
    :return: Correlation loss.
    """
    original_envelope = compute_envelope(original_log_magnitude, eps=eps, dim=dim)
    generated_envelope = compute_envelope(generated_log_magnitude, eps=eps, dim=dim)

    # Normalize the envelope
    # original_envelope = minmax_scale(original_envelope, eps=eps)
    # generated_envelope = minmax_scale(generated_envelope, eps=eps)
    
    # Compute the correlation
    correlation = compute_correlation(original_envelope, generated_envelope, eps=eps)
    
    # Compute the loss as negative correlation
    loss = 1-correlation
    
    return loss.mean()

def compute_envelope(log_magnitude: torch.Tensor, dim=-1, kernel_size=3, eps=1e-8):
    """
    Compute the envelope of the log-magnitude spectrum of an audio signal.

    Args:
        log_magnitude (torch.Tensor): The log-magnitude spectrum of the audio signal.
        dim (int): The dimension along which to average. Default is -1.
        kernel_size (int): The size of the max pooling kernel. Default is 3.

    Returns:
        torch.Tensor: The computed envelope of the log-magnitude spectrum.
    """

    # Normalize the pooled magnitude
    log_magnitude = F.normalize(log_magnitude, dim=dim, eps=eps)

    # Use max pooling to get the peaks
    max_filtered_tensor = F.max_pool1d(log_magnitude, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    # Replace NaNs with zeros and sum along the second-to-last dimension
    return max_filtered_tensor.nan_to_num(eps).sum(dim)

def compute_tefs(audio_signal: torch.Tensor, eps=1e-8):
    """
    Computes the Hilbert transform of a batched mono audio signal to extract the envelope.
    
    Args:
        audio_signal (torch.Tensor): Batched mono audio signal of shape (batch_size, signal_length).
        
    Returns:
        torch.Tensor: The envelope of the audio signal of shape (batch_size, signal_length).
    """
    # Perform FFT on the input signal
    audio_fft = torch.fft.fft(audio_signal.float(), dim=-1)

    # Create a filter to zero out negative frequencies
    signal_length = audio_signal.shape[-1]
    h = torch.zeros(signal_length, device=audio_signal.device)
    
    # First component is left untouched
    h[0] = 1
    
    if signal_length % 2 == 0:
        # For even signal length
        h[1:signal_length // 2] = 2
        h[signal_length // 2] = 1
    else:
        # For odd signal length
        h[1:(signal_length + 1) // 2] = 2

    # Apply the filter in the frequency domain
    hilbert_fft = audio_fft * h

    # Perform inverse FFT to obtain the analytic signal
    analytic_signal = torch.fft.ifft(hilbert_fft, dim=-1)

    # Extract the envelope by taking the magnitude of the analytic signal
    envelope = torch.abs(analytic_signal)

    # Normalize the envelope
    envelope = minmax_scale(envelope, eps=eps)

    # Compute the instantaneous phase from the analytic signal
    phase = torch.angle(analytic_signal).diff().cos()

    return envelope.nan_to_num(eps), phase.nan_to_num(eps)

def compute_harmonics(mag: torch.Tensor, kernel_sizes=[3, 7, 13, 19, 29], eps=1e-8):
    D = mag.detach().float().cpu().abs().numpy()
    harmonic_list = []
    percussive_list = []

    for kernel_size in kernel_sizes:
        H, P = librosa.decompose.hpss(D,kernel_size=kernel_size)
        harmonic_list.append(torch.from_numpy(H).to(device=mag.device))
        percussive_list.append(torch.from_numpy(P).to(device=mag.device))

    # Concatenate the results
    harmonic = torch.cat(harmonic_list, dim=-1)
    percussive = torch.cat(percussive_list, dim=-1)

    # normalize values
    harmonic = minmax_scale(harmonic, eps=eps)
    percussive = minmax_scale(percussive, eps=eps)

    return harmonic.nan_to_num(eps), percussive.nan_to_num(eps)

def combined_aux_loss(
        original_audio: torch.Tensor, generated_audio: torch.Tensor,
        c_tefs=1., c_hd=1., c_tsi=1., n_mels=128, sample_rate=40000,
        n_fft=1024, hop_length=320, win_length=1024, fmin=0, fmax=None, eps=None):

    kernel_size = n_fft//hop_length+1
    if kernel_size % 2 == 0: kernel_size+=1 #enforce odd kernel size
    if eps is None: eps = torch.finfo(original_audio.dtype).eps

    # Compute STFT once
    if c_hd+c_tsi>0:
        org_mag = mel_spectrogram_torch(
            original_audio,
            n_fft,
            n_mels,
            sample_rate,
            hop_length,
            win_length,
            fmin,
            fmax,
        )
        gen_mag = mel_spectrogram_torch(
            generated_audio,
            n_fft,
            n_mels,
            sample_rate,
            hop_length,
            win_length,
            fmin,
            fmax,
        )
    
    # Harmonic Loss
    if c_hd>0:
        original_harmonics, original_percussives = compute_harmonics(org_mag, eps=eps)
        generated_harmonics, generated_percussives = compute_harmonics(gen_mag, eps=eps)
        # Define loss terms
        harmonic_loss = F.l1_loss(generated_harmonics, original_harmonics)
        harmonic_loss += F.l1_loss(generated_percussives, original_percussives)
    else: harmonic_loss = 0

    # temporal invariant phase
    if c_tsi>0:
        freq_tsi = compute_tsi_loss(org_mag,gen_mag,dim=-1, eps=eps)
        temp_tsi = compute_tsi_loss(org_mag,gen_mag,dim=-2, eps=eps)
        tsi_loss = (freq_tsi+temp_tsi)
    else: tsi_loss = 0

    # Temperol Envelope and Fine Structure Loss
    if c_tefs>0:
        gen_te, gen_tfs = compute_tefs(generated_audio, eps=eps)
        org_te, org_tfs = compute_tefs(original_audio, eps=eps)
        tefs_loss = F.l1_loss(gen_te, org_te) + F.l1_loss(gen_tfs, org_tfs)
    else: tefs_loss = 0
    
    return harmonic_loss, tefs_loss, tsi_loss

def gradient_norm_loss(original_audio: torch.Tensor, generated_audio: torch.Tensor, net_d: torch.nn.Module, eps=1e-8):
    # Compute the gradient penalty
    # Randomly interpolate between real and generated data
    size = [1]*original_audio.ndim
    size[0] = original_audio.size(0)
    alpha = torch.rand(*size, device=original_audio.device)
    interpolated = alpha * original_audio + (1 - alpha) * generated_audio
    interpolated.requires_grad_(True)
    # Get the discriminator output for the interpolated data
    y_d_hat_r, y_d_hat_g, _, _ = net_d(original_audio, interpolated)
    loss_disc, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
    # Compute gradients of discriminator output w.r.t. interpolated data
    net_d.zero_grad()
    
    gradients = torch.autograd.grad(
        outputs=loss_disc,
        inputs=interpolated,
        retain_graph=True,
        allow_unused=True,
        materialize_grads=True
    )[0]
    
    if gradients.ndim<=1: gradients = gradients.unsqueeze(0)
    grad_norm = gradients.view(gradients.size(0), -1).square().sum(-1).sqrt()
    loss = ((grad_norm - 1) ** 2).mean()
    return loss

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/loss.py
# LICENSE: https://github.com/NVIDIA/BigVGAN/blob/main/LICENSE
class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [20, 64, 80, 128, 160, 256],
        loss: Literal["l1","l2","smoothl1"] = "l1",
        epsilon = 1e-8,
        mag_weight = 0.0,
        log_weight = 1.0,
        adjustment_factor = 0.0,  # How much to adjust fmin/fmax dynamically per step
        fmin = 50.,
        fmax = None,
        center = False,
        **kwargs
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple("STFTParams",["window_length", "hop_length"])
        window_lengths = [self.compute_window_length(mel, sampling_rate) for mel in n_mels]
        self.stft_params = [
            STFTParams(window_length=w,hop_length=sampling_rate // 100)
            for w in window_lengths
        ]
        self.n_mels = sorted(n_mels)
        if loss=="l1": self.loss_fn = torch.nn.L1Loss()
        elif loss=="l2": self.loss_fn = torch.nn.MSELoss()
        elif loss=="smoothl1": self.loss_fn = torch.nn.SmoothL1Loss()
        self.loss = loss
        self.epsilon = epsilon
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.center = center
        if fmax is None: fmax = sampling_rate//2
        self.fmax = fmax
        self.fmin = fmin
        self.mel_fmin = [fmin for _ in n_mels]
        self.mel_fmax = [fmax for _ in n_mels]
        self.adjustment_factor = adjustment_factor
        self.frequency_buffer = int(sampling_rate * adjustment_factor)+1
        self.window_lengths = window_lengths

        for k,v in kwargs.items(): self.__setattr__(k,v)

    def to_dict(self):
        return dict(
            sampling_rate=self.sampling_rate,
            n_mels = self.n_mels,
            loss = self.loss,
            epsilon = self.epsilon,
            mag_weight = self.mag_weight,
            log_weight = self.log_weight,
            adjustment_factor = self.adjustment_factor,
            fmin = self.fmin,
            fmax = self.fmax,
            center = self.center,
            mel_fmin=self.mel_fmin,
            mel_fmax=self.mel_fmax,
        )
        
    def show_freqs(self):
        print(f"MultiScaleMelSpectrogramLoss:\n\tn_mels \twindow \tfmin \tfmax")
        for i in range(len(self.mel_fmax)):
            print(f"\t{self.n_mels[i]} \t{self.window_lengths[i]} \t{self.mel_fmin[i]:.0f} \t{self.mel_fmax[i]:.0f}")

    @staticmethod
    def compute_window_length(n_mels: int, sample_rate: int):
        f_min = 0
        f_max = sample_rate / 2
        window_length_seconds = 8 * n_mels / (f_max - f_min)
        window_length = int(window_length_seconds * sample_rate)
        return 2**(window_length.bit_length()-1)
    
    def adjust_fmin_fmax(self, scale_losses: List[float]):
        """
        Adjusts the fmin and fmax values dynamically based on the scale's performance.
        """
        median_loss = np.nanmedian(scale_losses)
        cumloss = np.cumsum(scale_losses)
        cutoff_index = np.argmax(cumloss >= median_loss*len(scale_losses))
        median_low = np.nanmedian(scale_losses[:cutoff_index])
        median_high = np.nanmedian(scale_losses[cutoff_index:])

        for i, scale_loss in enumerate(scale_losses):
            # Calculate deviation from mean
            threshold = median_high if i>=cutoff_index else median_low
            deviation = (scale_loss - threshold)/(threshold + self.epsilon)
            adjustment = min(abs(self.adjustment_factor * deviation),self.adjustment_factor)

            if i>=cutoff_index: # high frequency mels
                self.mel_fmax[i] = min(self.mel_fmax[i] * (1 + adjustment), self.fmax) # increase fmax
                if deviation > self.epsilon:  # Loss is above average, indicating poor performance
                    self.mel_fmin[i] = min(self.mel_fmin[i] * (1 + adjustment), self.mel_fmax[i] - self.frequency_buffer) # increase fmin
                elif deviation < -self.epsilon:  # Loss is below average, indicating good performance
                    self.mel_fmin[i] = max(self.mel_fmin[i] * (1 - adjustment), self.fmin)  # Lower fmin
            else: # low frequency mels
                self.mel_fmin[i] = max(self.mel_fmin[i] * (1 - adjustment), self.fmin)  # Lower fmin
                if deviation > self.epsilon:  # Loss is above average, indicating poor performance
                    self.mel_fmax[i] = min(self.mel_fmax[i] * (1 + adjustment), self.fmax) # increase fmax
                elif deviation < -self.epsilon:  # Loss is below average, indicating good performance
                    self.mel_fmax[i] = max(self.mel_fmax[i] * (1 - adjustment), self.mel_fmin[i] + self.frequency_buffer) # lower fmax

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale mel loss with dynamic weighting."""
        
        scale_losses = []
        for (n_mels, fmin, fmax, s) in zip(self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params):
            x_log_mel = mel_spectrogram_torch(x, s.window_length, n_mels, self.sampling_rate, s.hop_length, s.window_length, fmin, fmax, self.center)
            y_log_mel = mel_spectrogram_torch(y, s.window_length, n_mels, self.sampling_rate, s.hop_length, s.window_length, fmin, fmax, self.center)

            # Compute the per-scale loss
            scale_loss = 0.

            if self.log_weight>0:
                log_loss = self.loss_fn(x_log_mel, y_log_mel)
                scale_loss += self.log_weight * log_loss

            if self.mag_weight>0:
                x_mel = spectral_de_normalize_torch(x_log_mel)
                y_mel = spectral_de_normalize_torch(y_log_mel)
                mag_loss = self.loss_fn(x_mel, y_mel)
                scale_loss += self.mag_weight * mag_loss
            
            # Combine losses for this scale
            scale_losses.append(scale_loss)

        total_loss = sum(scale_losses)/len(scale_losses)

        # Adjust fmin/fmax based on the losses
        if self.adjustment_factor>0: self.adjust_fmin_fmax([loss.item() for loss in scale_losses])

        return total_loss

    
def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss

def discriminator_loss(
        disc_real_outputs: List[torch.Tensor],
        disc_generated_outputs: List[torch.Tensor]
        ):
    loss = 0
    disc_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        L = r_loss + g_loss
        loss += L
        disc_losses.append(L)
    return loss, disc_losses


def generator_loss(disc_outputs: List[torch.Tensor]):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        L = torch.mean((1 - dg) ** 2)
        gen_losses.append(L)
        loss += L
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
