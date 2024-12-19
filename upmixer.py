import logging
import os
import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable, Any, Optional
from functools import wraps
from scipy.signal import butter, sosfilt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# CUDA/CPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

TensorType = torch.Tensor

def check_params(expected_keys: list) -> Callable:
    """Decorator to ensure required params are present."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            params = kwargs.get('params', {})
            if not isinstance(params, dict):
                raise ValueError("Parameters must be a dictionary.")
            missing = [key for key in expected_keys if key not in params]
            if missing:
                raise ValueError(f"Missing parameters: {', '.join(missing)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_tensor(arg_name: str) -> Callable:
    """Decorator to verify that a given argument is a non-None tensor on the correct device."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            tensor = kwargs.get(arg_name)
            if tensor is None:
                raise ValueError(f"{arg_name} cannot be None.")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"'{arg_name}' must be a torch.Tensor.")
            if tensor.device != device:
                raise ValueError(f"Tensor '{arg_name}' must be on device {device}, but found {tensor.device}.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class PsychoacousticUpmixer:
    """
    A versatile and advanced psychoacoustic upmixer class that:
    - Performs phase alignment & polarity correction for stereo stems
    - Applies VBAP-based spatialization with customizable mapping functions
    - Implements psychoacoustic masking, multi-band decorrelation, early reflections, and optional reverb
    - Offers adaptive spatial compression and dynamic range control
    - Supports LFE channel creation with low-pass filtering
    - Handles both mono and stereo inputs with mid/side processing
    - Allows multiband compression with customizable Q-factors
    - Provides presets, parameter validation, and detailed logging
    - Facilitates stem-specific processing chains for maximum flexibility
    """
    
    def __init__(self, sample_rate: int = 44100, hrtf_path: Optional[str] = None, built_in_hrtf: str = 'default', **params):
        """
        Initialize PsychoacousticUpmixer with defaults and allow overrides via params.

        Args:
            sample_rate (int, Optional): Target sample rate (Default=44100 Hz)
            hrtf_path (str, Optional): Path to external HRTF file. If None, use a built-in HRTF.
            built_in_hrtf (str, Optional): Name of a built-in HRTF ('default', 'hrtf1', 'hrtf2').
            params (dict, Optional): All parameters for customization or stem-specific processing.
        
        Raises:
            ValueError: If parameters are invalid or HRTF loading fails.
        """
        self.sample_rate = sample_rate
        if not isinstance(params, dict):
            raise ValueError("Please provide params as a dictionary.")
        
        # Default parameters
        self.params = {
            'n_fft': 4096,
            'hop_length': 512,
            'mask_spread': 0.5,
            'mask_loudness_adjust': 0.1,
            'lfe_cutoff': 40,
            'lfe_weight': 0.5,
            'surround_decorrelation_intensity': 0.5,
            'surround_decorrelation_delay': 20,
            'comp_threshold': -12,
            'comp_ratio': 4,
            'comp_attack_ms': 10.0,
            'comp_release_ms': 100.0,
            'comp_knee_db': 5.0,
            'comp_lookahead_ms': 0.0,
            'comp_makeup_gain': 0,
            'limiter_threshold': -1.0,
            'limiter_ratio': 100.0,
            'lookahead_limiter': False,
            "channel_map": {
                "vocal": [2],      # Center
                "drums": [0, 1],   # Front Left, Front Right
                "bass": [3],       # LFE
                "other": [4, 5]    # Surround Left, Surround Right
            },
            "stem_specific_params": {
                "vocal": {"comp_threshold": -15, "mask_spread": 0.3, "panning_strategy": "fixed_center"},
                "drums": {"comp_threshold": -10, "mask_spread": 0.7, "panning_strategy": "dynamic"},
                "bass": {"comp_threshold": -20, "mask_spread": 0.4, "panning_strategy": "fixed_lfe"},
                "other": {"comp_threshold": -12, "mask_spread": 0.5, "panning_strategy": "spectral"}
            },
            "mix_ratio": 0.7,
            "stereo_width": 1.0,
            "reverb": {
                "enabled": False,
                "impulse_response_path": None,
                "wet_level": 0.3,
                "dry_level": 0.7,
                "room_size": 0.5,
                "decay_time": 1.0,
                "pre_delay": 20.0,
                "damping": 0.5
            },
            "multiband_compression": {
                "enabled": False,
                "bands": 3,
                "thresholds_db": [-20, -15, -10],
                "ratios": [2.0, 3.0, 4.0],
                "attack_ms": [10.0, 10.0, 10.0],
                "release_ms": [100.0, 100.0, 100.0],
                "makeup_gain_db": [0.0, 0.0, 0.0],
                "Q_factors": [1.0, 1.0, 1.0]  # Added Q-factor for each band
            },
            "presets": None,
            "panning": {
                "mapping_function": "linear",  # Options: 'linear', 'logarithmic', 'exponential'
                "spread": 0.2  # Controls the distribution spread across pan positions
            }
        }

        # Update parameters with user-provided overrides
        self.params.update(params)
        self.validate_parameters()

        # Load HRTF
        self.hrtf = self.load_hrtf(hrtf_path, built_in_hrtf)

        # Load reverb IR if enabled
        if self.params['reverb']['enabled']:
            self.load_reverb_ir()

        # Initialize multiband compressors if enabled
        if self.params['multiband_compression']['enabled']:
            self.initialize_multiband_compression()

        # Initialize TorchScript for performance optimization
        self.dynamic_range_compression_script = torch.jit.script(self.dynamic_range_compression)
        self.multi_band_decorrelation_script = torch.jit.script(self.multi_band_decorrelation)
        logger.info("Initialized TorchScript modules for performance optimization.")
    
    def validate_parameters(self):
        """
        Validate that all parameters are within their expected ranges.
        """
        # Example validations
        if not 0 <= self.params['mask_spread'] <= 1:
            raise ValueError("mask_spread must be between 0 and 1.")
        if not 0 <= self.params['mask_loudness_adjust'] <= 1:
            raise ValueError("mask_loudness_adjust must be between 0 and 1.")
        if not 20 <= self.params['lfe_cutoff'] <= 100:
            raise ValueError("lfe_cutoff must be between 20 Hz and 100 Hz.")
        if not 0 <= self.params['lfe_weight'] <= 1:
            raise ValueError("lfe_weight must be between 0 and 1.")
        if not 0 <= self.params['surround_decorrelation_intensity'] <= 1:
            raise ValueError("surround_decorrelation_intensity must be between 0 and 1.")
        if not 10 <= self.params['surround_decorrelation_delay'] <= 50:
            raise ValueError("surround_decorrelation_delay must be between 10 ms and 50 ms.")
        # Validate Q-factors
        if self.params['multiband_compression']['enabled']:
            Q_factors = self.params['multiband_compression'].get('Q_factors', [])
            if len(Q_factors) != self.params['multiband_compression']['bands']:
                raise ValueError("Length of Q_factors must match the number of bands in multiband_compression.")
            for q in Q_factors:
                if q <= 0:
                    raise ValueError("Q_factors must be positive numbers.")
        # Validate panning mapping function
        if self.params['panning'].get('mapping_function', 'linear') not in ['linear', 'logarithmic', 'exponential']:
            raise ValueError("panning.mapping_function must be one of 'linear', 'logarithmic', or 'exponential'.")
        logger.info("All parameters validated successfully.")
    
    def load_hrtf(self, hrtf_path: Optional[str], built_in_hrtf: str) -> TensorType:
        """
        Load an external HRTF impulse response or select a built-in one.

        Args:
            hrtf_path (str, Optional): Path to external HRTF file.
            built_in_hrtf (str): Name of a built-in HRTF to use.

        Returns:
            torch.Tensor: HRTF impulse response tensor.

        Raises:
            ValueError: If HRTF loading fails or built-in HRTF is not recognized.
        """
        if hrtf_path:
            if not os.path.isfile(hrtf_path):
                raise ValueError(f"HRTF file not found at path: {hrtf_path}")
            waveform, sr = torchaudio.load(hrtf_path)
            if sr != self.sample_rate:
                logger.info(f"Resampling HRTF from {sr} Hz to {self.sample_rate} Hz.")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate).to(device)
                waveform = resampler(waveform)
            hrtf_tensor = waveform.to(device).float().mean(dim=0)  # Convert to mono
            logger.info(f"Loaded external HRTF from {hrtf_path}.")
            return hrtf_tensor
        else:
            # Built-in HRTFs (example: multiple options)
            built_in_hrtf_options = {
                'default': torch.tensor([
                    0.0, 0.2, -0.3, 0.4, 0.2,
                    -0.1, 0.0, 0.05, 0.02, 0.0,
                ], device=device, dtype=torch.float32),
                'hrtf1': torch.tensor([
                    0.1, 0.3, -0.2, 0.5, 0.1,
                    -0.2, 0.1, 0.04, 0.03, 0.0,
                ], device=device, dtype=torch.float32),
                'hrtf2': torch.tensor([
                    0.05, 0.25, -0.25, 0.45, 0.15,
                    -0.15, 0.05, 0.06, 0.01, 0.0,
                ], device=device, dtype=torch.float32)
            }
            if built_in_hrtf in built_in_hrtf_options:
                hrtf_tensor = built_in_hrtf_options[built_in_hrtf]
                logger.info(f"Using built-in HRTF: {built_in_hrtf}.")
                return hrtf_tensor
            else:
                raise ValueError(f"Built-in HRTF '{built_in_hrtf}' not recognized. Available options: {list(built_in_hrtf_options.keys())}")
    
    def load_reverb_ir(self):
        """
        Load an impulse response for reverb if specified in parameters.

        Raises:
            ValueError: If reverb is enabled but no valid IR path is provided.
        """
        ir_path = self.params['reverb']['impulse_response_path']
        if ir_path and os.path.isfile(ir_path):
            waveform, sr = torchaudio.load(ir_path)
            if sr != self.sample_rate:
                logger.info(f"Resampling Reverb IR from {sr} Hz to {self.sample_rate} Hz.")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate).to(device)
                waveform = resampler(waveform)
            self.reverb_ir = waveform.to(device).float().mean(dim=0)  # Convert to mono
            logger.info(f"Loaded Reverb IR from {ir_path}.")
        else:
            raise ValueError("Reverb is enabled but no valid impulse response path is provided.")
    
    def initialize_multiband_compression(self):
        """
        Initialize multiband compressors based on parameters.
        """
        bands = self.params['multiband_compression']['bands']
        self.multiband_compressors = []
        for i in range(bands):
            comp_params = {
                'threshold_db': self.params['multiband_compression']['thresholds_db'][i],
                'ratio': self.params['multiband_compression']['ratios'][i],
                'attack_ms': self.params['multiband_compression']['attack_ms'][i],
                'release_ms': self.params['multiband_compression']['release_ms'][i],
                'makeup_gain_db': self.params['multiband_compression']['makeup_gain_db'][i],
                'Q_factor': self.params['multiband_compression']['Q_factors'][i]  # Added Q-factor
            }
            self.multiband_compressors.append(comp_params)
        logger.info(f"Initialized {bands}-band multiband compressors.")
    
    def load_preset(self, preset_name: str):
        """
        Load a predefined parameter preset.

        Args:
            preset_name (str): Name of the preset to load.

        Raises:
            ValueError: If preset name is not recognized.
        """
        presets = {
            "pop_music": {
                "comp_threshold": -14,
                "comp_ratio": 3,
                "stereo_width": 1.2,
                "reverb": {
                    "enabled": True,
                    "impulse_response_path": "presets/ir_pop.wav",
                    "wet_level": 0.3,
                    "dry_level": 0.7,
                    "room_size": 0.6,
                    "decay_time": 1.2,
                    "pre_delay": 25.0,
                    "damping": 0.5
                },
                "multiband_compression": {
                    "enabled": True,
                    "bands": 3,
                    "thresholds_db": [-18, -15, -12],
                    "ratios": [2.5, 3.0, 3.5],
                    "attack_ms": [8.0, 8.0, 8.0],
                    "release_ms": [90.0, 90.0, 90.0],
                    "makeup_gain_db": [1.0, 1.0, 1.0],
                    "Q_factors": [1.0, 1.0, 1.0]
                },
                "stem_specific_params": {
                    "vocal": {"comp_threshold": -16, "mask_spread": 0.35, "panning_strategy": "fixed_center"},
                    "drums": {"comp_threshold": -11, "mask_spread": 0.65, "panning_strategy": "dynamic"},
                    "bass": {"comp_threshold": -21, "mask_spread": 0.35, "panning_strategy": "fixed_lfe"},
                    "other": {"comp_threshold": -13, "mask_spread": 0.45, "panning_strategy": "spectral"}
                }
            },
            "classical_music": {
                "comp_threshold": -20,
                "comp_ratio": 2,
                "stereo_width": 1.0,
                "reverb": {
                    "enabled": True,
                    "impulse_response_path": "presets/ir_classical.wav",
                    "wet_level": 0.4,
                    "dry_level": 0.6,
                    "room_size": 0.8,
                    "decay_time": 2.0,
                    "pre_delay": 30.0,
                    "damping": 0.7
                },
                "multiband_compression": {
                    "enabled": False,
                    "bands": 3,
                    "thresholds_db": [-20, -20, -20],
                    "ratios": [2.0, 2.0, 2.0],
                    "attack_ms": [10.0, 10.0, 10.0],
                    "release_ms": [100.0, 100.0, 100.0],
                    "makeup_gain_db": [0.0, 0.0, 0.0],
                    "Q_factors": [1.0, 1.0, 1.0]
                },
                "stem_specific_params": {
                    "vocal": {"comp_threshold": -19, "mask_spread": 0.4, "panning_strategy": "fixed_center"},
                    "drums": {"comp_threshold": -15, "mask_spread": 0.5, "panning_strategy": "fixed_center"},
                    "bass": {"comp_threshold": -23, "mask_spread": 0.3, "panning_strategy": "fixed_lfe"},
                    "other": {"comp_threshold": -17, "mask_spread": 0.4, "panning_strategy": "spectral"}
                }
            }
            # Add more presets as needed
        }
        
        if preset_name in presets:
            self.params.update(presets[preset_name])
            if self.params['reverb']['enabled']:
                self.load_reverb_ir()
            if self.params['multiband_compression']['enabled']:
                self.initialize_multiband_compression()
            logger.info(f"Loaded preset: {preset_name}")
        else:
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(presets.keys())}")
    
    def db_to_amplitude(self, db: float) -> float:
        """Convert decibel value to amplitude."""
        return 10 ** (db / 20)
    
    def load_audio(self, path: str) -> Tuple[TensorType, int]:
        """
        Load an audio file and return its waveform and sample rate.

        Args:
            path (str): Path to the audio file.

        Returns:
            Tuple[torch.Tensor, int]: (waveform tensor, sample rate)

        Raises:
            RuntimeError: If audio cannot be loaded.
        """
        try:
            waveform, sr = torchaudio.load(path)
            waveform = waveform.to(device).float()
            return waveform, sr
        except RuntimeError as e:
            raise RuntimeError(f"Could not load audio at {path}. Reason: {e}")
    
    def load_stems(self, paths: List[str]) -> Tuple[List[TensorType], int]:
        """
        Load multiple stems from given file paths.

        Args:
            paths (List[str]): List of file paths to stems.

        Returns:
            Tuple[List[torch.Tensor], int]: (list of waveform tensors, sample rate)

        Raises:
            ValueError: If sample rates mismatch and resampling fails.
        """
        stems = []
        sample_rate = None
        for path in paths:
            waveform, sr = self.load_audio(path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                logger.warning(f"Sample rate mismatch: {sr} Hz in {path}, expected {sample_rate} Hz. Resampling.")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate).to(device)
                waveform = resampler(waveform)
            stems.append(waveform)
        return stems, sample_rate
    
    def save_audio(self, file_path: str, audio: TensorType):
        """
        Save the processed audio signal using torchaudio.

        Args:
            file_path (str): The save path.
            audio (torch.Tensor): A multi-channel audio signal [channels, samples].

        Raises:
            OSError: If saving fails.
        """
        try:
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            torchaudio.save(file_path, audio.cpu(), self.sample_rate)
            logger.info(f"Saved audio to {file_path} with sample rate {self.sample_rate} Hz.")
        except OSError as e:
            raise OSError(f"Saving to path failed at {file_path}. Error: {e}")
    
    def check_polarity_and_align_stereo(self, stem: TensorType) -> TensorType:
        """
        Check polarity and align phases for stereo stems using cross-correlation.

        Args:
            stem (torch.Tensor): Stereo audio tensor [2, samples].

        Returns:
            torch.Tensor: Phase and polarity corrected stereo stem [2, samples].
        """
        if stem.dim() != 2 or stem.shape[0] != 2:
            raise ValueError("Only stereo stems can be phase-aligned and polarity-corrected.")

        left = stem[0].cpu().numpy()
        right = stem[1].cpu().numpy()

        # Cross-correlation
        correlation = np.correlate(left, right, mode='full')
        lag = np.argmax(correlation) - len(left) + 1

        # Align signals by lag
        if lag > 0:
            right_aligned = np.concatenate((right[lag:], np.zeros(lag)))
            left_aligned = left
        elif lag < 0:
            left_aligned = np.concatenate((left[-lag:], np.zeros(-lag)))
            right_aligned = right
        else:
            left_aligned = left
            right_aligned = right

        # Polarity check
        dot_product = np.sum(left_aligned * right_aligned)
        if dot_product < 0:
            right_aligned = -right_aligned
            logger.info("Polarity inversion applied to right channel for better coherence.")

        corrected_stem = torch.tensor([left_aligned, right_aligned], device=device, dtype=stem.dtype)
        return corrected_stem
    
    def apply_reverb(self, stem: TensorType) -> TensorType:
        """
        Apply convolutional reverb to the stem using the loaded impulse response.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].

        Returns:
            torch.Tensor: Reverberated audio tensor [channels, samples].
        """
        if not self.params['reverb']['enabled']:
            return stem

        # Convolve each channel with the impulse response
        reverb_ir = self.reverb_ir.unsqueeze(0)  # [1, samples]
        reverb_signal = torch.nn.functional.conv1d(
            stem.unsqueeze(0),
            reverb_ir.unsqueeze(1),  # [out_channels, in_channels, kernel_size]
            padding=self.reverb_ir.shape[0] // 2
        ).squeeze(0)

        # Mix dry and wet signals with additional reverb parameters
        wet_level = self.params['reverb']['wet_level']
        dry_level = self.params['reverb']['dry_level']
        room_size = self.params['reverb']['room_size']
        decay_time = self.params['reverb']['decay_time']
        pre_delay = self.params['reverb']['pre_delay']
        damping = self.params['reverb']['damping']

        # Apply pre-delay
        pre_delay_samples = int((pre_delay / 1000) * self.sample_rate)
        if pre_delay_samples > 0:
            reverb_signal = torch.nn.functional.pad(reverb_signal, (pre_delay_samples, 0))[:stem.shape[1]]

        # Apply damping (simple attenuation for demonstration)
        reverb_signal = reverb_signal * damping

        # Mix dry and wet signals
        mixed = (stem * dry_level) + (reverb_signal * wet_level)
        return mixed
    
    def apply_custom_filter(self, stem: TensorType, custom_kernel: TensorType) -> TensorType:
        """
        Apply a user-provided custom filter to the stem.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            custom_kernel (torch.Tensor): Custom filter kernel tensor.

        Returns:
            torch.Tensor: Filtered audio tensor [channels, samples].
        """
        filtered = torch.zeros_like(stem)
        for ch in range(stem.shape[0]):
            filtered[ch] = self._apply_filter(stem[ch], custom_kernel)
        return filtered
    
    def _apply_filter(self, stem: TensorType, kernel: TensorType) -> TensorType:
        """
        Apply provided filter with simple convolution.

        Args:
            stem (torch.Tensor): Audio signal to apply impulse onto.
            kernel (torch.Tensor): The kernel impulse tensor to convolute with.

        Returns:
            torch.Tensor: Audio signal after convolution.

        Raises:
            ValueError: If filter length is bigger than signal length.
        """
        input_length = stem.shape[-1]
        filter_length = kernel.shape[-1]
        if input_length < filter_length:
            raise ValueError("Filter length bigger than signal length.")
        filtered = torch.nn.functional.conv1d(
            stem.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=(filter_length - 1) // 2
        )
        return filtered.squeeze(0).squeeze(0)
    
    @check_params(expected_keys=[
        'n_fft', 'hop_length', 'mask_spread', 'mask_loudness_adjust',
        'lfe_cutoff', 'lfe_weight', 'surround_decorrelation_intensity',
        'surround_decorrelation_delay', 'comp_threshold', 'comp_ratio',
        'comp_attack_ms', 'comp_release_ms', 'comp_knee_db',
        'comp_lookahead_ms', 'comp_makeup_gain', 'limiter_threshold',
        'limiter_ratio', 'lookahead_limiter', 'channel_map',
        'stem_specific_params', 'mix_ratio', 'stereo_width',
        'reverb', 'multiband_compression', 'panning'
    ])
    def psychoacoustic_upmix_stems(self, stems: List[TensorType], params: Optional[dict] = None) -> TensorType:
        """
        Upmix stems into a multi-channel output using psychoacoustic enhancements.

        Args:
            stems (List[torch.Tensor]): List of waveform tensors (mono or stereo).
            params (dict, Optional): Processing parameters.

        Returns:
            torch.Tensor: Upmixed multi-channel audio tensor [channels, samples].

        Raises:
            ValueError: If number of stems does not match expectation or channel mapping exceeds range.
        """
        if params is None:
            params = self.params

        stem_names = ["vocal", "drums", "bass", "other"]
        if len(stems) != len(stem_names):
            raise ValueError(f"Expected {len(stem_names)} stems, got {len(stems)}.")

        channel_map = params['channel_map']
        stem_specific_params = params.get('stem_specific_params', {})
        output_channels = max([max(channels) for channels in channel_map.values()]) + 1

        # Initialize final mix
        final_mix = torch.zeros((output_channels, 1), device=device)

        # Spatial map for adaptive compression
        spatial_map = {}

        # Process each stem
        for i, stem in enumerate(stems):
            stem_name = stem_names[i]
            sp_params = stem_specific_params.get(stem_name, {})

            # Phase alignment and polarity correction for stereo stems
            if stem.dim() == 2 and stem.shape[0] == 2:
                stem = self.check_polarity_and_align_stereo(stem)

            # Handle stereo inputs directly
            if len(stems) == 1 and stem.dim() == 2:
                # Single stereo stem: apply mid/side processing
                mid, side = self.mid_side_encode(stem)
                # Apply processing to mid and side
                mid_processed = self.process_stem(mid, stem_name, params)
                side_processed = self.process_stem(side, stem_name, params)
                # Re-encode mid/side to stereo
                stereo_processed = self.mid_side_decode(mid_processed, side_processed)
                final_mix = self.mix_into_final_mix(final_mix, stereo_processed)
                continue

            # Convert to mono for centroid calc and VBAP
            if stem.dim() == 2:
                stem_mono = torch.mean(stem, dim=0)
            elif stem.dim() == 1:
                stem_mono = stem
            else:
                raise ValueError("Each stem must be either mono or stereo.")

            stem_np = stem_mono.cpu().numpy()

            # --- Stem-Specific Processing ---
            if stem_name == "vocal":
                # Center channel processing for vocals
                panned_audio = torch.zeros((output_channels, stem_mono.shape[0]), device=device)
                center_channel_index = channel_map["vocal"][0]
                panned_audio[center_channel_index] = stem_mono  # Place mono vocal stem in the center channel

            elif stem_name == "drums":
                # Apply wider stereo width to drums
                stereo_width = sp_params.get('stereo_width', params['stereo_width'])
                panned_audio = self.apply_stereo_width(stem, stereo_width)
                # Perform VBAP panning
                spectral_centroid = self.calculate_spectral_centroid(stem_np)
                pan_position = self.map_centroid_to_pan_2d(spectral_centroid, mapping_function=params['panning']['mapping_function'])
                pan_position = self.smooth_pan_transition(torch.tensor([0.0, 0.0], device=device), pan_position, smoothing_factor=0.1, spread=params['panning']['spread'])
                panned_audio = self.apply_vbap(panned_audio, pan_position, output_channels)

            elif stem_name == "bass":
                # Apply low-pass filter for bass stem and LFE channel creation
                lfe_cutoff = sp_params.get('lfe_cutoff', params['lfe_cutoff'])
                lfe_channel_index = channel_map["bass"][0]
                panned_audio = torch.zeros((output_channels, stem_mono.shape[0]), device=device)
                panned_audio[lfe_channel_index] = torchaudio.functional.lowpass_biquad(stem_mono, self.sample_rate, cutoff_freq=lfe_cutoff)

            elif stem_name == "other":
                # Apply spectral panning to "other" stem
                spectral_centroid = self.calculate_spectral_centroid(stem_np)
                pan_position = self.map_centroid_to_pan_2d(spectral_centroid, mapping_function=params['panning']['mapping_function'])
                pan_position = self.smooth_pan_transition(torch.tensor([0.0, 0.0], device=device), pan_position, smoothing_factor=0.1, spread=params['panning']['spread'])
                panned_audio = self.apply_vbap(stem_mono, pan_position, output_channels)

                # Apply reverb to "other" stem
                if params['reverb']['enabled']:
                    panned_audio = self.apply_reverb(panned_audio)

            else:
                panned_audio = stem

            # Apply multi-band decorrelation and early reflections
            for ch in channel_map.get(stem_name, []):
                panned_audio[ch] = self.multi_band_decorrelation_script(panned_audio[ch],
                                                                         intensity=params['surround_decorrelation_intensity'],
                                                                         delay_ms=params['surround_decorrelation_delay'])
                panned_audio[ch] = self.add_early_reflections(panned_audio[ch])

            # Create and apply masking threshold with stem-specific mask_spread
            mask_spread = sp_params.get('mask_spread', params['mask_spread'])
            mask_loudness_adjust = sp_params.get('mask_loudness_adjust', params['mask_loudness_adjust'])
            mask = self.create_masking_threshold_with_spread(panned_audio, mask_spread=mask_spread, mask_loudness_adjust=mask_loudness_adjust)
            panned_audio = self.reduce_masked_energy(panned_audio, mask)

            # Loudness normalization
            panned_audio = self.normalize_stem_loudness(panned_audio, target_lufs=-14.0)

            # Apply multiband compression if enabled
            panned_audio = self.apply_multiband_compression(panned_audio)

            # Dynamic range compression with stem-specific parameters
            panned_audio = self.dynamic_range_compression_script(
                stem=panned_audio,
                threshold_db=sp_params.get('comp_threshold', params['comp_threshold']),
                ratio=sp_params.get('comp_ratio', params['comp_ratio']),
                attack_ms=sp_params.get('comp_attack_ms', params['comp_attack_ms']),
                release_ms=sp_params.get('comp_release_ms', params['comp_release_ms']),
                lookahead_ms=sp_params.get('comp_lookahead_ms', params['comp_lookahead_ms']),
                makeup_gain_db=sp_params.get('comp_makeup_gain', params['comp_makeup_gain'])
            )

            panned_audio = self.reduce_clipping(panned_audio, target_peak=0.99)

            # Analyze for visualization
            self.analyze_spatial_spectral(panned_audio, stem_name=stem_name)

            # Add processed stem to the final mix
            final_mix = self.mix_into_final_mix(final_mix, panned_audio)

            # Update spatial map
            spatial_map[i] = self.map_centroid_to_pan_2d(self.calculate_spectral_centroid(stem_np), mapping_function=params['panning']['mapping_function'])

        # Adaptive Compression based on spatial overlap
        final_mix = self.adaptive_spatial_compression_single(final_mix, spatial_map, threshold=0.2)

        # Final dynamic range compression
        final_mix = self.dynamic_range_compression_script(
            stem=final_mix,
            threshold_db=params['comp_threshold'],
            ratio=params['comp_ratio'],
            attack_ms=params['comp_attack_ms'],
            release_ms=params['comp_release_ms'],
            lookahead_ms=params['comp_lookahead_ms'],
            makeup_gain_db=params['comp_makeup_gain']
        )

        # Final limiting
        if params['lookahead_limiter']:
            final_mix = self.lookahead_limit(final_mix, params['limiter_threshold'])
        else:
            final_mix = self.soft_limit(final_mix, params['limiter_threshold'], params['limiter_ratio'])

        return final_mix
    
    def mid_side_encode(self, stereo_stem: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Encode stereo audio to mid/side.

        Args:
            stereo_stem (torch.Tensor): Stereo audio tensor [2, samples].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mid, side) tensors.
        """
        mid = (stereo_stem[0] + stereo_stem[1]) / 2
        side = (stereo_stem[0] - stereo_stem[1]) / 2
        return mid, side
    
    def mid_side_decode(self, mid: TensorType, side: TensorType) -> TensorType:
        """
        Decode mid/side audio to stereo.

        Args:
            mid (torch.Tensor): Mid channel tensor.
            side (torch.Tensor): Side channel tensor.

        Returns:
            torch.Tensor: Stereo audio tensor [2, samples].
        """
        left = mid + side
        right = mid - side
        return torch.stack([left, right], dim=0)
    
    def calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """
        Calculate the spectral centroid of an audio signal.

        Args:
            audio (np.ndarray): Audio waveform.

        Returns:
            float: Spectral centroid in Hz.
        """
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        return float(np.mean(centroid))
    
    def map_centroid_to_pan_2d(self, centroid: float, mapping_function: str = 'linear') -> TensorType:
        """
        Map spectral centroid to a 2D pan position using specified mapping function.

        Args:
            centroid (float): Spectral centroid in Hz.
            mapping_function (str): 'linear', 'logarithmic', or 'exponential'.

        Returns:
            torch.Tensor: 2D pan position vector.
        """
        low, high = 200, 4000  # Hz
        centroid_clamped = np.clip(centroid, low, high)
        normalized = (centroid_clamped - low) / (high - low)  # 0 to 1

        if mapping_function == 'linear':
            x = 2 * normalized - 1  # -1 to 1
        elif mapping_function == 'logarithmic':
            x = 2 * (np.log10(1 + 9 * normalized) / np.log10(10)) - 1
        elif mapping_function == 'exponential':
            x = 2 * (1 - np.exp(-normalized * 5)) - 1
        else:
            x = 2 * normalized - 1  # Default to linear

        y = 0.0  # Flat panning on Y-axis
        return torch.tensor([x, y], device=device).float()
    
    def smooth_pan_transition(self, old_pan: TensorType, new_pan: TensorType, smoothing_factor: float = 0.1, spread: float = 0.2) -> TensorType:
        """
        Smoothly transition between old and new pan positions with spread.

        Args:
            old_pan (torch.Tensor): Current pan position.
            new_pan (torch.Tensor): Target pan position.
            smoothing_factor (float): Smoothing factor between 0 and 1.
            spread (float): Spread factor for distributing across pan positions.

        Returns:
            torch.Tensor: New pan position after smoothing and spread.
        """
        smoothed_pan = old_pan * (1 - smoothing_factor) + new_pan * smoothing_factor
        # Apply spread by adding a small variation
        variation = torch.randn_like(smoothed_pan) * spread
        spread_pan = smoothed_pan + variation
        spread_pan = torch.clamp(spread_pan, -1.0, 1.0)
        return spread_pan
    
    def apply_vbap(self, source: TensorType, pan_position: TensorType, num_channels: int) -> TensorType:
        """
        Apply VBAP spatialization based on pan_position vector.

        Args:
            source (torch.Tensor): Mono audio tensor [samples].
            pan_position (torch.Tensor): 2D pan position vector.
            num_channels (int): Number of output channels.

        Returns:
            torch.Tensor: Multi-channel audio tensor [channels, samples].

        Raises:
            ValueError: If number of channels exceeds predefined speaker layout.
        """
        norm = torch.norm(pan_position)
        if norm == 0:
            pan_position = torch.tensor([0.0, 0.0], device=device).float()
        else:
            pan_position = (pan_position / norm).float()

        # Define a 6-speaker layout for more surround options
        speaker_positions = torch.tensor([
            [-1, 0],    # Front Left
            [1, 0],     # Front Right
            [-0.5, -0.866],  # Surround Left
            [0.5, -0.866],   # Surround Right
            [0, 1],     # Top
            [0, -1]     # Bottom
        ], device=device).float()

        if num_channels > speaker_positions.shape[0]:
            # Expand speaker layout if needed
            additional_speakers = torch.randn((num_channels - speaker_positions.shape[0], 2), device=device)
            speaker_positions = torch.cat((speaker_positions, additional_speakers), dim=0)

        if num_channels != speaker_positions.shape[0]:
            raise ValueError(f"Number of channels ({num_channels}) does not match speaker positions ({speaker_positions.shape[0]})")

        gains = torch.matmul(pan_position, speaker_positions.T)
        gains = torch.clamp(gains, 0, 1)
        sum_gains = torch.sum(gains)
        if sum_gains > 0:
            gains = gains / sum_gains

        # output: (channels, samples)
        return source.unsqueeze(0) * gains.unsqueeze(1)
    
    def calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """
        Calculate the spectral centroid of an audio signal.

        Args:
            audio (np.ndarray): Audio waveform.

        Returns:
            float: Spectral centroid in Hz.
        """
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        return float(np.mean(centroid))
    
    def mid_side_encode(self, stereo_stem: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Encode stereo audio to mid/side.

        Args:
            stereo_stem (torch.Tensor): Stereo audio tensor [2, samples].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mid, side) tensors.
        """
        mid = (stereo_stem[0] + stereo_stem[1]) / 2
        side = (stereo_stem[0] - stereo_stem[1]) / 2
        return mid, side
    
    def mid_side_decode(self, mid: TensorType, side: TensorType) -> TensorType:
        """
        Decode mid/side audio to stereo.

        Args:
            mid (torch.Tensor): Mid channel tensor.
            side (torch.Tensor): Side channel tensor.

        Returns:
            torch.Tensor: Stereo audio tensor [2, samples].
        """
        left = mid + side
        right = mid - side
        return torch.stack([left, right], dim=0)
    
    @check_tensor("stem")
    def multi_band_decorrelation(self, stem: TensorType, intensity: float = 0.5, delay_ms: int = 20) -> TensorType:
        """
        Applies multi-band decorrelation by mixing the dry signal with a delayed + Noise-Modulated version.

        Args:
            stem (torch.Tensor): Audio Tensor [samples].
            intensity (float): Intensity of the added noise between 0 and 1 for the sides.
            delay_ms (int): Delay for the second source of the same signal, in milliseconds.

        Returns:
            torch.Tensor: The resulting spatially processed signal with delay.
        """
        delay_samples = int((delay_ms / 1000) * self.sample_rate)
        if delay_samples >= stem.shape[-1]:
            logger.warning("Delay time exceeds stem signal length. Skipping decorrelation.")
            return stem
        pad = torch.zeros((delay_samples,), device=device)
        delayed = torch.cat((pad, stem), dim=-1)

        random_noise = (torch.rand_like(delayed) * 2 - 1) * intensity

        output = stem + random_noise[:stem.shape[-1]] + delayed[:stem.shape[-1]]
        return output
    
    @check_tensor("stem")
    def add_early_reflections(self, stem: TensorType) -> TensorType:
        """
        Adds subtle early reflections using a predefined short HRTF filter.

        Args:
            stem (torch.Tensor): Input audio tensor [samples].

        Returns:
            torch.Tensor: The result of mixing early reflections (convolved signal) and input signal.
        """
        early_reflection_filtered = self._apply_filter(stem, self.hrtf)
        mixed = stem + early_reflection_filtered * 0.2  # Reduced gain due to early reflections
        return mixed
    
    @check_tensor("stem")
    def create_masking_threshold_with_spread(self, stem: TensorType, mask_spread: float, mask_loudness_adjust: float) -> TensorType:
        """
        Create a masking threshold with specific mask spread and loudness adjustment.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            mask_spread (float): Frequency range of the masking mask as a ratio (0 - 1).
            mask_loudness_adjust (float): Level reduction of masked bands (0 - 1).

        Returns:
            torch.Tensor: Masking threshold tensor [channels, samples].
        """
        mask = torch.ones_like(stem, device=device).float()
        cutoff_freq = int(mask_spread * (self.params['n_fft'] // 2))  # Example: mask spread
        mask[:, :cutoff_freq] = 1 - mask_loudness_adjust
        mag_out = stem * mask
        return mag_out
    
    @check_tensor("stem")
    @check_tensor("mask")
    def reduce_masked_energy(self, stem: TensorType, mask: TensorType) -> TensorType:
        """
        Reduce energy in masked regions.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            mask (torch.Tensor): Masking threshold tensor [channels, samples].

        Returns:
            torch.Tensor: Audio tensor with reduced masked energy.
        """
        threshold = torch.mean(mask)
        reduction_factor = 0.5  # Reduce by 50%
        reduction = (mask > threshold).float() * reduction_factor
        output = stem * (1 - reduction)
        return output
    
    @check_tensor("stem")
    def dynamic_range_compression(self, stem: TensorType, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, lookahead_ms: float, makeup_gain_db: float) -> TensorType:
        """
        Apply dynamic range compression to a stem.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            threshold_db (float): Compression threshold in dB.
            ratio (float): Compression ratio.
            attack_ms (float): Attack time in milliseconds.
            release_ms (float): Release time in milliseconds.
            lookahead_ms (float): Lookahead time in milliseconds.
            makeup_gain_db (float): Makeup gain in dB.

        Returns:
            torch.Tensor: Compressed audio tensor.
        """
        attack = attack_ms / 1000.0
        release = release_ms / 1000.0

        audio = stem.clone()
        envelope = torch.abs(audio)
        alpha_attack = torch.exp(-1.0 / (attack * self.sample_rate))
        alpha_release = torch.exp(-1.0 / (release * self.sample_rate))
        gain = torch.ones_like(audio).float()

        for n in range(1, audio.shape[1]):
            for ch in range(audio.shape[0]):
                if envelope[ch, n] > envelope[ch, n-1]:
                    envelope[ch, n] = alpha_attack * envelope[ch, n-1] + (1 - alpha_attack) * envelope[ch, n]
                else:
                    envelope[ch, n] = alpha_release * envelope[ch, n-1] + (1 - alpha_release) * envelope[ch, n]

                if 20 * torch.log10(envelope[ch, n] + 1e-6) > threshold_db:
                    desired_gain_db = threshold_db - 20 * torch.log10(envelope[ch, n] + 1e-6)
                    gain[ch, n] = self.db_to_amplitude(desired_gain_db) * ratio
                else:
                    gain[ch, n] = 1.0

        audio = audio * gain
        makeup_gain = self.db_to_amplitude(makeup_gain_db)
        audio = audio * makeup_gain
        return audio

    @check_tensor("stem")
    def lookahead_limit(self, stem: TensorType, threshold_db: float) -> TensorType:
        """
        Apply a lookahead limiter to the audio.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            threshold_db (float): Limiter threshold in dB.

        Returns:
            torch.Tensor: Limited audio tensor.
        """
        limiter = self.dynamic_range_compression(
            stem=stem,
            threshold_db=threshold_db,
            ratio=100.0,
            attack_ms=1.0,
            release_ms=10.0,
            lookahead_ms=0.0,
            makeup_gain_db=0.0
        )
        return limiter

    @check_tensor("stem")
    def soft_limit(self, stem: TensorType, threshold_db: float, ratio: float) -> TensorType:
        """
        Apply a soft limiter to the audio.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            threshold_db (float): Limiter threshold in dB.
            ratio (float): Limiting ratio.

        Returns:
            torch.Tensor: Limited audio tensor.
        """
        limiter = self.dynamic_range_compression(
            stem=stem,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=10.0,
            release_ms=100.0,
            lookahead_ms=0.0,
            makeup_gain_db=0.0
        )
        return limiter

    def mix_into_final_mix(self, final_mix: TensorType, panned_audio: TensorType) -> TensorType:
        """
        Mix a processed stem into the final mix.

        Args:
            final_mix (torch.Tensor): Current final mix [channels, samples].
            panned_audio (torch.Tensor): Processed stem audio [channels, samples].

        Returns:
            torch.Tensor: Updated final mix [channels, samples].
        """
        min_length = min(final_mix.shape[1], panned_audio.shape[1])
        if final_mix.shape[1] < panned_audio.shape[1]:
            pad_length = panned_audio.shape[1] - final_mix.shape[1]
            final_mix = torch.nn.functional.pad(final_mix, (0, pad_length))
        elif final_mix.shape[1] > panned_audio.shape[1]:
            pad_length = final_mix.shape[1] - panned_audio.shape[1]
            panned_audio = torch.nn.functional.pad(panned_audio, (0, pad_length))
        
        final_mix += panned_audio
        return final_mix

    def adaptive_spatial_compression_single(self, final_mix: TensorType, spatial_map: Dict[int, TensorType], threshold: float = 0.2) -> TensorType:
        """
        Apply adaptive compression to the final mix based on spatial overlap.

        Args:
            final_mix (torch.Tensor): Final mix [channels, samples].
            spatial_map (Dict[int, torch.Tensor]): Spatial positions of stems.
            threshold (float, Optional): Distance threshold for compression.

        Returns:
            torch.Tensor: Compressed final mix [channels, samples].
        """
        stems = list(spatial_map.keys())
        for i in range(len(stems)):
            for j in range(i+1, len(stems)):
                pan_i = spatial_map[stems[i]]
                pan_j = spatial_map[stems[j]]
                dist = torch.dist(pan_i, pan_j).item()
                if dist < threshold:
                    logger.info(f"Adaptive compression: Stems {stems[i]} and {stems[j]} are within threshold distance ({dist:.2f}). Applying compression.")
                    final_mix = self.dynamic_range_compression(
                        stem=final_mix,
                        threshold_db=-3.0,
                        ratio=2.0,
                        attack_ms=10.0,
                        release_ms=100.0,
                        lookahead_ms=0.0,
                        makeup_gain_db=0.0
                    )
        return final_mix

    def normalize_stem_loudness(self, stem: TensorType, target_lufs: float = -14.0) -> TensorType:
        """
        Normalize a given stem to a target loudness in LUFS.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            target_lufs (float, Optional): Target loudness in LUFS.

        Returns:
            torch.Tensor: Loudness-normalized audio tensor [channels, samples].
        """
        audio_np = stem.cpu().numpy()
        meter = pyln.Meter(self.sample_rate)
        if stem.dim() == 2 and stem.shape[0] > 2:
            stereo_mix = np.mean(audio_np, axis=0)
        elif stem.dim() == 2:
            stereo_mix = audio_np.mean(axis=0)
        else:
            stereo_mix = audio_np

        loudness = meter.integrated_loudness(stereo_mix.T)
        gain = target_lufs - loudness
        stem = stem * (10**(gain/20))
        return stem
    
    def apply_stereo_width(self, audio: TensorType, stereo_width: float = 1.0) -> TensorType:
        """
        Adjust the stereo width of the audio.

        Args:
            audio (torch.Tensor): Stereo audio tensor [channels, samples].
            stereo_width (float): Stereo width factor (0 - 2).

        Returns:
            torch.Tensor: Stereo audio tensor with adjusted width [channels, samples].
        """
        if audio.dim() == 1:
            return audio.unsqueeze(0)
        if audio.shape[0] < 2:
            return audio
        left = audio[0]
        right = audio[1]
        center = (left + right) / 2
        side = (left - right) / 2 * stereo_width

        left_new = center + side
        right_new = center - side

        adjusted_audio = torch.stack([left_new, right_new], dim=0)
        return adjusted_audio
    
    def apply_multiband_compression(self, stem: TensorType) -> TensorType:
        """
        Apply multiband compression to the stem.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].

        Returns:
            torch.Tensor: Compressed audio tensor [channels, samples].
        """
        if not self.params['multiband_compression']['enabled']:
            return stem

        bands = self.params['multiband_compression']['bands']
        compressed_stem = torch.zeros_like(stem)
        
        # Create band filters using butter and sosfilt
        for i in range(bands):
            low = i * (self.sample_rate / 2) / bands
            high = (i + 1) * (self.sample_rate / 2) / bands
            Q = self.params['multiband_compression']['Q_factors'][i]
            if low == 0:
                # Lowpass filter for the first band
                sos_band = butter(N=2, Wn=high/(self.sample_rate/2), btype='low', output='sos')
            elif high >= self.sample_rate/2:
                # Highpass filter for the last band
                sos_band = butter(N=2, Wn=low/(self.sample_rate/2), btype='high', output='sos')
            else:
                # Bandpass filter
                sos_band = butter(N=2, Wn=[low/(self.sample_rate/2), high/(self.sample_rate/2)], btype='band', output='sos')

            # Apply filter using sosfilt
            filtered = torch.tensor(sosfilt(sos_band, stem.cpu().numpy()), device=device).float()
            # Apply compression per band
            comp_params = self.multiband_compressors[i]
            band_compressed = self.dynamic_range_compression(
                stem=filtered,
                threshold_db=comp_params['threshold_db'],
                ratio=comp_params['ratio'],
                attack_ms=comp_params['attack_ms'],
                release_ms=comp_params['release_ms'],
                lookahead_ms=0.0,
                makeup_gain_db=comp_params['makeup_gain_db']
            )
            compressed_stem += band_compressed
        return compressed_stem

    def process_stem(self, stem: TensorType, stem_name: str, params: dict) -> TensorType:
        """
        Apply a processing chain to a stem based on stem-specific parameters.

        Args:
            stem (torch.Tensor): Audio tensor [samples].
            stem_name (str): Name of the stem.
            params (dict): Processing parameters.

        Returns:
            torch.Tensor: Processed audio tensor [samples].
        """
        sp_params = params.get('stem_specific_params', {}).get(stem_name, {})
        # Example processing steps
        # Users can extend this method to include entirely different processing chains
        processed = stem

        # Example: Apply masking
        mask_spread = sp_params.get('mask_spread', params['mask_spread'])
        mask_loudness_adjust = sp_params.get('mask_loudness_adjust', params['mask_loudness_adjust'])
        mask = self.create_masking_threshold_with_spread(processed.unsqueeze(0), mask_spread, mask_loudness_adjust)
        mask = mask.squeeze(0)
        processed = self.reduce_masked_energy(processed.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

        # Loudness normalization
        processed = self.normalize_stem_loudness(processed.unsqueeze(0), target_lufs=-14.0).squeeze(0)

        return processed

    @check_params(expected_keys=[
        'n_fft', 'hop_length', 'mask_spread', 'mask_loudness_adjust',
        'lfe_cutoff', 'lfe_weight', 'surround_decorrelation_intensity',
        'surround_decorrelation_delay', 'comp_threshold', 'comp_ratio',
        'comp_attack_ms', 'comp_release_ms', 'comp_knee_db',
        'comp_lookahead_ms', 'comp_makeup_gain', 'limiter_threshold',
        'limiter_ratio', 'lookahead_limiter', 'channel_map',
        'stem_specific_params', 'mix_ratio', 'stereo_width',
        'reverb', 'multiband_compression', 'panning'
    ])
    def psychoacoustic_upmix_stems(self, stems: List[TensorType], params: Optional[dict] = None) -> TensorType:
        """
        Upmix stems into a multi-channel output using psychoacoustic enhancements.

        Args:
            stems (List[torch.Tensor]): List of waveform tensors (mono or stereo).
            params (dict, Optional): Processing parameters.

        Returns:
            torch.Tensor: Upmixed multi-channel audio tensor [channels, samples].

        Raises:
            ValueError: If number of stems does not match expectation or channel mapping exceeds range.
        """
        if params is None:
            params = self.params

        stem_names = ["vocal", "drums", "bass", "other"]
        if len(stems) != len(stem_names):
            raise ValueError(f"Expected {len(stem_names)} stems, got {len(stems)}.")

        channel_map = params['channel_map']
        stem_specific_params = params.get('stem_specific_params', {})
        output_channels = max([max(channels) for channels in channel_map.values()]) + 1

        # Initialize final mix
        final_mix = torch.zeros((output_channels, 1), device=device)

        # Spatial map for adaptive compression
        spatial_map = {}

        # Process each stem
        for i, stem in enumerate(stems):
            stem_name = stem_names[i]
            sp_params = stem_specific_params.get(stem_name, {})

            # Phase alignment and polarity correction for stereo stems
            if stem.dim() == 2 and stem.shape[0] == 2:
                stem = self.check_polarity_and_align_stereo(stem)

            # Handle stereo inputs directly
            if len(stems) == 1 and stem.dim() == 2:
                # Single stereo stem: apply mid/side processing
                mid, side = self.mid_side_encode(stem)
                # Apply processing to mid and side
                mid_processed = self.process_stem(mid, stem_name, params)
                side_processed = self.process_stem(side, stem_name, params)
                # Re-encode mid/side to stereo
                stereo_processed = self.mid_side_decode(mid_processed, side_processed)
                final_mix = self.mix_into_final_mix(final_mix, stereo_processed)
                continue

            # Convert to mono for centroid calc and VBAP
            if stem.dim() == 2:
                stem_mono = torch.mean(stem, dim=0)
            elif stem.dim() == 1:
                stem_mono = stem
            else:
                raise ValueError("Each stem must be either mono or stereo.")

            stem_np = stem_mono.cpu().numpy()

            # --- Stem-Specific Processing ---
            if stem_name == "vocal":
                # Center channel processing for vocals
                panned_audio = torch.zeros((output_channels, stem_mono.shape[0]), device=device)
                center_channel_index = channel_map["vocal"][0]
                panned_audio[center_channel_index] = stem_mono  # Place mono vocal stem in the center channel

            elif stem_name == "drums":
                # Apply wider stereo width to drums
                stereo_width = sp_params.get('stereo_width', params['stereo_width'])
                panned_audio = self.apply_stereo_width(stem, stereo_width)
                # Perform VBAP panning
                spectral_centroid = self.calculate_spectral_centroid(stem_np)
                pan_position = self.map_centroid_to_pan_2d(spectral_centroid, mapping_function=params['panning']['mapping_function'])
                pan_position = self.smooth_pan_transition(torch.tensor([0.0, 0.0], device=device), pan_position, smoothing_factor=0.1, spread=params['panning']['spread'])
                panned_audio = self.apply_vbap(panned_audio, pan_position, output_channels)

            elif stem_name == "bass":
                # Apply low-pass filter for bass stem and LFE channel creation
                lfe_cutoff = sp_params.get('lfe_cutoff', params['lfe_cutoff'])
                lfe_channel_index = channel_map["bass"][0]
                panned_audio = torch.zeros((output_channels, stem_mono.shape[0]), device=device)
                panned_audio[lfe_channel_index] = torchaudio.functional.lowpass_biquad(stem_mono, self.sample_rate, cutoff_freq=lfe_cutoff)

            elif stem_name == "other":
                # Apply spectral panning to "other" stem
                spectral_centroid = self.calculate_spectral_centroid(stem_np)
                pan_position = self.map_centroid_to_pan_2d(spectral_centroid, mapping_function=params['panning']['mapping_function'])
                pan_position = self.smooth_pan_transition(torch.tensor([0.0, 0.0], device=device), pan_position, smoothing_factor=0.1, spread=params['panning']['spread'])
                panned_audio = self.apply_vbap(stem_mono, pan_position, output_channels)

                # Apply reverb to "other" stem
                if params['reverb']['enabled']:
                    panned_audio = self.apply_reverb(panned_audio)

            else:
                panned_audio = stem

            # Apply multi-band decorrelation and early reflections
            for ch in channel_map.get(stem_name, []):
                panned_audio[ch] = self.multi_band_decorrelation_script(panned_audio[ch],
                                                                         intensity=params['surround_decorrelation_intensity'],
                                                                         delay_ms=params['surround_decorrelation_delay'])
                panned_audio[ch] = self.add_early_reflections(panned_audio[ch])

            # Create and apply masking threshold with stem-specific mask_spread
            mask_spread = sp_params.get('mask_spread', params['mask_spread'])
            mask_loudness_adjust = sp_params.get('mask_loudness_adjust', params['mask_loudness_adjust'])
            mask = self.create_masking_threshold_with_spread(panned_audio, mask_spread=mask_spread, mask_loudness_adjust=mask_loudness_adjust)
            panned_audio = self.reduce_masked_energy(panned_audio, mask)

            # Loudness normalization
            panned_audio = self.normalize_stem_loudness(panned_audio, target_lufs=-14.0)

            # Apply multiband compression if enabled
            panned_audio = self.apply_multiband_compression(panned_audio)

            # Dynamic range compression with stem-specific parameters
            panned_audio = self.dynamic_range_compression_script(
                stem=panned_audio,
                threshold_db=sp_params.get('comp_threshold', params['comp_threshold']),
                ratio=sp_params.get('comp_ratio', params['comp_ratio']),
                attack_ms=sp_params.get('comp_attack_ms', params['comp_attack_ms']),
                release_ms=sp_params.get('comp_release_ms', params['comp_release_ms']),
                lookahead_ms=sp_params.get('comp_lookahead_ms', params['comp_lookahead_ms']),
                makeup_gain_db=sp_params.get('comp_makeup_gain', params['comp_makeup_gain'])
            )

            panned_audio = self.reduce_clipping(panned_audio, target_peak=0.99)

            # Analyze for visualization
            self.analyze_spatial_spectral(panned_audio, stem_name=stem_name)

            # Add processed stem to the final mix
            final_mix = self.mix_into_final_mix(final_mix, panned_audio)

            # Update spatial map
            spatial_map[i] = self.map_centroid_to_pan_2d(self.calculate_spectral_centroid(stem_np), mapping_function=params['panning']['mapping_function'])

        # Adaptive Compression based on spatial overlap
        final_mix = self.adaptive_spatial_compression_single(final_mix, spatial_map, threshold=0.2)

        # Final dynamic range compression
        final_mix = self.dynamic_range_compression_script(
            stem=final_mix,
            threshold_db=params['comp_threshold'],
            ratio=params['comp_ratio'],
            attack_ms=params['comp_attack_ms'],
            release_ms=params['comp_release_ms'],
            lookahead_ms=params['comp_lookahead_ms'],
            makeup_gain_db=params['comp_makeup_gain']
        )

        # Final limiting
        if params['lookahead_limiter']:
            final_mix = self.lookahead_limit(final_mix, params['limiter_threshold'])
        else:
            final_mix = self.soft_limit(final_mix, params['limiter_threshold'], params['limiter_ratio'])

        return final_mix

    def analyze_spatial_spectral(self, stem: TensorType, stem_name: str = "Stem"):
        """
        Analyze spatial and spectral properties of a stem.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            stem_name (str, Optional): Name of the stem for labeling.
        """
        if stem.dim() == 2 and stem.shape[0] >= 2:
            # Convert to numpy
            stem_np = stem.cpu().numpy()
            # Spectrogram
            S = librosa.stft(stem_np.mean(axis=0).astype(np.float32), n_fft=self.params['n_fft'], hop_length=self.params['hop_length'])
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', hop_length=self.params['hop_length'])
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram of {stem_name}')
            plt.show()

            # Cross-correlation
            left = stem_np[0]
            right = stem_np[1]
            correlation = np.correlate(left, right, mode='full')
            lag = np.argmax(correlation) - len(left) + 1
            logger.info(f'Cross-correlation lag for {stem_name}: {lag} samples ({lag / self.sample_rate * 1000:.2f} ms)')

    @check_tensor("stem")
    def reduce_clipping(self, stem: TensorType, target_peak: float = 0.99) -> TensorType:
        """
        Reduce clipping by normalizing audio to a target peak.

        Args:
            stem (torch.Tensor): Audio tensor [channels, samples].
            target_peak (float, Optional): Target peak amplitude.

        Returns:
            torch.Tensor: Normalized audio tensor [channels, samples].
        """
        peak = torch.max(torch.abs(stem))
        if peak > target_peak:
            stem = stem * (target_peak / peak)
        return stem
