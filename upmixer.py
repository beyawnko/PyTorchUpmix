#!/usr/bin/env python3
"""
psychoacoustic_upmixer_advanced_expanded.py

An advanced, single-file Python script for a Psychoacoustic Upmixer that leverages:
  - Pydantic-based configuration & validation
  - Vectorized minimal gating-based denoising
  - Vectorized STFT-based operations (default n_fft=2048, Hann window)
  - Vectorized multi-band & single-band compression, lookahead limiting
  - Expanded placeholders for advanced psychoacoustic features:
       * Harmonic Reconstruction (with approximate F0-based partial boosting)
       * Transient Shaping (onset detection & gain envelope)
       * Spectral EQ (basic tilt or parametric-like approach)
       * Advanced VBAP (Delaunay triangulation example, easily extended for 3D/HRTF-based panning)
       * Iterative Phase Approach (Griffin-Lim style stub)
  - Basic reverb IR convolution
  - Optional multi-band decorrelation
  - Plugin architecture for extensibility

Usage Example:
  python psychoacoustic_upmixer_advanced_expanded.py --stems vocal.wav drums.wav bass.wav other.wav \
                                                     --output upmixed_output.wav \
                                                     [--config_file myconfig.yaml/json] \
                                                     [--params key=val ...] \
                                                     [--plot_analysis]

Note:
  - This script remains oriented for offline usage. Real-time operation would require
    block-based processing and more sophisticated state management.
  - The advanced psychoacoustic placeholders (harmonic reconstruction, etc.)
    contain conceptual logic that can be refined or replaced with more
    robust algorithms (e.g., advanced F0 estimators, fancy onset detectors, etc.).
"""

import logging
import os
import sys
import json
import argparse
import glob
import math
from typing import List, Tuple, Dict, Optional, Literal

import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
import librosa
import yaml

from pydantic import BaseModel, validator, ValidationError, Field
from scipy.signal import butter, sosfiltfilt, correlate
from scipy.spatial import Delaunay

try:
    import nussl  # For advanced separation/masking if needed
    HAVE_NUSSL = True
except ImportError:
    HAVE_NUSSL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

###############################################################################
# Custom Exceptions
###############################################################################
class UpmixerError(Exception):
    """Base class for psychoacoustic upmixer exceptions."""
    pass

class ParameterError(UpmixerError):
    """Exception raised for configuration or parameter issues."""
    pass

class FileLoadingError(UpmixerError):
    """Exception raised for file loading failures."""
    pass

class PanningError(UpmixerError):
    """Exception raised for VBAP or panning-related errors."""
    pass

###############################################################################
# Plugin Architecture
###############################################################################
class BasePlugin:
    """
    BasePlugin for the upmixer. Must implement 'process(audio) -> audio'.
    """
    def __init__(self, params: dict = None):
        self.params = params or {}

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Plugin must implement 'process' method.")

class GainPlugin(BasePlugin):
    """
    Example plugin that applies a constant gain to the audio signal.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.gain = self.params.get('gain', 1.0)

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        return audio * self.gain

###############################################################################
# Pydantic Configuration Models
###############################################################################
class ReverbConfig(BaseModel):
    enabled: bool = False
    impulse_response_path: Optional[str] = None
    wet_level: float = Field(0.3, ge=0, le=1)
    dry_level: float = Field(0.7, ge=0, le=1)

class MultibandCompressionConfig(BaseModel):
    enabled: bool = False
    bands: int = Field(3, ge=1)
    filter_order: int = Field(4, ge=1)
    thresholds_db: List[float] = Field(default_factory=lambda: [-20, -15, -10])
    ratios: List[float] = Field(default_factory=lambda: [2.0, 3.0, 4.0])
    attack_ms: List[float] = Field(default_factory=lambda: [10.0, 15.0, 20.0])
    release_ms: List[float] = Field(default_factory=lambda: [100.0, 120.0, 150.0])

    @validator('thresholds_db')
    def check_thresholds(cls, v):
        if not all(v[i] <= v[i+1] for i in range(len(v)-1)):
            raise ValueError("thresholds_db must be non-decreasing.")
        return v

class PanningConfig(BaseModel):
    mapping_function: Literal['linear','logarithmic','exponential'] = 'linear'
    spread: float = Field(0.2, ge=0, le=1)

class SpeakerLayoutConfig(BaseModel):
    layout: Dict[int, List[float]]

    @validator('layout')
    def validate_speaker_positions(cls, layout):
        for ch, pos in layout.items():
            if len(pos)!=2:
                raise ValueError(f"Channel {ch} must have [x,y] coords.")
        return layout

class Config(BaseModel):
    n_fft: int = 2048
    hop_length: int = 512
    mask_spread: float = Field(0.5, ge=0, le=1)
    mask_loudness_adjust: float = Field(0.1, ge=0, le=1)
    comp_threshold: float = -12.0
    comp_ratio: float = Field(4.0, gt=0)
    comp_attack_ms: float = Field(10.0, gt=0)
    comp_release_ms: float = Field(100.0, gt=0)
    comp_knee_db: float = Field(5.0, ge=0)
    comp_makeup_gain_db: float = 0.0
    limiter_threshold: float = Field(-1.0, lt=0)
    limiter_ratio: float = Field(100.0, gt=0)
    channel_map: Dict[str, List[int]] = Field(
        default_factory=lambda: {
            "vocal":[2],
            "drums":[0,1],
            "bass":[3],
            "other":[4,5]
        }
    )
    stem_specific_params: Dict[str,Dict] = Field(default_factory=dict)
    built_in_hrtf_options: Dict[str, torch.Tensor] = Field(default_factory=dict)
    panning: PanningConfig = Field(default_factory=PanningConfig)
    speaker_layout_config: SpeakerLayoutConfig = Field(
        default_factory=lambda: SpeakerLayoutConfig(layout={
            0:[-1.0,0.0],1:[1.0,0.0],2:[0.0,0.0],
            3:[0.0,-1.0],4:[-1.0,1.0],5:[1.0,1.0]
        })
    )
    reverb: ReverbConfig = Field(default_factory=ReverbConfig)
    multiband_compression: MultibandCompressionConfig = Field(default_factory=MultibandCompressionConfig)
    stereo_width: float = Field(1.0, ge=0, le=2)
    centroid_low: float = Field(200.0, ge=20)
    centroid_high: float = Field(5000.0, ge=200)
    do_denoising: bool = True
    do_harmonic_reconstruction: bool = False
    do_transient_shaping: bool = False
    do_spectral_eq: bool = False
    do_phase_iterative: bool = False
    plugins: Dict[str,Dict] = Field(default_factory=dict)
    target_lufs: float = Field(-23.0)

    @validator('centroid_high')
    def check_centroid_range(cls, v, values):
        low = values.get('centroid_low', None)
        if low is not None and v<=low:
            raise ValueError("centroid_high must be > centroid_low.")
        return v

###############################################################################
# Main Psychoacoustic Upmixer
###############################################################################
class PsychoacousticUpmixer:
    """
    Main psychoacoustic upmixer class. Features advanced placeholders:
        harmonic reconstruction, transient shaping, spectral eq, advanced VBAP,
        iterative phase approach, reverb IR convolution, multi-band compression, etc.
    """
    def __init__(
        self,
        sample_rate: int = 44100,
        hrtf_path: Optional[str] = None,
        built_in_hrtf: str = 'default',
        target_loudness: float = -23.0,
        config_file: Optional[str] = None,
        **params
    ):
        self.sample_rate = sample_rate
        self.target_loudness = target_loudness

        if config_file and os.path.isfile(config_file):
            try:
                with open(config_file,'r') as f:
                    if config_file.endswith(('.yaml','.yml')):
                        conf_dict = yaml.safe_load(f)
                    else:
                        conf_dict = json.load(f)
                self.config = Config(**conf_dict)
            except (ValidationError, ValueError) as e:
                raise ParameterError(f"Config load error: {e}")
        else:
            self.config = Config()

        if params:
            base_dict = self.config.dict()
            for k,v in params.items():
                base_dict[k] = v
            self.config = Config.parse_obj(base_dict)

        self.validate_channel_map()
        self.hrtf = self.load_hrtf(hrtf_path, built_in_hrtf)
        self.reverb_ir = self.load_reverb_ir()
        self.initialize_multiband_compression()
        self.meter = pyln.Meter(self.sample_rate)
        self.plugins = []
        self.load_plugins()

        self._hann_window = torch.hann_window(self.config.n_fft, device=device)
        logger.info("PsychoacousticUpmixer initialized.")

    def validate_channel_map(self):
        sp_layout = self.config.speaker_layout_config.layout
        for stem_name, chans in self.config.channel_map.items():
            for ch in chans:
                if ch not in sp_layout:
                    logger.warning(f"Channel {ch} for stem '{stem_name}' not in speaker layout.")
        logger.info("Channel map validated.")

    ############################################################################
    # HRTF & Reverb
    ############################################################################
    def load_hrtf(self, hrtf_path: Optional[str], built_in_hrtf: str) -> torch.Tensor:
        if hrtf_path and os.path.isfile(hrtf_path):
            try:
                wf, sr = torchaudio.load(hrtf_path)
                if sr!=self.sample_rate:
                    logger.info(f"Resampling HRTF from {sr} -> {self.sample_rate}")
                    wf = torchaudio.transforms.Resample(sr,self.sample_rate)(wf)
                if wf.size(0)==1:
                    wf = wf.repeat(2,1)
                logger.info(f"Loaded external HRTF from {hrtf_path}")
                return wf.to(device)
            except Exception as e:
                logger.error(f"Could not load external HRTF {hrtf_path}: {e}")
        return self.load_built_in_hrtf(built_in_hrtf)

    def load_built_in_hrtf(self, name: str) -> torch.Tensor:
        bdict = self.config.built_in_hrtf_options
        if name in bdict:
            logger.info(f"Using built-in HRTF: {name}")
            return bdict[name].to(device)
        else:
            logger.warning(f"No built-in HRTF for '{name}'. Using fallback.")
            fallback = torch.tensor([
                [0.0, 0.1, 0.0, -0.1, 0.1],
                [0.1, 0.0, 0.1,  0.05, 0.0]
            ], device=device)
            return fallback

    def load_reverb_ir(self) -> Optional[torch.Tensor]:
        if not self.config.reverb.enabled:
            return None
        path = self.config.reverb.impulse_response_path
        if not path or not os.path.isfile(path):
            logger.warning("No valid reverb IR path. Disabling reverb.")
            return None
        try:
            wf, sr = torchaudio.load(path)
            if sr!=self.sample_rate:
                logger.info(f"Resampling reverb IR from {sr} -> {self.sample_rate}")
                wf = torchaudio.transforms.Resample(sr,self.sample_rate)(wf)
            return wf.to(device)
        except Exception as e:
            logger.error(f"Reverb IR load error: {e}")
            return None

    def initialize_multiband_compression(self):
        self.multiband_compressors = []
        mbc = self.config.multiband_compression
        if not mbc.enabled:
            return
        n_bands = mbc.bands
        for i in range(n_bands):
            th = mbc.thresholds_db[i] if i<len(mbc.thresholds_db) else -12.0
            ratio_ = mbc.ratios[i] if i<len(mbc.ratios) else 2.0
            atk = mbc.attack_ms[i] if i<len(mbc.attack_ms) else 10.0
            rls = mbc.release_ms[i] if i<len(mbc.release_ms) else 100.0
            self.multiband_compressors.append({
                'threshold_db':th,
                'ratio':ratio_,
                'attack_ms':atk,
                'release_ms':rls
            })

    ############################################################################
    # Audio I/O
    ############################################################################
    def load_audio(self, path: str) -> torch.Tensor:
        if not os.path.isfile(path):
            raise FileLoadingError(f"Audio file '{path}' not found.")
        wf, sr = torchaudio.load(path)
        if sr!=self.sample_rate:
            logger.info(f"Resample from {sr} -> {self.sample_rate}")
            wf = torchaudio.transforms.Resample(sr,self.sample_rate)(wf)
        return wf.to(device)

    def load_stems(self, paths: List[str]) -> Tuple[List[torch.Tensor], int]:
        stms = []
        for p in paths:
            stms.append(self.load_audio(p))
        return stms, self.sample_rate

    def save_audio(self, file_path: str, audio: torch.Tensor):
        odir = os.path.dirname(file_path)
        if odir and not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)
        audio = self.reduce_clipping(audio)
        if audio.dim()==1:
            audio = audio.unsqueeze(0)
        torchaudio.save(file_path, audio.cpu(), self.sample_rate)
        logger.info(f"Saved upmixed audio to '{file_path}'.")

    ############################################################################
    # Minimal Gating-based Denoising (Vectorized)
    ############################################################################
    def _denoise_gating(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Minimal gating approach using RMS threshold. If RMS < thr, zero it out.
        """
        thr_rms = 0.02
        if audio.dim()==1:
            rms_val = torch.sqrt(torch.mean(audio**2))
            return torch.where(rms_val < thr_rms, torch.zeros_like(audio), audio)
        else:
            rms_ch = torch.sqrt(torch.mean(audio**2, dim=-1))
            mask = (rms_ch>=thr_rms).unsqueeze(-1)
            return audio*mask

    ############################################################################
    # Phase Alignment (Stereo)
    ############################################################################
    def phase_align_stems(self, stems: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Basic cross-correlation based stereo alignment.
        """
        aligned = []
        for st in stems:
            if st.dim()==2 and st.size(0)==2:
                left, right = st[0], st[1]
                corr = correlate(left.cpu().numpy(), right.cpu().numpy(), mode='full')
                lag = np.argmax(corr) - (len(left)-1)
                right_shifted = torch.roll(right, shifts=-lag)
                if right_shifted.abs().max()>1e-8:
                    if right_shifted.min() < -0.5*right_shifted.abs().max():
                        right_shifted = -right_shifted
                aligned.append(torch.stack([left,right_shifted], dim=0))
            else:
                aligned.append(st)
        return aligned

    ############################################################################
    # Main Upmixing Pipeline
    ############################################################################
    def psychoacoustic_upmix_stems(self, stems: List[torch.Tensor]) -> torch.Tensor:
        aligned = self.phase_align_stems(stems)
        max_len = max(st.size(-1) for st in aligned)
        sp_layout = self.config.speaker_layout_config.layout
        out_ch = max(sp_layout.keys())+1
        final_mix = torch.zeros((out_ch, max_len), device=device)

        stem_names = list(self.config.channel_map.keys())
        for i, st_audio in enumerate(aligned):
            st_name = stem_names[i] if i<len(stem_names) else f"stem_{i}"
            processed = self._process_stem(st_audio, st_name, self.config.channel_map.get(st_name, []))
            final_mix = self.mix_into_final_mix(processed, final_mix, max_len)

        # optional adaptive spatial compression
        final_mix = self.adaptive_spatial_compression(final_mix)
        # final single-band DRC
        final_mix = self.dynamic_range_compression_vectorized(final_mix)
        # final lookahead limiting
        final_mix = self.lookahead_limit_vectorized(final_mix)
        # final clip reduction
        final_mix = self.reduce_clipping(final_mix)
        return final_mix

    def _process_stem(self, stem: torch.Tensor, stem_name: str, channels: List[int]) -> torch.Tensor:
        """
        Applies the entire processing chain for a single stem.
        """
        # minimal gating-based denoising
        if self.config.do_denoising:
            stem = self._denoise_gating(stem)

        # advanced psychoacoustic expansions
        if self.config.do_harmonic_reconstruction:
            stem = self._harmonic_reconstruction(stem)

        if self.config.do_transient_shaping:
            stem = self._transient_shaping(stem)

        # stereo width
        if stem.dim()==2 and stem.size(0)==2:
            stem = self.apply_stereo_width(stem, self.config.stereo_width)

        # centroid => panning
        mono_mix = stem.mean(dim=0) if stem.dim()==2 else stem
        centroid_np = mono_mix.cpu().numpy()
        c_feat = librosa.feature.spectral_centroid(
            y=centroid_np,
            sr=self.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        centroid_val = float(c_feat.mean()) if c_feat.size>0 else 0.0
        pan_pos = self.map_centroid_to_pan_2d(centroid_val)

        # advanced VBAP
        st_panned = self.vbap_spatialization(stem, channels, pan_pos)

        # psychoacoustic masking
        st_masked = self.apply_psychoacoustic_masking(st_panned)

        # loudness normalization
        st_loud = self.normalize_stem_loudness(st_masked)

        # multi-band compression
        if self.config.multiband_compression.enabled:
            st_loud = self.apply_multiband_compression_vectorized(st_loud)

        # reverb
        if self.config.reverb.enabled and self.reverb_ir is not None:
            st_loud = self.add_early_reflections(st_loud)

        # multi-band decorrelation
        st_decor = self.multi_band_decorrelation(st_loud)

        # spectral EQ placeholder
        if self.config.do_spectral_eq:
            st_decor = self._spectral_shaping(st_decor, stem_name)

        # plugins
        for plg in self.plugins:
            st_decor = plg.process(st_decor)

        # optional iterative phase approach
        if self.config.do_phase_iterative:
            st_decor = self._phase_iterative(st_decor)

        return self.reduce_clipping(st_decor)

    ###########################################################################
    # Advanced Psychoacoustic Features (Expanded Placeholders)
    ###########################################################################
    def _harmonic_reconstruction(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for advanced harmonic reconstruction.
        Here we do a simple F0-based partial boosting in the STFT domain.
        """
        logger.debug("Starting harmonic reconstruction (expanded placeholder).")
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        stft_result = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                                 window=self._hann_window.to(device), return_complex=True)
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)

        # naive F0 estimation: pick the freq bin with highest mean magnitude
        avg_mag = magnitude.mean(dim=-1)  # (channels, freq_bins)
        f0_bin = torch.argmax(avg_mag, dim=-1)
        freq_bins = torch.linspace(0, self.sample_rate/2, steps=magnitude.size(1), device=device)

        # For each channel, boost the 2nd-4th harmonics of the identified F0 bin
        for ch in range(audio.size(0)):
            bin_f0 = f0_bin[ch].item()
            if bin_f0<1:
                continue
            f0_freq = freq_bins[int(bin_f0)].item()
            if f0_freq>0:
                for harm_num in range(2,5):
                    harm_freq = f0_freq*harm_num
                    # find nearest bin
                    bin_idx = (torch.abs(freq_bins - harm_freq)).argmin()
                    if bin_idx.item()<magnitude.size(1):
                        magnitude[ch, bin_idx, :]*=1.2

        new_stft = magnitude*torch.exp(1j*phase)
        out_audio = torch.istft(new_stft, n_fft=n_fft, hop_length=hop_length,
                                window=self._hann_window.to(device))
        logger.debug("Harmonic reconstruction completed.")
        return out_audio

    def _transient_shaping(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for advanced transient shaping. 
        Uses onset detection and a small gain boost around onsets.
        """
        logger.debug("Starting transient shaping (expanded placeholder).")
        if audio.dim()==1:
            audio = audio.unsqueeze(0)

        out_audio = audio.clone()
        for ch in range(audio.size(0)):
            sig_np = audio[ch].cpu().numpy()
            onset_frames = librosa.onset.onset_detect(
                y=sig_np, sr=self.sample_rate, hop_length=self.config.hop_length
            )
            onset_samples = librosa.frames_to_samples(onset_frames, hop_length=self.config.hop_length)

            # Example: short attack boost
            attack_samps = int(0.01*self.sample_rate)  # 10 ms
            for onset_sample in onset_samples:
                start = max(0, onset_sample)
                end = min(audio.size(-1), onset_sample+attack_samps)
                out_audio[ch, start:end]*=1.2

        logger.debug("Transient shaping completed.")
        return out_audio

    def _spectral_shaping(self, audio: torch.Tensor, stem_name: str) -> torch.Tensor:
        """
        Placeholder for advanced spectral EQ. 
        Example: simple tilt or partial param-EQ in the STFT domain.
        """
        logger.debug(f"Starting spectral shaping for '{stem_name}' (expanded placeholder).")
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        if audio.dim()==1:
            audio = audio.unsqueeze(0)

        out_audio = torch.zeros_like(audio)
        freq_lin = torch.linspace(0, self.sample_rate/2, steps=n_fft//2+1, device=device)

        for ch in range(audio.size(0)):
            stft_c = torch.stft(audio[ch], n_fft=n_fft, hop_length=hop_length,
                                window=self._hann_window.to(device), return_complex=True)
            mag = torch.abs(stft_c)
            phs = torch.angle(stft_c)

            # Simple tilt from 0.8 at low freq to 1.2 at high freq
            tilt_curve = torch.linspace(0.8, 1.2, steps=mag.size(0), device=device).unsqueeze(-1)
            new_mag = mag*tilt_curve
            new_stft = new_mag*torch.exp(1j*phs)
            out_audio[ch] = torch.istft(new_stft, n_fft=n_fft, hop_length=hop_length,
                                        window=self._hann_window.to(device))
        logger.debug(f"Spectral shaping for '{stem_name}' completed.")
        return out_audio

    def _phase_iterative(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for iterative phase approach (similar to Griffin-Lim).
        """
        logger.debug("Starting iterative phase approach (expanded placeholder).")
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        n_iter = 5

        if audio.dim()==1:
            audio = audio.unsqueeze(0)

        out_audio = torch.zeros_like(audio)
        for ch in range(audio.size(0)):
            stft_c = torch.stft(audio[ch], n_fft=n_fft, hop_length=hop_length,
                                window=self._hann_window.to(device), return_complex=True)
            mag = torch.abs(stft_c)
            phase_est = torch.exp(1j*torch.rand_like(stft_c))  # random initial phase

            for _ in range(n_iter):
                recon = torch.istft(mag*torch.exp(1j*torch.angle(phase_est)),
                                    n_fft=n_fft, hop_length=hop_length,
                                    window=self._hann_window.to(device))
                recon_stft = torch.stft(recon, n_fft=n_fft, hop_length=hop_length,
                                        window=self._hann_window.to(device), return_complex=True)
                # enforce original magnitude
                phase_est = recon_stft

            out_audio[ch] = torch.istft(mag*torch.exp(1j*torch.angle(phase_est)),
                                        n_fft=n_fft, hop_length=hop_length,
                                        window=self._hann_window.to(device))
        logger.debug("Iterative phase approach completed.")
        return out_audio

    ###########################################################################
    # Psychoacoustic Masking & Basic Gains
    ###########################################################################
    def apply_psychoacoustic_masking(self, audio: torch.Tensor) -> torch.Tensor:
        mask_db = -10.0*self.config.mask_loudness_adjust
        mask_amp = 10.0**(mask_db/20.0)

        def freq_gate(sig: torch.Tensor):
            fft_c = torch.fft.rfft(sig)
            mag = fft_c.abs()
            thr = mag.mean()*mask_amp
            return torch.fft.irfft(fft_c*(mag>thr), n=sig.size(-1))

        if audio.dim()==1:
            return freq_gate(audio)
        outs = []
        for c in range(audio.size(0)):
            outs.append(freq_gate(audio[c]))
        return torch.stack(outs, dim=0)

    ###########################################################################
    # Stereo Width
    ###########################################################################
    def apply_stereo_width(self, audio: torch.Tensor, width: float) -> torch.Tensor:
        mid = 0.5*(audio[0]+audio[1])
        side= 0.5*(audio[0]-audio[1])*width
        left = mid+side
        right= mid-side
        return torch.stack([left,right],dim=0)

    ###########################################################################
    # Mapping Spectral Centroid => 2D Pan Position
    ###########################################################################
    def map_centroid_to_pan_2d(self, centroid: float) -> torch.Tensor:
        """
        Maps spectral centroid -> 2D pan position using config's mapping_function & spread.
        """
        if self.config.centroid_high<=self.config.centroid_low:
            logger.warning("Invalid centroid range, returning center position.")
            return torch.tensor([0.0,0.0], device=device)

        norm_val = (centroid - self.config.centroid_low)/(self.config.centroid_high-self.config.centroid_low)
        norm_val = torch.clamp(torch.tensor(norm_val, device=device), 0.0, 1.0)
        if self.config.panning.mapping_function=='logarithmic':
            norm_val = torch.log1p(norm_val*(math.e-1))
        elif self.config.panning.mapping_function=='exponential':
            norm_val = norm_val**2

        x = (norm_val*2.0 -1.0)*self.config.panning.spread
        y = 0.0
        return torch.tensor([x,y], device=device)

    ###########################################################################
    # Advanced VBAP
    ###########################################################################
    def vbap_spatialization(self, stem: torch.Tensor, channels: List[int], pan_position: torch.Tensor) -> torch.Tensor:
        """
        Example advanced VBAP using Delaunay triangulation for speaker layout.
        Falls back if outside hull.
        """
        if not channels:
            return stem
        sp_layout = self.config.speaker_layout_config.layout
        out_ch = max(sp_layout.keys())+1
        result = torch.zeros((out_ch, stem.size(-1)), device=device)
        if stem.dim()==1:
            stem = stem.unsqueeze(0)

        speaker_positions = []
        speaker_indices = []
        for c in channels:
            if c in sp_layout:
                speaker_positions.append(sp_layout[c])
                speaker_indices.append(c)

        if len(speaker_positions)<3:
            logger.warning("Not enough speakers for advanced VBAP. Falling back.")
            mean_stem = stem.mean(dim=0) if stem.dim()==2 else stem
            for c in speaker_indices:
                result[c,:mean_stem.size(-1)]+=mean_stem
            return result

        coords = np.array(speaker_positions)
        try:
            tri = Delaunay(coords)
            pt = np.array([pan_position[0].item(), pan_position[1].item()])
            simplex_idx = tri.find_simplex(pt)
            if simplex_idx==-1:
                logger.warning("Pan position outside speaker hull, fallback to amplitude panning.")
                mean_stem = stem.mean(dim=0) if stem.dim()==2 else stem
                for c in speaker_indices:
                    result[c,:mean_stem.size(-1)]+=mean_stem
                return result
            else:
                triangle = tri.simplices[simplex_idx]
                A = np.concatenate((coords[triangle], np.ones((3,1))), axis=1)
                b = np.array([pt[0], pt[1],1.0])
                x_sol, _, _, _= np.linalg.lstsq(A, b, rcond=None)
                x_sol[x_sol<0]=0
                if x_sol.sum()>0:
                    x_sol/=x_sol.sum()

                if stem.dim()==2 and stem.size(0)==2:
                    for i, sp_idx in enumerate(triangle):
                        cchan = speaker_indices[sp_idx]
                        result[cchan,:stem.size(-1)] += stem[0]*x_sol[i]+stem[1]*x_sol[i]
                else:
                    for i, sp_idx in enumerate(triangle):
                        cchan = speaker_indices[sp_idx]
                        result[cchan,:stem.size(-1)] += stem[0]*x_sol[i]
        except Exception as e:
            logger.error(f"VBAP triangulation error: {e}")
            mean_stem = stem.mean(dim=0) if stem.dim()==2 else stem
            for c in speaker_indices:
                result[c,:mean_stem.size(-1)] += mean_stem
        return result

    ###########################################################################
    # Multi-band Decorrelation & Reverb
    ###########################################################################
    def multi_band_decorrelation(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Simple multi-band approach with fixed freq edges and small time delay 
        to decorrelate. CPU-based with sosfiltfilt for demonstration.
        """
        if audio.dim()==1:
            return audio
        freq_edges = [0,2000,6000,20000]
        out_sig = torch.zeros_like(audio)
        for i in range(3):
            low, high = freq_edges[i], freq_edges[i+1]
            try:
                sos = butter(4, [low,high], btype='band', fs=self.sample_rate, output='sos')
                band_np = sosfiltfilt(sos, audio.cpu().numpy(), axis=1)
                band_t = torch.from_numpy(band_np).to(device)
                delay_samps = 20
                band_delayed = torch.roll(band_t, shifts=delay_samps, dims=-1)
                mix_ratio = 0.5
                out_sig += band_t*(1-mix_ratio)+band_delayed*mix_ratio
            except Exception as e:
                logger.error(f"Decorrelation band {i} error: {e}")
                out_sig += audio
        return out_sig

    def add_early_reflections(self, audio: torch.Tensor) -> torch.Tensor:
        if self.reverb_ir is None:
            return audio
        wet, dry = self.config.reverb.wet_level, self.config.reverb.dry_level
        if audio.dim()==1:
            audio = audio.unsqueeze(0)
        if self.reverb_ir.dim()==2 and self.reverb_ir.size(0)==2 and audio.size(0)==2:
            left_ir = self.reverb_ir[0].unsqueeze(0).unsqueeze(0)
            right_ir= self.reverb_ir[1].unsqueeze(0).unsqueeze(0)
            left_out = torch.nn.functional.conv1d(
                audio[0].unsqueeze(0).unsqueeze(0),
                left_ir, padding=self.reverb_ir.size(1)//2
            ).squeeze()
            right_out= torch.nn.functional.conv1d(
                audio[1].unsqueeze(0).unsqueeze(0),
                right_ir, padding=self.reverb_ir.size(1)//2
            ).squeeze()
            return torch.stack([
                dry*audio[0]+wet*left_out,
                dry*audio[1]+wet*right_out
            ], dim=0)
        else:
            ir_mono = self.reverb_ir.mean(dim=0, keepdim=True).unsqueeze(0)
            outs = []
            for c in range(audio.size(0)):
                conv_out = torch.nn.functional.conv1d(
                    audio[c].unsqueeze(0).unsqueeze(0),
                    ir_mono, padding=ir_mono.size(-1)//2
                ).squeeze()
                outs.append(dry*audio[c]+wet*conv_out)
            return torch.stack(outs, dim=0)

    ###########################################################################
    # Multi-band Compression (Vectorized Stub)
    ###########################################################################
    def apply_multiband_compression_vectorized(self, audio: torch.Tensor) -> torch.Tensor:
        mb_cfg = self.config.multiband_compression
        if not mb_cfg.enabled:
            return audio
        logger.debug("Starting multi-band compression (expanded stub).")
        freq_edges = np.logspace(math.log10(20), math.log10(self.sample_rate/2), mb_cfg.bands+1)
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        window = self._hann_window.to(device)

        if audio.dim()==1:
            audio = audio.unsqueeze(0)

        stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
        mag = torch.abs(stft)
        phs = torch.angle(stft)
        freq_bin_vals = torch.linspace(0, self.sample_rate/2, steps=mag.size(1), device=device)

        for bidx, comp_data in enumerate(self.multiband_compressors):
            if bidx>=len(freq_edges)-1:
                continue
            low_hz, high_hz = freq_edges[bidx], freq_edges[bidx+1]
            thr_db = comp_data['threshold_db']
            ratio_ = comp_data['ratio']

            band_mask = (freq_bin_vals>=low_hz) & (freq_bin_vals<high_hz)
            band_mag = mag[:, band_mask, :]

            env_db = 20.0*torch.log10(band_mag+1e-8)
            diff_db= torch.clamp(env_db - thr_db, min=0.0)
            gain_db= -(diff_db*(1.0-1.0/ratio_))
            gain_lin= 10.0**(gain_db/20.0)
            band_mag*=gain_lin

        new_stft = mag*torch.exp(1j*phs)
        out_audio = torch.istft(new_stft, n_fft=n_fft, hop_length=hop_length, window=window)
        logger.debug("Multi-band compression completed.")
        return out_audio

    ###########################################################################
    # Single-Band DRC & Lookahead Limiter (Vectorized)
    ###########################################################################
    def dynamic_range_compression_vectorized(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim()==1:
            audio = audio.unsqueeze(0)
        atk_ms = self.config.comp_attack_ms
        rls_ms = self.config.comp_release_ms
        thr_db = self.config.comp_threshold
        ratio_ = self.config.comp_ratio
        knee_db = self.config.comp_knee_db
        makeup_db = self.config.comp_makeup_gain_db

        window_size = int(0.05*self.sample_rate)
        window_size = max(window_size,1)
        window = torch.ones(window_size, device=device)/window_size

        sq_data = audio**2
        pad_sq = torch.nn.functional.pad(sq_data, (window_size//2, window_size//2))
        conv_out = torch.nn.functional.conv1d(
            pad_sq.unsqueeze(0), window.unsqueeze(0).unsqueeze(0), padding=0
        ).squeeze(0)[...,:audio.size(-1)]
        rms = torch.sqrt(torch.clamp(conv_out, min=1e-10))
        env_db = 20.0*torch.log10(rms+1e-8)
        diff_db = torch.clamp(env_db-thr_db, min=0.0)
        knee_mask = (diff_db<knee_db)
        gain_db = torch.zeros_like(diff_db)
        gain_db[knee_mask] = - ((diff_db[knee_mask]**2)/(2.0*knee_db))
        gain_db[~knee_mask] = - diff_db[~knee_mask]*(1.0-1.0/ratio_)
        gain_lin = 10.0**(gain_db/20.0)
        audio_out = audio*gain_lin
        mk_lin= 10.0**(makeup_db/20.0)
        audio_out*=mk_lin
        return audio_out

    def lookahead_limit_vectorized(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim()==1:
            audio = audio.unsqueeze(0)
        la_ms = 5.0
        la_samps = int(la_ms*self.sample_rate/1000.0)
        la_samps = max(la_samps,1)
        thr_db = self.config.limiter_threshold
        ratio_ = self.config.limiter_ratio

        pad_aud = torch.nn.functional.pad(audio, (la_samps,0))
        pad_aud = pad_aud[...,:audio.size(-1)]
        wsize = int(0.02*self.sample_rate)
        wsize = max(wsize,1)
        w = torch.ones(wsize, device=device)/wsize
        pad_sq = torch.nn.functional.pad(pad_aud**2, (wsize//2,wsize//2))
        conv_out = torch.nn.functional.conv1d(
            pad_sq.unsqueeze(0), w.unsqueeze(0).unsqueeze(0), padding=0
        ).squeeze(0)[...,:audio.size(-1)]
        env = torch.sqrt(torch.clamp(conv_out, min=1e-10))
        env_db = 20.0*torch.log10(env+1e-8)
        diff_db = torch.clamp(env_db-thr_db, min=0.0)
        gain_db = -diff_db*(1.0-1.0/ratio_)
        gain_lin= 10.0**(gain_db/20.0)
        return audio*gain_lin

    def adaptive_spatial_compression(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for adaptive spatial compression. 
        Currently calls single-band compression vectorized for demonstration.
        """
        return self.dynamic_range_compression_vectorized(audio)

    ###########################################################################
    # Summation & Clipping
    ###########################################################################
    def mix_into_final_mix(self, stem_audio: torch.Tensor, final_mix: torch.Tensor, max_length: int) -> torch.Tensor:
        if stem_audio.size(-1)<max_length:
            padlen = max_length - stem_audio.size(-1)
            stem_audio = torch.nn.functional.pad(stem_audio, (0,padlen))
        elif stem_audio.size(-1)>max_length:
            stem_audio = stem_audio[..., :max_length]

        ch_f = final_mix.size(0)
        ch_s = stem_audio.size(0)
        if ch_s<ch_f:
            diff = ch_f-ch_s
            stem_audio = torch.nn.functional.pad(stem_audio, (0,0,0,diff))
        elif ch_s>ch_f:
            stem_audio = stem_audio[:ch_f,...]

        final_mix += stem_audio
        return final_mix

    def reduce_clipping(self, audio: torch.Tensor, target_peak: float=0.99) -> torch.Tensor:
        pk = audio.abs().max()
        if pk>target_peak:
            audio = audio*(target_peak/pk)
        return audio

    ###########################################################################
    # Loudness Normalization
    ###########################################################################
    def normalize_stem_loudness(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim()==1:
            audio_np = audio.cpu().numpy()
            loudness = self.meter.integrated_loudness(audio_np)
            diff = self.target_loudness - loudness
            gain_lin = 10.0**(diff/20.0)
            return audio*gain_lin
        else:
            mixed = audio.mean(dim=0).cpu().numpy()
            loudness = self.meter.integrated_loudness(mixed)
            diff = self.target_loudness - loudness
            gain_lin = 10.0**(diff/20.0)
            return audio*gain_lin

    ###########################################################################
    # Plugin Loading
    ###########################################################################
    def load_plugins(self, plugins_dir:str='plugins'):
        if not os.path.isdir(plugins_dir):
            logger.info(f"No plugins dir '{plugins_dir}'. Skipping plugin load.")
            return
        import importlib.util
        py_files = glob.glob(os.path.join(plugins_dir,"*.py"))
        for pyf in py_files:
            try:
                spec = importlib.util.spec_from_file_location(
                    os.path.splitext(os.path.basename(pyf))[0], pyf
                )
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for atn in dir(mod):
                    attr = getattr(mod, atn)
                    if (isinstance(attr, type) and issubclass(attr, BasePlugin) and attr is not BasePlugin):
                        plugin_params = self.config.plugins.get(atn.lower(), {})
                        plg_inst = attr(plugin_params)
                        self.plugins.append(plg_inst)
                        logger.info(f"Loaded plugin '{atn}' from '{pyf}'.")
            except Exception as e:
                logger.error(f"Failed loading plugin from '{pyf}': {e}")

    ###########################################################################
    # Public stubs if needed
    ###########################################################################
    def harmonic_reconstruction(self, audio: torch.Tensor) -> torch.Tensor:
        return self._harmonic_reconstruction(audio)

    def transient_shaping(self, audio: torch.Tensor) -> torch.Tensor:
        return self._transient_shaping(audio)

    def spectral_eq(self, audio: torch.Tensor, stem_name: str) -> torch.Tensor:
        return self._spectral_shaping(audio, stem_name)

    def iterative_phase_approach(self, audio: torch.Tensor) -> torch.Tensor:
        return self._phase_iterative(audio)

###############################################################################
# CLI
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Advanced Vectorized Psychoacoustic Upmixer (Expanded Placeholders)")
    parser.add_argument('--config_file', type=str, help="Config path (YAML/JSON)")
    parser.add_argument('--sample_rate', type=int, default=44100, help="Sample rate.")
    parser.add_argument('--hrtf_path', type=str, default=None, help="External HRTF path.")
    parser.add_argument('--built_in_hrtf', type=str, default='default', help="Built-in HRTF name.")
    parser.add_argument('--target_loudness', type=float, default=-23.0, help="Target LUFS.")
    parser.add_argument('--params', nargs='*', help="Config overrides (key=val).")
    parser.add_argument('--stems', nargs='+', help="Input stems.")
    parser.add_argument('--output', type=str, default="upmixed_output.wav", help="Output path.")
    parser.add_argument('--plot_analysis', action='store_true', help="Plot debug spectrogram.")
    args = parser.parse_args()

    override_dict = {}
    if args.params:
        for kv in args.params:
            if '=' in kv:
                k,v = kv.split('=',1)
                try:
                    if '.' in v:
                        override_dict[k] = float(v)
                    else:
                        override_dict[k] = int(v)
                except ValueError:
                    vl = v.lower()
                    if vl in ['true','false']:
                        override_dict[k] = (vl=='true')
                    else:
                        override_dict[k] = v

    try:
        upm = PsychoacousticUpmixer(
            sample_rate=args.sample_rate,
            hrtf_path=args.hrtf_path,
            built_in_hrtf=args.built_in_hrtf,
            target_loudness=args.target_loudness,
            config_file=args.config_file,
            **override_dict
        )

        if not args.stems:
            raise FileLoadingError("No stems provided. Use --stems to specify input files.")

        stems, sr = upm.load_stems(args.stems)
        final_mix = upm.psychoacoustic_upmix_stems(stems)

        if args.plot_analysis:
            import matplotlib.pyplot as plt
            import librosa.display
            audio_np = final_mix.mean(dim=0).cpu().numpy()
            S = np.abs(librosa.stft(audio_np, n_fft=upm.config.n_fft, hop_length=upm.config.hop_length))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            plt.figure(figsize=(10,6))
            librosa.display.specshow(S_db, sr=upm.sample_rate, hop_length=upm.config.hop_length,
                                     x_axis='time', y_axis='log')
            plt.colorbar()
            plt.title("Debug Spectrogram of Final Mix")
            plt.show()

        upm.save_audio(args.output, final_mix)
        logger.info(f"Done! Upmixed audio -> {args.output}")

    except UpmixerError as ue:
        logger.error(f"Upmixer error: {ue}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)

if __name__=='__main__':
    main()
