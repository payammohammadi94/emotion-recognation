import numpy as np
import scipy.io
import pandas as pd
import torch
from scipy import stats, signal, fft
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.stats import entropy as kl_entropy
from scipy.signal import welch, coherence, butter, sosfiltfilt
from collections import Counter
from mne.time_frequency import psd_array_multitaper
from mne.preprocessing import ICA
import mne
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon
import warnings
# import cupy as cp  # Uncomment if you have cupy installed for GPU acceleration
import networkx as nx
from tqdm import tqdm
from math import factorial
import joblib
from scipy.stats import entropy
import time
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
import os
from PyQt6.QtCore import QThread,QTimer,QObject,pyqtSignal,pyqtSlot
from datetime import datetime
from database_handler import EmotionDatabase
from pdf_utils import EmotionReport



current_dir = Path(__file__).parent

warnings.filterwarnings('ignore')

# تنظیم device برای GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 EEG processing using device: {device}")

# بررسی دسترسی CUDA
if torch.cuda.is_available():
    print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ CUDA not available, using CPU for EEG processing")

class EEGFeatureExtractor:
    def __init__(self, sf=200, compute_all_features=True):
        self.sf = sf  # فرکانس نمونه‌برداری
        self.feature_names = [
            'Mean', 'Energy', 'Teager_Energy', 'MSE_1', 'MSE_2', 'MSE_3',
            'Multivariate_PE', 'Permutation_Entropy', 'Renyi_Entropy',
            'Response_Entropy', 'SampEn', 'Shannon_Entropy', 'Spectral_Autocorrelation',
            'Spectral_Complexity', 'Spectral_Edge_Freq', 'Spectral_Entropy',
            'Spectral_Flatness', 'Spectral_Median_Freq', 'Std', 'State_Entropy',
            'SVDEntropy', 'Transfer_Entropy', 'ApproximateEntropy', 'Bubble_Entropy',
            'Coefficient_of_Variation', 'Coherence', 'Correlation', 'CSP_Feature',
            'Differential_Entropy', 'DispersionEn', 'FFT_Mean', 'FGI_Feature',
            'Fractal_Dimension', 'Fuzzy_Entropy', 'Skewness', 'Kurtosis',
            'Hjorth_Activity', 'Hjorth_Mobility', 'Hjorth_Complexity', 'IQR'
        ]
        # Cache for frequency domain features
        self.fft_cache = {}
        self.psd_cache = {}
        
        # Control which features to compute (for speed optimization)
        self.compute_all_features = compute_all_features
        
        # Define feature groups by computational cost
        self.cheap_features = {
            'Mean', 'Energy', 'Teager_Energy', 'Std', 'Coefficient_of_Variation',
            'Skewness', 'Kurtosis', 'Hjorth_Activity', 'Hjorth_Mobility',
            'Hjorth_Complexity', 'IQR'
        }
        
        self.moderate_features = {
            'Shannon_Entropy', 'Spectral_Entropy', 'Spectral_Flatness',
            'Spectral_Median_Freq', 'Spectral_Edge_Freq', 'FFT_Mean',
            'Differential_Entropy', 'CSP_Feature'
        }
        
        self.expensive_features = {
            'MSE_1', 'MSE_2', 'MSE_3', 'Multivariate_PE', 'Permutation_Entropy',
            'Renyi_Entropy', 'Response_Entropy', 'SampEn', 'Spectral_Autocorrelation',
            'Spectral_Complexity', 'State_Entropy', 'SVDEntropy', 'Transfer_Entropy',
            'ApproximateEntropy', 'Bubble_Entropy', 'Coherence', 'Correlation',
            'DispersionEn', 'FGI_Feature', 'Fractal_Dimension', 'Fuzzy_Entropy'
        }
    
    def permutation_entropy(self, signal, order=3, delay=1, normalize=False):
  
        try:
            n = len(signal)
            permutations = []
            
            for i in range(n - (order - 1) * delay):
                pattern = signal[i:i + order * delay:delay]
                perm_pattern = tuple(np.argsort(pattern))
                permutations.append(perm_pattern)
            
            unique_perms, counts = np.unique(permutations, return_counts=True, axis=0)
            probs = counts / counts.sum()
            
            pe = -np.sum(probs * np.log2(probs + 1e-10))
            
            if normalize:
                pe /= np.log2(factorial(order))
            
            return pe
        except:
            return np.nan

    def sample_entropy(self, signal, tolerance=None):
        try:
            if tolerance is None:
                tolerance = 0.2 * np.std(signal)
            
            n = len(signal)
            m = 2
            
            def _maxdist(x, y):
                return np.max(np.abs(x - y))
            
            def _phi(m):
                x = np.array([signal[i:i+m] for i in range(n - m + 1)])
                C = 0
                for i in range(len(x)):
                    for j in range(i+1, len(x)):
                        if _maxdist(x[i], x[j]) <= tolerance:
                            C += 1
                return C / (n - m + 1)
            
            return -np.log(_phi(m+1) / _phi(m))
        except:
            return np.nan

    def svd_entropy(self, signal, normalize=False):
        try:
            u, s, _ = np.linalg.svd(signal.reshape(-1, 1))
            s_norm = s / np.sum(s)
            svd_e = -np.sum(s_norm * np.log2(s_norm + 1e-10))
            
            if normalize:
                svd_e /= np.log2(len(signal))
            
            return svd_e
        except:
            return np.nan

    def response_entropy(self, signal, f_low=0.5, f_high=47):
        try:
            freqs, psd = welch(signal, fs=self.sf, nperseg=min(256, len(signal)))
            valid_idx = np.where((freqs >= f_low) & (freqs <= f_high))
            psd_selected = psd[valid_idx]
            psd_prob = psd_selected / (np.sum(psd_selected) + 1e-10)
            return -np.sum(psd_prob * np.log2(psd_prob + 1e-10))
        except:
            return np.nan
    
    def state_entropy(self, signal, num_bins=10):
        try:
            digitized = np.digitize(signal, np.linspace(signal.min(), signal.max(), num_bins))
            state_counts = Counter(digitized)
            probabilities = np.array(list(state_counts.values())) / len(signal)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        except:
            return np.nan
    
    def transfer_entropy(self, source, destination, lag=1, k=5):
     
        try:
            if len(source) <= lag or len(destination) <= lag:
                return np.nan
            
            # Ensure source and destination have the same length
            min_length = min(len(source), len(destination)) - lag
            if min_length < 1:
                return np.nan
            
            source = source[:min_length]
            dest_past = destination[:min_length]
            dest_future = destination[lag:min_length + lag]
            
            # Verify that all arrays have the same length
            if not (len(source) == len(dest_past) == len(dest_future)):
                return np.nan
            
            # Mutual Information with fallback for GPU/CPU
            def mutual_info(x, y):
                try:
                    # Try GPU with cupy if available
                    import cupy as cp
                    x = cp.asarray(x).reshape(-1, 1)
                    y = cp.asarray(y).reshape(-1, y.shape[1] if y.ndim > 1 else 1)
                    min_rows = min(x.shape[0], y.shape[0])
                    x = x[:min_rows]
                    y = y[:min_rows]
                    data = cp.concatenate((x, y), axis=1)
                    tree = cKDTree(cp.asnumpy(data))
                    distances, _ = tree.query(cp.asnumpy(data), k+1)
                    avg_log_eps = np.mean(np.log(distances[:, -1] + 1e-10))
                    return digamma(len(x)) - digamma(k) + avg_log_eps
                except ImportError:
                    # Fallback to CPU numpy implementation
                    x = np.asarray(x).reshape(-1, 1)
                    y = np.asarray(y).reshape(-1, y.shape[1] if y.ndim > 1 else 1)
                    min_rows = min(x.shape[0], y.shape[0])
                    x = x[:min_rows]
                    y = y[:min_rows]
                    data = np.concatenate((x, y), axis=1)
                    tree = cKDTree(data)
                    distances, _ = tree.query(data, k+1)
                    avg_log_eps = np.mean(np.log(distances[:, -1] + 1e-10))
                    return digamma(len(x)) - digamma(k) + avg_log_eps
            
            # Compute transfer entropy
            TE = mutual_info(dest_future, np.column_stack((dest_past, source))) - \
                 mutual_info(dest_future, dest_past)
            return float(TE)
        except:
            return np.nan
    
    def bubble_entropy(self, signal, m=3):
        try:
            # Try GPU implementation with cupy if available
            try:
                import cupy as cp
                signal_gpu = cp.asarray(signal)
                N = len(signal_gpu)
                if N < m + 1:
                    return np.nan
                
                vectors = cp.zeros((N - m + 1, m))
                for i in range(m):
                    vectors[:, i] = signal_gpu[i:i+N-m+1]
                
                swaps = cp.zeros(N - m + 1)
                for i in range(N - m + 1):
                    vec = vectors[i].copy()
                    for j in range(m-1):
                        for k in range(m-1-j):
                            if vec[k] > vec[k+1]:
                                vec[k], vec[k+1] = vec[k+1], vec[k]
                                swaps[i] += 1
                
                swap_counts = cp.unique(swaps, return_counts=True)[1]
                probs = swap_counts / cp.sum(swap_counts)
                return kl_entropy(cp.asnumpy(probs))
            except ImportError:
                # Fallback to CPU numpy implementation
                N = len(signal)
                if N < m + 1:
                    return np.nan
                
                vectors = np.zeros((N - m + 1, m))
                for i in range(m):
                    vectors[:, i] = signal[i:i+N-m+1]
                
                swaps = np.zeros(N - m + 1)
                for i in range(N - m + 1):
                    vec = vectors[i].copy()
                    for j in range(m-1):
                        for k in range(m-1-j):
                            if vec[k] > vec[k+1]:
                                vec[k], vec[k+1] = vec[k+1], vec[k]
                                swaps[i] += 1
                
                swap_counts = np.unique(swaps, return_counts=True)[1]
                probs = swap_counts / np.sum(swap_counts)
                return kl_entropy(probs)
        except:
            return np.nan
    
    def csp_feature(self, signal):
        try:
            return np.log(np.var(signal) + 1e-8)
        except:
            return np.nan
    
    def dispersion_entropy(self, signal, m=2, c=6):
        try:
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
            symbols = np.digitize(signal, np.linspace(0, 1, c+1)) - 1
            patterns = [tuple(symbols[i:i+m]) for i in range(len(symbols) - m + 1)]
            unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
            probs = counts / len(patterns)
            return -np.sum(probs * np.log2(probs + 1e-10))
        except:
            return np.nan
    
    def fgi_feature(self, signal):
        try:
            return entropy(np.abs(fft.fft(signal))**2 + 1e-10)
        except:
            return np.nan
    
    def fuzzy_entropy(self, signal, m=2, r=None, n=2):
        try:
            signal = np.asarray(signal)
            N = len(signal)
            if r is None:
                r = 0.2 * np.std(signal)
            
            def _phi(signal, m):
                X = np.array([signal[i:i+m] for i in range(N - m + 1)])
                D = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
                return np.mean(np.exp(-(D / r) ** n))
            
            phi_m = _phi(signal, m)
            phi_m1 = _phi(signal, m+1)
            return np.log(phi_m) - np.log(phi_m1)
        except:
            return np.nan
    
    def kolmogorov_entropy(self, signal, m_max=10, r=None):
        try:
            signal = np.asarray(signal)
            if r is None:
                r = 0.2 * np.std(signal)
            
            def correlation_sum(m):
                X = np.array([signal[i:i+m] for i in range(len(signal) - m + 1)])
                tree = cKDTree(X)
                distances, _ = tree.query(X, k=2)
                return np.mean(distances[:, 1] < r)
            
            C_m = [correlation_sum(m) for m in range(1, m_max+1)]
            K = 0
            for m in range(1, m_max):
                if C_m[m] > 1e-10 and C_m[m-1] > 1e-10:
                    K += np.log(C_m[m-1] / C_m[m])
            return K
        except:
            return np.nan
    
    def kraskov_entropy(self, signal, k=5):
        try:
            signal = np.asarray(signal).reshape(-1, 1)
            tree = cKDTree(signal)
            distances, _ = tree.query(signal, k+1)
            avg_log_eps = np.mean(np.log(distances[:, -1] + 1e-10))
            return -digamma(k) + digamma(len(signal)) + avg_log_eps
        except:
            return np.nan
    
    def extract_features(self, signal, compute_level='all'):
        """
        
        compute_level: 
            'cheap' - compute only cheap features
            'moderate' - compute cheap and moderate features
            'all' - compute all features (default)
        """
        if compute_level not in ['cheap', 'moderate', 'all']:
            compute_level = 'moderate'
            
        features = []
        
        if torch.is_tensor(signal):
            signal = signal.cpu().numpy()
            
        # Generate cache key based on signal data
        cache_key = hash(signal.tobytes())
        
        try:
            # Calculate FFT once and cache it
            if cache_key not in self.fft_cache:
                self.fft_cache[cache_key] = fft.fft(signal)
            fft_result = self.fft_cache[cache_key]
            
            # Calculate PSD once and cache it
            if cache_key not in self.psd_cache:
                self.psd_cache[cache_key] = np.abs(fft_result)**2
            psd = self.psd_cache[cache_key]
            psd_norm = psd / (np.sum(psd) + 1e-10)
        except:
            fft_result = None
            psd = None
            psd_norm = None
        
        # ----- CHEAP FEATURES (Always compute these) -----
        # Mean
        try:
            features.append(np.mean(signal))
        except:
            features.append(np.nan)
        
        # Energy
        try:
            features.append(np.sum(signal ** 2))
        except:
            features.append(np.nan)
        
        # Teager Energy
        try:
            teager = np.sum(signal[1:-1]**2 - signal[:-2] * signal[2:])
            features.append(teager)
        except:
            features.append(np.nan)
            
        # Skip expensive calculations if not needed
        if compute_level == 'cheap':
            # Fill remaining features with zeros or NaN
            features.extend([0] * (len(self.feature_names) - len(features)))
            return features
        
        # ----- MODERATE FEATURES -----
        # These are computed for 'moderate' and 'all' levels
        
        # Shannon Entropy (using cached PSD)
        if psd_norm is not None:
            try:
                features.append(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
            except:
                features.append(np.nan)
        else:
            features.append(np.nan)
            
        # Standard deviation
        try:
            features.append(np.std(signal))
        except:
            features.append(np.nan)
            
        # Skewness
        try:
            features.append(stats.skew(signal))
        except:
            features.append(np.nan)
        
        # Kurtosis
        try:
            features.append(stats.kurtosis(signal))
        except:
            features.append(np.nan)
        
        # Hjorth parameters
        try:
            diff = np.diff(signal)
            activity = np.var(signal)
            mobility = np.sqrt(np.var(diff) / activity)
            complexity = np.sqrt(np.var(np.diff(diff)) / np.var(diff)) / mobility
            features.extend([activity, mobility, complexity])
        except:
            features.extend([np.nan] * 3)
        
        # IQR
        try:
            features.append(np.percentile(signal, 75) - np.percentile(signal, 25))
        except:
            features.append(np.nan)
            
        if compute_level == 'moderate':
            # Fill remaining features with zeros or NaN
            features.extend([0] * (len(self.feature_names) - len(features)))
            return features
            
        # ----- EXPENSIVE FEATURES ----- 
        # Only compute these if compute_level is 'all'
        
        # MSE for different scales
        try:
            for scale in [1, 2, 3]:
                coarse = np.mean(signal[:len(signal)//scale*scale].reshape(-1, scale), axis=1)
                features.append(self.sample_entropy(coarse))
        except:
            features.extend([np.nan] * 3)
            
        # Only add the most important expensive features
        # You can add more based on their importance
        try:
            features.append(self.permutation_entropy(signal, normalize=True))
        except:
            features.append(np.nan)
            
        # Placeholder for remaining features
        # Adjust the number based on how many feature slots are left
        remaining_features = len(self.feature_names) - len(features)
        features.extend([0] * remaining_features)
        
        return features

def higuchi_fd(signal, kmax=5):
    try:
        N = len(signal)
        L = []
        x = []
        for k in range(1, kmax + 1):
            Lk = 0
            for m in range(k):
                sum_L = 0
                for i in range((N - m - 1) // k):
                    sum_L += abs(signal[m + i * k] - signal[m + (i + 1) * k])
                Lk += sum_L * (N - 1) / (((N - m - 1) // k) * k)
            Lk = Lk / k
            L.append(np.log(Lk + 1e-10))
            x.append(np.log(1.0 / k))
        coeffs = np.polyfit(x, L, 1)
        return coeffs[0]
    except:
        return np.nan

def preprocess_eeg(data, sfreq=200, l_freq=1, h_freq=40, channel_names=None, ref_channels=['Cz', 'Pz']):
    """Preprocess EEG data: bandpass filter, re-reference, and ICA"""
    try:
        # Validate input data
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array (n_channels, n_samples)")
        
        n_channels = data.shape[0]
        if n_channels < 1:
            raise ValueError("Input data must have at least one channel")
        
        # Initialize channel_names
        if channel_names is None:
            channel_names = [f'EEG{i+1}' for i in range(n_channels)]
        elif len(channel_names) != n_channels:
            raise ValueError(f"Length of channel_names ({len(channel_names)}) must match number of channels ({n_channels})")
        
        # Create MNE Info object
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        # Set montage (standard 10-20 system)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')
        
        # Re-reference to average of Cz and Pz
        if all(ch in channel_names for ch in ref_channels):
            raw.set_eeg_reference(ref_channels=ref_channels, projection=False)
        else:
            print(f"Warning: Reference channels {ref_channels} not found in {channel_names}. Using average reference.")
            raw.set_eeg_reference(ref_channels='average')
        
        # Apply ICA
        n_components = min(20, n_channels)
        ica = ICA(n_components=n_components, random_state=97, max_iter=800)
        ica.fit(raw)
        
        # Automatically detect and exclude artifact components (e.g., eye blinks)
        ica.exclude = []
        if 'Fp1' in channel_names:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1', threshold=3.0)
            ica.exclude.extend(eog_indices)
            print(f"Excluding ICA components: {ica.exclude}")
        else:
            print("Warning: Fp1 not found for EOG artifact detection. Skipping automatic EOG removal.")
        
        # Apply ICA to remove artifacts
        raw_clean = ica.apply(raw.copy())
        
        # Get preprocessed data
        preprocessed_data = raw_clean.get_data()
        
        return preprocessed_data
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None
    
class EmotionDetector:
    def __init__(self, model_path, sf=200, feature_level='moderate'):
        self.extractor = EEGFeatureExtractor(sf=sf)
        self.model = joblib.load(model_path)
        self.emotion_labels = {0: 'Neutral', 1: 'Happiness', 2: 'Fear', 3: 'Anger', 4: 'Sadness', 5: 'Disgust'}
        self.n_features_expected = 100
        self.selected_feature_indices = None
        self.n_jobs = 4
        self.feature_level = feature_level
        print(f"✅ EmotionDetector initialized with feature level: '{self.feature_level}'")

    def extract_features_parallel(self, signal):
        from joblib import Parallel, delayed
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extractor.extract_features)(signal[ch], self.feature_level)
            for ch in range(signal.shape[0])
        )
        return np.concatenate(features)

    def process_signal(self, signal):
        try:
            start_time = time.time()
            
            print(f"🧠 Extracting features using '{self.feature_level}' setting...")
            features = self.extract_features_parallel(signal)
            
            # Handle NaN and Inf values in features
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Normalize features
            features_mean = np.mean(features)
            features_std = np.std(features)
            if features_std > 0:
                features = (features - features_mean) / features_std
            else:
                features = features - features_mean  # Just center if std is 0

            # Handle feature dimension
            if len(features) > self.n_features_expected:
                features_reshaped = features.reshape(-1, 1)
                feature_variances = np.var(features_reshaped, axis=1)
                # Handle NaN in variances
                feature_variances = np.nan_to_num(feature_variances, nan=0.0)
                selected_indices = np.argsort(feature_variances)[-self.n_features_expected:]
                features = features[selected_indices]
            elif len(features) < self.n_features_expected:
                padded_features = np.zeros(self.n_features_expected)
                padded_features[:len(features)] = features
                features = padded_features

            features = features.reshape(1, -1)

            # Get prediction probabilities with better error handling
            try:
                probabilities = self.model.predict_proba(features)[0]
            except AttributeError:
                try:
                    decision_scores = self.model.decision_function(features)[0]
                    # Handle potential NaN or Inf in decision scores
                    decision_scores = np.nan_to_num(decision_scores, nan=0.0, posinf=1e10, neginf=-1e10)
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    probabilities = exp_scores / (np.sum(exp_scores) + 1e-10)  # Add small epsilon to avoid division by zero
                except Exception as e:
                    print(f"⚠️ Error in probability calculation: {str(e)}")
                    # Fallback to uniform probabilities
                    probabilities = np.ones(len(self.emotion_labels)) / len(self.emotion_labels)

            # Ensure probabilities are valid
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
            probabilities = np.clip(probabilities, 0.0, 1.0)  # Clip to [0, 1] range
            probabilities = probabilities / np.sum(probabilities)  # Normalize to sum to 1

            # Map probabilities to labels
            emotion_probabilities = {}
            for i, prob in enumerate(probabilities):
                if i in self.emotion_labels:
                    emotion_probabilities[self.emotion_labels[i]] = float(prob)  # Convert to Python float
            
            # Get the top prediction
            top_prediction = self.emotion_labels[np.argmax(probabilities)]
            
            processing_time = time.time() - start_time
            print(f"⏱️ Processing time: {processing_time:.3f} seconds")

            return top_prediction, emotion_probabilities
        except Exception as e:
            print(f"❌ Error in processing signal: {str(e)}")
            # Return uniform probabilities in case of error
            uniform_prob = 1.0 / len(self.emotion_labels)
            return 'Error', {label: uniform_prob for label in self.emotion_labels.values()}

def read_eeg_signal(file_path):
    """
    Loads EEG data from an EDF file.
    
    Args:
        file_path (str): Path to the EDF file.

    Returns:
        np.ndarray: EEG data array with shape (n_channels, n_samples).
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Define channel names (standard 10-20 system)
        channel_key = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
        
        # Pick channels if they exist
        existing_channels = [ch for ch in channel_key if ch in raw.ch_names]
        if len(existing_channels) > 0:
            raw.pick_channels(existing_channels)
            
        data = raw.get_data()
        return data
    except Exception as e:
        print(f"Error loading EEG signal: {str(e)}")
        return None

def main(input_file):
    model_path = f'{current_dir}\\svm_poly_model.pkl'  # Path to your trained SVM model    
    # Choose feature level based on performance needs:
    # - 'cheap': Fastest but less accurate
    # - 'moderate': Good balance of speed and accuracy
    # - 'all': Most accurate but slower
    feature_level = 'cheap'  # Try 'cheap' for maximum speed
    
    detector = EmotionDetector(model_path, sf=200, feature_level=feature_level)
    
    # Define standard channel names (used throughout the function)
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
    
    # Example: Process a single EEG file
  # Replace with your EEG data file
    print("Loading EEG file...")
    raw_data = read_eeg_signal(input_file)
    
    if raw_data is not None:
        # Ensure data has 21 channels
        if raw_data.shape[0] != 21:
            print(f"Error: Expected 21 channels, got {raw_data.shape[0]}")
            return
        
        print("Preprocessing EEG data...")
        preprocessed_data = preprocess_eeg(raw_data, sfreq=200, l_freq=1, h_freq=40,
        channel_names=channel_names, ref_channels=['Cz', 'Pz'])
        
        if preprocessed_data is not None:
            # Initialize feature selection (optional)
            # Note: This requires training data, so it's commented out
            # To use it, you would need labeled training data
            """
            train_signals = [preprocessed_data]  # You would need more training examples
            train_labels = [0]  # Corresponding emotion labels
            detector.select_optimal_features(train_signals, train_labels)
            """
            
            print("Processing EEG signal...")
            emotion, emotion_probabilities = detector.process_signal(preprocessed_data)
            print(f"Detected Emotion: {emotion}")
        else:
            print("Failed to preprocess EEG data.")
    else:
        print("Failed to load EEG signal.")


class FileHandler(QObject):
    fileSizeChanged = pyqtSignal(str)
    emotionRsult = pyqtSignal(str, float, list, arguments=['emotion_status', 'emotion_prob', 'total_prob'])
    featureLevelChanged = pyqtSignal(str)  # New signal for feature level changes
    
    def __init__(self):
        super().__init__()
        self.current_feature_level = 'cheap'  # Default feature level
        self.data_buffer = []  # ذخیره داده‌های لحظه‌ای
        
        # ایجاد پوشه دیتابیس اگر وجود نداشته باشد
        db_dir = "dataBase/eegRecognation"
        os.makedirs(db_dir, exist_ok=True)
        
        self.db = EmotionDatabase(f"{db_dir}/eeg_emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        
        # تایمر ذخیره‌سازی دوره‌ای
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_to_database)
        self.save_timer.start(60000)  # هر 60 ثانیه
        
        # اتصال signal به saveToMemory
        self.emotionRsult.connect(self.saveToMemory)
    
    @pyqtSlot(str)
    def setFeatureLevel(self, level):
        """Set the feature level for emotion detection"""
        if level in ['cheap', 'moderate', 'all','feature level']:
            if level == 'feature level':
                level = 'cheap'
            self.current_feature_level = level
            print(f"Feature level changed to: {level}")
            self.featureLevelChanged.emit(level)
        else:
            print(f"Invalid feature level: {level}. Using 'cheap' instead.")
            self.current_feature_level = 'cheap'
            self.featureLevelChanged.emit('cheap')
    
        
    @pyqtSlot(str)
    def sendPath(self, file_url):
        print("file path:", file_url)
        local_path = file_url.replace("file:///", "").replace("%20", " ")
        if os.path.exists(local_path):
            size_bytes = os.path.getsize(local_path)
            size_kb = size_bytes / 1024
            human_size = f"{size_kb:.2f} KB"
            print("human_size: ", human_size)
            self.fileSizeChanged.emit(human_size)
            
            # Load and process the EEG file
            raw_data = read_eeg_signal(local_path)
            if raw_data is not None:
                if raw_data.shape[0] != 21:
                    print(f"Error: Expected 21 channels, got {raw_data.shape[0]}")
                    return
                
                print("Preprocessing EEG data...")
                channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                               'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
                preprocessed_data = preprocess_eeg(raw_data, sfreq=200, l_freq=1, h_freq=40,
                                                channel_names=channel_names, ref_channels=['Cz', 'Pz'])
                
                if preprocessed_data is not None:
                    print("Processing EEG signal...")
                    model_path = f'{current_dir}\\svm_poly_model.pkl'
                    detector = EmotionDetector(model_path, sf=200, feature_level=self.current_feature_level)
                    top_emotion, emotion_probabilities = detector.process_signal(preprocessed_data)
                    
                    # Convert emotion probabilities to the format expected by QML
                    total_prob = [
                        emotion_probabilities.get('Anger', 0),
                        emotion_probabilities.get('Disgust', 0),
                        emotion_probabilities.get('Fear', 0),
                        emotion_probabilities.get('Happiness', 0),
                        emotion_probabilities.get('Neutral', 0),
                        emotion_probabilities.get('Sadness', 0),
                        0.0  # surprise (not supported in current model)
                    ]
                    
                    # Get the probability of the top emotion
                    emotion_prob = emotion_probabilities.get(top_emotion, 0)
                    
                    # Emit the emotion result
                    self.emotionRsult.emit(top_emotion, emotion_prob, total_prob)
    
    @pyqtSlot(str, float, list)
    def saveToMemory(self, emotion_status, demotion_prob, total_prob):
        """ذخیره داده‌های emotion در memory"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data_buffer.append({
            "timestamp": timestamp,
            "emotion": emotion_status,
            "prob": demotion_prob,
            "probs": total_prob
        })
        print(f"[*] EEG emotion saved to memory: {emotion_status} ({demotion_prob:.2%}) - Buffer size: {len(self.data_buffer)}")
    
    @pyqtSlot()
    def save_to_database(self):
        """ذخیره داده‌ها در دیتابیس"""
        if self.data_buffer:
            print(f"[*] Saving {len(self.data_buffer)} EEG data points to database...")
            self.db.save_batch(self.data_buffer)
            self.data_buffer.clear()
            print("[✓] EEG data saved to database successfully")
    
    @pyqtSlot(str, str, int, str)
    def savePersonInfo(self, name, lastName, age, nationalCode):
        """ذخیره اطلاعات شخص و ایجاد دیتابیس جدید"""
        self.person_info = {
            "name": name,
            "lastName": lastName,
            "age": age,
            "nationalCode": nationalCode
        }
        # ایجاد پوشه دیتابیس اگر وجود نداشته باشد
        db_dir = "dataBase/eegRecognation"
        os.makedirs(db_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f"{db_dir}/eeg_emotions_{self.person_info.get('name')}_{self.person_info.get('lastName')}_{self.person_info.get('age')}_{self.person_info.get('nationalCode')}_{timestamp}.db"
        print(f"[*] EEG Database path: {db_path}")
        self.db = EmotionDatabase(db_path)
    
    @pyqtSlot()
    def generatePdfReport(self):
        """تولید گزارش PDF برای EEG"""
        # ذخیره داده‌های باقی‌مانده در memory قبل از تولید گزارش
        if self.data_buffer:
            print(f"[*] Saving {len(self.data_buffer)} remaining EEG data points to database...")
            self.save_to_database()
        
        data = self.db.fetch_all()
        if not data:
            print("[-] No EEG data to generate report.")
            print(f"[*] Data buffer size: {len(self.data_buffer)}")
            print(f"[*] Database path: {self.db.db_path}")
            return
        print(f"[*] Generating PDF report with {len(data)} EEG data points...")
        report = EmotionReport()
        report.generate_report_eeg_recognation(data)