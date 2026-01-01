"""
jWnB - Python SDK
A lightweight experiment tracking library for deep learning

Installation:
    pip install jwnb

Or copy this file to your project and install dependencies:
    pip install requests numpy pillow

Usage:
    import jubaan
    
    run = jubaan.init(
        project="my-project",
        name="experiment-1",
        config={"lr": 0.001, "batch_size": 32}
    )
    
    for epoch in range(100):
        jubaan.log({"loss": loss, "acc": acc}, step=epoch)
    
    jubaan.finish()
"""

import os
import json
import uuid
import time
import hashlib
import logging
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from io import BytesIO
import base64

try:
    import requests
except ImportError:
    raise ImportError("requests is required. Install with: pip install requests")

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jwnb")

# Configuration
API_BASE_URL = os.environ.get("JWNB_API_URL", "https://dl.jubaan.com/api")
API_KEY = os.environ.get("JWNB_API_KEY", "")
LOG_DIR = os.environ.get("JWNB_LOG_DIR", "./jwnb_logs")


def generate_run_id(prefix: str = "") -> str:
    """Generate a unique run ID with optional prefix."""
    unique_id = hashlib.md5(f"{time.time()}{uuid.uuid4()}".encode()).hexdigest()[:8]
    if prefix:
        return f"{prefix}-{unique_id}"
    return unique_id


@dataclass
class Histogram:
    """
    Create a histogram from a list of values.
    Similar to wandb.Histogram()
    
    Args:
        values: List or numpy array of values
        num_bins: Number of histogram bins (default: 64)
        min_val: Minimum value for binning (optional)
        max_val: Maximum value for binning (optional)
    
    Example:
        gradients = model.layer.weight.grad.numpy().flatten()
        jubaan.log({"gradients": jubaan.Histogram(gradients)}, step=epoch)
    """
    values: Union[List, Any]  # Can be list or numpy array
    num_bins: int = 64
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert histogram to bins and counts."""
        if np is not None and hasattr(self.values, '__array__'):
            arr = np.asarray(self.values).flatten()
        else:
            arr = list(self.values)
        
        min_v = self.min_val if self.min_val is not None else min(arr)
        max_v = self.max_val if self.max_val is not None else max(arr)
        
        if np is not None:
            counts, bin_edges = np.histogram(arr, bins=self.num_bins, range=(min_v, max_v))
            return {
                "bins": bin_edges.tolist(),
                "counts": counts.tolist()
            }
        else:
            # Pure Python fallback
            bin_width = (max_v - min_v) / self.num_bins
            bins = [min_v + i * bin_width for i in range(self.num_bins + 1)]
            counts = [0] * self.num_bins
            for v in arr:
                idx = min(int((v - min_v) / bin_width), self.num_bins - 1)
                if 0 <= idx < self.num_bins:
                    counts[idx] += 1
            return {"bins": bins, "counts": counts}


@dataclass 
class Image:
    """
    Log images from matplotlib figures, PIL images, or numpy arrays.
    Similar to wandb.Image()
    
    Args:
        data: matplotlib figure, PIL Image, numpy array, or file path
        caption: Optional caption for the image
        
    Example:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(losses)
        jubaan.log({"loss_curve": jubaan.Image(fig, caption="Training Loss")}, step=epoch)
    """
    data: Any
    caption: Optional[str] = None
    
    def to_base64(self) -> str:
        """Convert image data to base64 string for upload."""
        buf = BytesIO()
        
        # Handle matplotlib figure
        if hasattr(self.data, 'savefig'):
            self.data.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
        # Handle PIL Image
        elif PILImage is not None and isinstance(self.data, PILImage.Image):
            self.data.save(buf, format='PNG')
            buf.seek(0)
        # Handle numpy array
        elif np is not None and hasattr(self.data, '__array__'):
            arr = np.asarray(self.data)
            if arr.ndim == 2:  # Grayscale
                arr = np.stack([arr] * 3, axis=-1)
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            if PILImage is not None:
                img = PILImage.fromarray(arr.astype(np.uint8))
                img.save(buf, format='PNG')
                buf.seek(0)
            else:
                raise ImportError("PIL is required for numpy array images")
        # Handle file path
        elif isinstance(self.data, (str, Path)):
            with open(self.data, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        else:
            raise ValueError(f"Unsupported image type: {type(self.data)}")
        
        return base64.b64encode(buf.read()).decode()


@dataclass
class Model:
    """
    Log model state dictionaries or checkpoints.
    
    Args:
        state_dict: PyTorch model state dict or similar
        name: Name for the model artifact
        metadata: Optional metadata dict
        
    Example:
        jubaan.log({
            "checkpoint": jubaan.Model(model.state_dict(), name="best_model")
        }, step=epoch)
    """
    state_dict: Dict[str, Any]
    name: str = "model"
    metadata: Optional[Dict[str, Any]] = None
    
    def save_to_file(self, path: str):
        """Save model to file."""
        try:
            import torch
            torch.save(self.state_dict, path)
        except ImportError:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.state_dict, f)


@dataclass
class Run:
    """Represents a single training run."""
    id: str
    project_id: str
    project_name: str
    run_id: str
    run_name: str
    config: Dict[str, Any]
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: str = "running"
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    current_epoch: int = 0
    total_epochs: Optional[int] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    _log_buffer: List[Dict] = field(default_factory=list)
    _flush_interval: float = 5.0
    _last_flush: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)


class JWnBClient:
    """Main client for interacting with jWnB."""
    
    def __init__(self, api_url: str = API_BASE_URL, api_key: str = API_KEY):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.current_run: Optional[Run] = None
        self._setup_local_logging()
        
    def _setup_local_logging(self):
        """Setup local log directory."""
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information."""
        import platform
        import sys
        
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["cuda"] = torch.version.cuda
                info["pytorch"] = torch.__version__
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            info["tensorflow"] = tf.__version__
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info["gpu_count"] = len(gpus)
        except ImportError:
            pass
            
        return info
    
    def _api_request(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """Make API request."""
        url = f"{self.api_url}/{endpoint}"
        try:
            if method == "POST":
                response = requests.post(url, json=data, headers=self._get_headers(), timeout=30)
            elif method == "PUT":
                response = requests.put(url, json=data, headers=self._get_headers(), timeout=30)
            elif method == "GET":
                response = requests.get(url, headers=self._get_headers(), timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response.raise_for_status()
            return response.json() if response.content else None
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed: {e}. Data will be saved locally.")
            return None
    
    def _save_local_log(self, data: Dict, filename: str):
        """Save data to local log file."""
        if self.current_run:
            run_dir = self.log_dir / self.current_run.project_name / self.current_run.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = run_dir / filename
            with open(filepath, 'a') as f:
                f.write(json.dumps(data) + '\n')
    
    def _upload_image(self, image: Image, name: str) -> Optional[str]:
        """Upload image and return URL."""
        try:
            img_data = image.to_base64()
            # In production, upload to storage and return URL
            # For now, save locally and return local path
            if self.current_run:
                run_dir = self.log_dir / self.current_run.project_name / self.current_run.run_id / "images"
                run_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{name}_{self.current_run.current_epoch}.png"
                filepath = run_dir / filename
                
                img_bytes = base64.b64decode(img_data)
                with open(filepath, 'wb') as f:
                    f.write(img_bytes)
                
                # Try to upload to server
                result = self._api_request("POST", "upload", {
                    "filename": filename,
                    "data": img_data,
                    "run_id": self.current_run.id
                })
                
                if result and "url" in result:
                    return result["url"]
                    
                return str(filepath)
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            return None
    
    def _upload_model(self, model: Model) -> Optional[str]:
        """Upload model and return URL."""
        try:
            if self.current_run:
                run_dir = self.log_dir / self.current_run.project_name / self.current_run.run_id / "models"
                run_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{model.name}_{self.current_run.current_epoch}.pt"
                filepath = run_dir / filename
                
                model.save_to_file(str(filepath))
                
                return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None

    def init(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Run:
        """
        Initialize a new run.
        
        Args:
            project: Project name to group runs under
            name: Run name prefix (random ID will be appended)
            config: Configuration dictionary with hyperparameters
            notes: Free-form text notes
            tags: List of tags for organization
            **kwargs: Additional configuration parameters
            
        Returns:
            Run object
            
        Example:
            run = jubaan.init(
                project="cifar10",
                name="resnet-experiment",
                config={"lr": 0.001, "batch_size": 32},
                notes="Testing new architecture"
            )
        """
        if self.current_run is not None:
            logger.warning("A run is already active. Finishing previous run.")
            self.finish()
        
        # Merge kwargs into config
        final_config = config.copy() if config else {}
        final_config.update(kwargs)
        
        # Generate run ID
        run_id = generate_run_id(name or "run")
        run_name = name or "run"
        
        # Get or create project
        project_data = {
            "name": project,
            "description": "",
            "last_activity": datetime.utcnow().isoformat()
        }
        
        # Create project if needed (API will handle deduplication)
        project_result = self._api_request("POST", "projects", project_data)
        project_id = project_result.get("id") if project_result else project
        
        # Create run
        system_info = self._get_system_info()
        
        self.current_run = Run(
            id="",  # Will be set by API
            project_id=project_id,
            project_name=project,
            run_id=run_id,
            run_name=run_name,
            config=final_config,
            notes=notes,
            tags=tags or [],
            system_info=system_info,
            total_epochs=final_config.get("epochs")
        )
        
        # Register run with API
        run_data = {
            "project_id": project_id,
            "project_name": project,
            "run_id": run_id,
            "run_name": run_name,
            "status": "running",
            "config": final_config,
            "notes": notes,
            "tags": tags or [],
            "start_time": self.current_run.start_time,
            "system_info": system_info,
            "total_epochs": final_config.get("epochs")
        }
        
        result = self._api_request("POST", "runs", run_data)
        if result:
            self.current_run.id = result.get("id", "")
        
        # Save locally
        self._save_local_log(run_data, "run_info.json")
        
        logger.info(f"Initialized run: {run_name} ({run_id})")
        logger.info(f"  Project: {project}")
        logger.info(f"  View at: {self.api_url.replace('/api', '')}/run/{self.current_run.id}")
        
        return self.current_run
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics, images, histograms, etc.
        
        Args:
            data: Dictionary of values to log
            step: Epoch/step number (optional)
            
        Example:
            jubaan.log({
                "loss": 0.234,
                "accuracy": 0.95,
                "predictions": jubaan.Image(fig),
                "weights": jubaan.Histogram(weights)
            }, step=epoch)
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call jubaan.init() first.")
        
        epoch = step if step is not None else self.current_run.current_epoch
        self.current_run.current_epoch = epoch
        
        metrics = []
        
        for name, value in data.items():
            metric = {
                "run_id": self.current_run.id,
                "project_id": self.current_run.project_id,
                "epoch": epoch,
                "name": name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if isinstance(value, Histogram):
                metric["value_type"] = "histogram"
                metric["histogram_data"] = value.to_dict()
            elif isinstance(value, Image):
                metric["value_type"] = "image"
                metric["image_url"] = self._upload_image(value, name)
                metric["caption"] = value.caption
            elif isinstance(value, Model):
                metric["value_type"] = "model"
                metric["file_url"] = self._upload_model(value)
            elif isinstance(value, bool):
                metric["value_type"] = "boolean"
                metric["string_value"] = str(value)
            elif isinstance(value, str):
                metric["value_type"] = "string"
                metric["string_value"] = value
            elif isinstance(value, (list, tuple)):
                metric["value_type"] = "list"
                metric["list_value"] = list(value)
            elif isinstance(value, (int, float)):
                metric["value_type"] = "scalar"
                metric["value"] = float(value)
            else:
                # Try to convert to string
                metric["value_type"] = "string"
                metric["string_value"] = str(value)
            
            metrics.append(metric)
        
        # Buffer metrics
        with self.current_run._lock:
            self.current_run._log_buffer.extend(metrics)
        
        # Flush if needed
        if time.time() - self.current_run._last_flush > self.current_run._flush_interval:
            self._flush_logs()
        
        # Save locally
        for metric in metrics:
            self._save_local_log(metric, "metrics.jsonl")
    
    def _flush_logs(self):
        """Flush buffered logs to API."""
        if self.current_run is None:
            return
            
        with self.current_run._lock:
            if not self.current_run._log_buffer:
                return
            
            metrics = self.current_run._log_buffer.copy()
            self.current_run._log_buffer.clear()
            self.current_run._last_flush = time.time()
        
        # Send to API
        self._api_request("POST", "metrics/bulk", {"metrics": metrics})
        
        # Update run status
        self._api_request("PUT", f"runs/{self.current_run.id}", {
            "current_epoch": self.current_run.current_epoch
        })
    
    def finish(
        self,
        status: str = "finished",
        summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """
        Finish the current run.
        
        Args:
            status: Final status ("finished" or "crashed")
            summary: Summary metrics to store
            error: Error message if crashed
            
        Example:
            jubaan.finish(summary={"best_accuracy": 0.95})
        """
        if self.current_run is None:
            logger.warning("No active run to finish.")
            return
        
        # Flush remaining logs
        self._flush_logs()
        
        # Update run
        end_data = {
            "status": status,
            "end_time": datetime.utcnow().isoformat(),
            "summary": summary or {}
        }
        
        if error:
            end_data["summary"]["error"] = error
        
        self._api_request("PUT", f"runs/{self.current_run.id}", end_data)
        
        # Save locally
        self._save_local_log(end_data, "run_finish.json")
        
        logger.info(f"Run finished: {self.current_run.run_name} ({status})")
        
        self.current_run = None
    
    def alert(self, title: str, text: str, level: str = "info"):
        """Send an alert/notification."""
        if self.current_run:
            self._api_request("POST", "alerts", {
                "run_id": self.current_run.id,
                "title": title,
                "text": text,
                "level": level
            })


# Global client instance
_client = JWnBClient()


# Convenience functions (module-level API)
def init(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> Run:
    """Initialize a new run. See JWnBClient.init for details."""
    return _client.init(project, name, config, notes, tags, **kwargs)


def log(data: Dict[str, Any], step: Optional[int] = None):
    """Log metrics. See JWnBClient.log for details."""
    _client.log(data, step)


def finish(
    status: str = "finished",
    summary: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """Finish the current run. See JWnBClient.finish for details."""
    _client.finish(status, summary, error)


def alert(title: str, text: str, level: str = "info"):
    """Send an alert."""
    _client.alert(title, text, level)


# Aliases for common patterns
config = lambda: _client.current_run.config if _client.current_run else {}
run = lambda: _client.current_run


if __name__ == "__main__":
    # Example usage
    run = init(
        project="test-project",
        name="test-run",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        notes="Testing the SDK"
    )
    
    import random
    for epoch in range(10):
        loss = 1.0 / (epoch + 1) + random.random() * 0.1
        acc = min(0.95, 0.5 + epoch * 0.05 + random.random() * 0.05)
        
        log({
            "train/loss": loss,
            "train/accuracy": acc,
        }, step=epoch)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
    
    finish(summary={"final_accuracy": acc})
