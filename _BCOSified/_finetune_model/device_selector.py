
from typing import List, Optional, Sequence, Tuple

import logging
from contextlib import suppress
import subprocess

import torch

logger = logging.getLogger(__name__)

class DeviceSelector:
    """
    Select one or more PyTorch devices based on user input and runtime availability.

    Parameters
    ----------
    requested:
        Optional device specification (``"cuda:0"``, ``"cpu"``, etc.). May be a single
        string or any sequence of strings. When omitted, the selector chooses the most
        memory-available CUDA devices automatically.

    Notes
    -----
    * Validation ensures requested CUDA indices exist and that the CUDA runtime is
      available before returning devices.
    * When no specification is provided, devices are ordered by free memory so the
      first one returned is the most suitable primary device.

    Examples
    --------
    Request the default (highest-memory) CUDA device::

        selector = DeviceSelector()
        device = selector.select()[0]

    Pin training to a specific GPU and reserve two devices for multi-GPU workloads::

        selector = DeviceSelector(["cuda:1", "cuda:2"])
        devices = selector.select(count=2)
    """

    def __init__(self, requested: Optional[str | Sequence[str]] = None) -> None:
        if requested is None:
            self._requested: Optional[List[str]] = None
        elif isinstance(requested, str):
            self._requested = [requested]
        else:
            self._requested = [str(spec) for spec in requested]

    def select(self, count: int = 1) -> List[torch.device]:
        if count < 1:
            raise ValueError("Requested device count must be at least 1.")

        if self._requested:
            if len(self._requested) < count:
                raise ValueError(
                    f"Requested {count} device(s) but only {len(self._requested)} specification(s) provided."
                )
            devices = [self._validate_device(spec) for spec in self._requested[:count]]
            return devices

        return self._auto_select(count)

    def _validate_device(self, spec: str) -> torch.device:
        device = torch.device(spec)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA runtime is unavailable.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index out of range: {spec}")
        elif device.type != "cpu":
            raise RuntimeError(f"Unsupported device type requested: {spec}")
        return device

    def _auto_select(self, count: int) -> List[torch.device]:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available for training.")

        free_per_device = self._query_free_memory()

        visible = torch.cuda.device_count()
        free_per_device = [(mem, idx) for mem, idx in free_per_device if idx < visible]

        if len(free_per_device) < count:
            raise RuntimeError(
                f"Only {len(free_per_device)} CUDA device(s) available, but {count} requested."
            )

        free_per_device.sort(reverse=True)
        selected = free_per_device[:count]
        devices: List[torch.device] = []
        for free_mem, idx in selected:
            if free_mem <= 0:
                raise RuntimeError("Selected CUDA device has no free memory.")
            devices.append(torch.device(f"cuda:{idx}"))
        return devices

    def _query_free_memory(self) -> List[Tuple[int, int]]:
        queried = self._query_with_nvidia_smi()
        if queried:
            return queried

        logger.debug("Falling back to torch.cuda.mem_get_info for free memory query.")
        free_per_device: List[Tuple[int, int]] = []
        for idx in range(torch.cuda.device_count()):
            try:
                free_mem, _ = torch.cuda.mem_get_info(idx)
            except RuntimeError:
                continue
            free_per_device.append((free_mem, idx))
        visible = torch.cuda.device_count()
        return [(mem, idx) for mem, idx in free_per_device if idx < visible]

    def _query_with_nvidia_smi(self) -> List[Tuple[int, int]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.debug("nvidia-smi unavailable; cannot query free memory externally.")
            return []

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        free_per_device: List[Tuple[int, int]] = []
        for idx, value in enumerate(lines):
            try:
                free_mb = float(value)
            except ValueError:
                logger.debug("Skipping nvidia-smi entry %r (non-numeric).", value)
                continue
            free_bytes = int(free_mb * 1024 * 1024)
            free_per_device.append((free_bytes, idx))

        return free_per_device


def demo() -> None:
    logging.basicConfig(level=logging.INFO)
    selector = DeviceSelector()

    with suppress(Exception):
        auto_device = selector.select(1)
        logger.info("Auto-selected device: %s", ", ".join(str(d) for d in auto_device))

    requested_selector = DeviceSelector(["cuda:0", "cuda:1"])
    with suppress(Exception):
        requested_devices = requested_selector.select(2)
        logger.info(
            "Requested devices: %s", ", ".join(str(dev) for dev in requested_devices)
        )


if __name__ == "__main__":
    demo()
