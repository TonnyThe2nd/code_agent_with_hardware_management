from ..domain.entities import HardwareProfile
from ..domain.value_objects import Memory, ComputeCapability, ComputeBackend
from .system_probe import SystemProbe


class NvidiaProbe:
    """detecta GPU NVIDIA via pynvml. Cai para SystemProbe se não houver."""

    def __init__(self):
        self._fallback = SystemProbe()
        self._pynvml = None

    def _load(self):
        if self._pynvml is not None:
            return self._pynvml
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            return pynvml
        except Exception:
            return None

    def is_available(self) -> bool:
        pynvml = self._load()
        if pynvml is None:
            return False
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False

    def detect(self) -> HardwareProfile:
        pynvml = self._load()
        if pynvml is None:
            return self._fallback.detect()

        base = self._fallback.detect()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = Memory(mem_info.total / (1024 ** 3))

            return HardwareProfile(
                total_ram=base.total_ram,
                cpu_cores=base.cpu_cores,
                capability=ComputeCapability(
                    backend=ComputeBackend.CUDA,
                    device_name=name,
                    vram=vram,
                ),
            )
        except Exception:
            return base