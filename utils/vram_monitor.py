# utils/vram_monitor.py
""" 실시간 VRAM 모니터링 (GTX 1070 8GB 최적화 디버깅용) """
import torch
import gc
from typing import Dict

class VRAMMonitor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.enabled = torch.cuda.is_available() if device == "cuda" else False
        
    def get_stats(self) -> Dict[str, float]:
        """ 현재 VRAM 상태 반환 (GB 단위) """
        if not self.enabled:
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024**3)
        free = total - allocated
        
        return {
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(free, 2),
            "total": round(total, 2)
        }
    
    def print_stats(self, label: str = ""):
        """ VRAM 상태 출력 """
        if not self.enabled:
            return
        
        stats = self.get_stats()
        prefix = f"[VRAM{' - ' + label if label else ''}]"
        print(f"{prefix} Allocated: {stats['allocated']}GB | "
              f"Free: {stats['free']}GB / {stats['total']}GB")
    
    def check_oom_risk(self, required_gb: float) -> bool:
        """ OOM 위험 체크 (여유 공간 < 필요량) """
        if not self.enabled:
            return False
        
        stats = self.get_stats()
        return stats['free'] < required_gb
    
    def force_cleanup(self):
        """ 강제 메모리 정리 """
        if not self.enabled:
            return
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()