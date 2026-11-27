from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class DualModalDegradationProtocol:
    version: str
    severity_levels: Tuple[int, ...]
    hsi_degradations: Tuple[str, ...]
    lidar_degradations: Tuple[str, ...]
    cross_modal_degradations: Tuple[str, ...]
    hsi_parameters: Dict[str, Dict[int, Dict[str, Any]]]
    lidar_parameters: Dict[str, Dict[int, Dict[str, Any]]]
    cross_modal_parameters: Dict[str, Dict[int, Dict[str, Any]]]
    cache_dirname: str
    cache_dtype: str
    config_sha1: str

    @classmethod
    def from_config_bundle(cls, bundle: Mapping[str, Any]) -> "DualModalDegradationProtocol":
        raw = dict(bundle["degradation"])
        hsi = dict(raw["hsi"])
        lidar = dict(raw["lidar"])
        cross = dict(raw.get("cross_modal", {"degradations": [], "parameters": {}}))
        cache = dict(raw["cache"])

        def convert(section):
            return {
                str(name): {int(k): dict(v) for k, v in dict(spec).items()}
                for name, spec in dict(section).items()
            }

        return cls(
            version=str(raw["protocol_version"]),
            severity_levels=tuple(int(x) for x in raw["severity_levels"]),
            hsi_degradations=tuple(str(x) for x in hsi["degradations"]),
            lidar_degradations=tuple(str(x) for x in lidar["degradations"]),
            cross_modal_degradations=tuple(str(x) for x in cross.get("degradations", [])),
            hsi_parameters=convert(hsi["parameters"]),
            lidar_parameters=convert(lidar["parameters"]),
            cross_modal_parameters=convert(cross.get("parameters", {})),
            cache_dirname=str(cache["dirname"]),
            cache_dtype=str(cache["dtype"]),
            config_sha1=str(bundle["hashes"]["degradation_config_sha1"]),
        )

    def hsi_params(self, name: str, severity: int) -> Dict[str, Any]:
        return dict(self.hsi_parameters[name][int(severity)])

    def lidar_params(self, name: str, severity: int) -> Dict[str, Any]:
        return dict(self.lidar_parameters[name][int(severity)])

    def cross_modal_params(self, name: str, severity: int) -> Dict[str, Any]:
        return dict(self.cross_modal_parameters[name][int(severity)])
