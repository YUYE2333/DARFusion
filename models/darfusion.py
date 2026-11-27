"""Public DARFusion name; identical version-4 module and checkpoint layout."""
from .pcrfusion import PCRFusion

DARFusion = PCRFusion
DARF = DARFusion
__all__ = ['DARFusion', 'DARF']
