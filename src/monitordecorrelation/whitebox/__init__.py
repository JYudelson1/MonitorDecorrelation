"""White-box probe monitors: residual-stream linear probes that plug into the ``Monitor`` seam.

v1 trains probes on the **original HF base model** (no tinker LoRA download) and applies them
**post-hoc** over saved rollouts. See docs/MONITORS.md and the plan.
"""

from __future__ import annotations

from monitordecorrelation.whitebox.model import WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe

__all__ = ["WhiteBoxModel", "LinearProbe"]
