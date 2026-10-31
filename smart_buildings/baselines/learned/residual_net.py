"""Small DeepSets residual network for variable-size HVAC actions.

Zone features are pooled per AHU (mean + max), AHU features are pooled to a
building embedding, then heads emit bounded residuals for continuous setpoints
and logits for discrete run commands. Designed for millisecond CPU inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ZONE_FEAT = 6  # temp_err_lo, temp_err_hi, damper, reheat, occ_proxy, oat_norm
AHU_FEAT = 8  # sat, pressure, recirc, fan, oat, n_zones, mean_err, max_abs_err
BUILDING_FEAT = 6  # comfort_now, comfort_soon, hod_sin, hod_cos, oat, occupancy


class ResidualNet(nn.Module):
    def __init__(self, hidden: int = 64, residual_scale: float = 0.25):
        super().__init__()
        self.residual_scale = residual_scale
        self.zone_enc = nn.Sequential(
            nn.Linear(ZONE_FEAT, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.ahu_enc = nn.Sequential(
            nn.Linear(AHU_FEAT + 2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.building_enc = nn.Sequential(
            nn.Linear(BUILDING_FEAT + 2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Residuals: SAT, pressure, recirc per AHU; water temp/pressure; damper/reheat per zone
        self.ahu_cont_head = nn.Linear(hidden * 2, 3)
        self.water_head = nn.Linear(hidden, 2)
        self.zone_cont_head = nn.Linear(hidden * 2, 2)
        self.ahu_disc_head = nn.Linear(hidden * 2, 1)
        self.water_disc_head = nn.Linear(hidden, 1)

    def forward(
        self,
        zone_feats: torch.Tensor,  # [V, ZONE_FEAT]
        ahu_feats: torch.Tensor,  # [K, AHU_FEAT]
        building_feats: torch.Tensor,  # [BUILDING_FEAT]
        zone_to_ahu: torch.Tensor,  # [V] int64 AHU index
    ) -> dict[str, torch.Tensor]:
        k = ahu_feats.shape[0]
        v = zone_feats.shape[0]
        z = self.zone_enc(zone_feats)  # [V, H]

        # Pool zones -> AHU (vectorized and differentiable)
        h = z.shape[-1]
        ahu_mean = torch.zeros(k, h, device=z.device).index_add(0, zone_to_ahu, z)
        counts = torch.zeros(k, 1, device=z.device).index_add(
            0, zone_to_ahu, torch.ones(v, 1, device=z.device)
        ).clamp(min=1.0)
        ahu_mean = ahu_mean / counts

        ahu_max = torch.zeros(k, h, device=z.device).scatter_reduce(
            0,
            zone_to_ahu.unsqueeze(-1).expand(-1, h),
            z,
            reduce="amax",
            include_self=False,
        )

        a_in = torch.cat([ahu_feats, ahu_mean, ahu_max], dim=-1)
        a = self.ahu_enc(a_in)  # [K, H]

        b_mean = a.mean(dim=0)
        b_max = a.max(dim=0).values
        b_in = torch.cat([building_feats, b_mean, b_max], dim=-1)
        b = self.building_enc(b_in)  # [H]

        a_ctx = torch.cat([a, b.unsqueeze(0).expand(k, -1)], dim=-1)
        ahu_cont = torch.tanh(self.ahu_cont_head(a_ctx)) * self.residual_scale
        ahu_disc_logit = self.ahu_disc_head(a_ctx).squeeze(-1)

        water_cont = torch.tanh(self.water_head(b)) * self.residual_scale
        water_disc_logit = self.water_disc_head(b).squeeze(-1)

        # Zone residuals conditioned on their AHU embedding
        z_ctx = torch.cat([z, a[zone_to_ahu]], dim=-1)
        zone_cont = torch.tanh(self.zone_cont_head(z_ctx)) * self.residual_scale

        return {
            "ahu_cont": ahu_cont,  # [K, 3]
            "ahu_disc_logit": ahu_disc_logit,  # [K]
            "water_cont": water_cont,  # [2]
            "water_disc_logit": water_disc_logit,  # scalar
            "zone_cont": zone_cont,  # [V, 2]
        }
