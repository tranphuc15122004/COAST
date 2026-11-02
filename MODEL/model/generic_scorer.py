"""Generic Capacity- and Information-Matched Scorer for G5.

Replaces the 3 domain-specific inductive heads (Cross-Attention + Fleet Ownership Softmax + Lookahead Scalar)
with a unified generic cross-attention + dense scorer that receives identical inputs:
- Customer representation h_i (GraphEncoder)
- Acting vehicle state v_k (FleetEncoder)
- 8D edge embedding e_{k,i} (EdgeFeatureEncoder)
- Fleet coordination memory M (CoordMemory)

Parameter calibration:
COAST 4 heads: 115,906 parameters (total model: 763,882)
GenericScoringHead: 115,713 parameters (total model: 763,689, delta = -193 params / -0.025%)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from MODEL.model.vectra import VECTRA


class GenericScoringHead(nn.Module):
    """Calibrated Generic Scoring Head for G5 (Capacity- and Information-Matched)."""
    def __init__(self, model_size: int = 128, nhead: int = 4):
        super().__init__()
        self.model_size = model_size
        self.nhead = nhead
        
        # Local candidate projection: [veh_repr (D) + cust_repr (D) + edge_emb (D)] -> D
        self.query_proj = nn.Linear(model_size * 3, model_size)
        
        # Generic multi-head cross attention to fleet coordination memory
        self.fleet_mha = nn.MultiheadAttention(model_size, nhead, batch_first=True)
        self.norm = nn.LayerNorm(model_size)
        
        # Direct projection to scalar compatibility logit
        self.out_proj = nn.Linear(model_size, 1)

    def forward(
        self,
        veh_repr: torch.Tensor,
        cust_repr: torch.Tensor,
        edge_emb: torch.Tensor,
        veh_memory: torch.Tensor | None = None,
        veh_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            veh_repr:   (N, 1, D)
            cust_repr:  (N, L_c, D)
            edge_emb:   (N, 1, L_c, D) or (N, L_c, D)
            veh_memory: (N, M_v, D)
            veh_mask:   (N, 1, L_c)
        Returns:
            scores:     (N, 1, L_c)
        """
        N, _, D = veh_repr.shape
        L_c = cust_repr.shape[1]
        
        if edge_emb.dim() == 4:
            edge_emb = edge_emb.squeeze(1) # (N, L_c, D)
            
        veh_exp = veh_repr.expand(-1, L_c, -1) # (N, L_c, D)
        local_cand = torch.cat([veh_exp, cust_repr, edge_emb], dim=-1) # (N, L_c, 3*D)
        query = self.query_proj(local_cand) # (N, L_c, D)
        
        if veh_memory is not None and veh_memory.size(1) > 0:
            fleet_ctx, _ = self.fleet_mha(query, veh_memory, veh_memory)
            h = self.norm(query + fleet_ctx)
        else:
            h = self.norm(query)
            
        logits = self.out_proj(h).squeeze(-1).unsqueeze(1) # (N, 1, L_c)
        
        if veh_mask is not None:
            logits = logits.masked_fill(veh_mask, -1e9)
            
        return logits


class RoleAwareGenericScoringHead(GenericScoringHead):
    """Generic scorer that distinguishes the acting vehicle memory from peer memories."""
    def __init__(self, model_size: int = 128, nhead: int = 4):
        super().__init__(model_size, nhead)
        self.role_embedding = nn.Embedding(2, model_size)

    def forward(
        self,
        veh_repr: torch.Tensor,
        cust_repr: torch.Tensor,
        edge_emb: torch.Tensor,
        veh_memory: torch.Tensor | None = None,
        active_vehicle_idx: torch.Tensor | None = None,
        veh_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if veh_memory is None or veh_memory.size(1) == 0:
            return super().forward(veh_repr, cust_repr, edge_emb, veh_memory, veh_mask)
        if active_vehicle_idx is None:
            raise ValueError("active_vehicle_idx is required when vehicle memory is provided")

        batch_size, vehicle_count, _ = veh_memory.shape
        active_idx = active_vehicle_idx.reshape(batch_size, 1).long()
        if ((active_idx < 0) | (active_idx >= vehicle_count)).any():
            raise ValueError("active_vehicle_idx is outside the fleet memory")
        roles = torch.zeros((batch_size, vehicle_count), dtype=torch.long, device=veh_memory.device)
        roles.scatter_(1, active_idx, 1)
        role_memory = veh_memory + self.role_embedding(roles)
        return super().forward(veh_repr, cust_repr, edge_emb, role_memory, veh_mask)


class GenericVECTRA(VECTRA):
    """VECTRA with GenericScoringHead matching COAST capacity within +/-0.03%."""
    def __init__(self, *args, head_count: int = 4, **kwargs):
        super().__init__(*args, head_count=head_count, **kwargs)
        # Remove COAST's 3 domain-specific heads
        del self.cross_fusion
        del self.owner_head
        del self.lookahead_head
        del self.score_fusion
        self.generic_head = GenericScoringHead(self.model_size, nhead=head_count)

    def _compute_owner_bias(self, cur_veh_idx):
        return None

    def _compute_lookahead(self, veh_repr, cust_repr, edge_emb):
        return None

    def _score_customers(self, veh_repr, cust_repr, edge_emb, owner_bias, lookahead, veh_mask=None):
        compat = self.generic_head(veh_repr, cust_repr, edge_emb, self._veh_memory, veh_mask)
        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()
        return compat


class RoleAwareGenericVECTRA(GenericVECTRA):
    """Capacity-matched generic VECTRA with an explicit active/peer memory role."""
    def __init__(self, *args, head_count: int = 4, **kwargs):
        super().__init__(*args, head_count=head_count, **kwargs)
        self.generic_head = RoleAwareGenericScoringHead(self.model_size, nhead=head_count)
        self._active_vehicle_idx = None

    def _compute_owner_bias(self, cur_veh_idx):
        self._active_vehicle_idx = cur_veh_idx
        return None

    def _score_customers(self, veh_repr, cust_repr, edge_emb, owner_bias, lookahead, veh_mask=None):
        compat = self.generic_head(
            veh_repr,
            cust_repr,
            edge_emb,
            self._veh_memory,
            self._active_vehicle_idx,
            veh_mask,
        )
        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()
        return compat
