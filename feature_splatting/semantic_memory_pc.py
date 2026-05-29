import torch
import torch.nn.functional as F


class SemanticPointCloudMemory:
    """
    Fixed-capacity semantic memory over 3D points with CLIP features.

    - Memory stores (xyz, feat, age) for up to `capacity` points.
    - update(points, feats): merge nearby new points via EMA, else insert;
      if full, replace oldest entries.
    - query(points, k): kNN fetch with optional spatial/temporal weighting
      and cosine-sim weighting to stabilize identity.

    Assumptions:
      * feats are L2-normalized row-wise before calling update() (we re-normalize after EMA).
      * points/feats device matches memory device (or will be moved).
    """

    def __init__(self, feature_dim: int, capacity: int, device: torch.device | str = "cuda"):
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)

        self.xyz = torch.empty((self.capacity, 3), dtype=torch.float32, device=self.device)
        self.feat = torch.empty((self.capacity, self.feature_dim), dtype=torch.float32, device=self.device)
        self.age = torch.full(
            (self.capacity,), fill_value=10_000, dtype=torch.int32, device=self.device
        )  # large => “old”
        self._size = 0

    # ------------------------- Introspection -------------------------
    @property
    def size(self) -> int:
        return int(self._size)

    def is_full(self) -> bool:
        return self._size >= self.capacity

    # ------------------------- Initialization ------------------------
    @torch.no_grad()
    def bootstrap(self, points_bxn3: torch.Tensor, feats_bxD: torch.Tensor):
        """
        One-shot initialization (no merging). Truncates if > capacity.
        Uses random sampling to avoid spatial bias.
        """
        pts = points_bxn3.to(self.device, dtype=torch.float32)
        feats = feats_bxD.to(self.device, dtype=torch.float32)
        if pts.shape[0] != feats.shape[0]:
            raise ValueError("bootstrap: points and feats must have same first dim")

        n = min(pts.shape[0], self.capacity)

        # # Random permutation to avoid spatial bias when downsampling
        # if pts.shape[0] > self.capacity:
        #     perm = torch.randperm(pts.shape[0], device=self.device)[:n]
        #     pts = pts[perm]
        #     feats = feats[perm]

        self.xyz[:n] = pts[:n]
        self.feat[:n] = F.normalize(feats[:n], dim=1)
        self.age[:n] = 0
        self._size = n

    # --------------------------- Update ------------------------------
    @torch.no_grad()
    def update(
        self,
        points_bxn3: torch.Tensor,
        feats_bxD: torch.Tensor,
        *,
        alpha: float = 0.6,
        r_merge: float = 0.03,
        chunk: int = 12768,
    ):
        """
        Merge-or-insert new (points, feats) into memory.

        alpha: EMA weight for merging (new := alpha*new + (1-alpha)*old).
        r_merge (m): distance threshold to consider a new point the same place.
        chunk: splits kNN to avoid OOM (compute cdist(new_chunk, mem) in pieces).
        """
        if points_bxn3.numel() == 0:
            self.age.add_(1)
            return

        pts_new = points_bxn3.to(self.device, dtype=torch.float32)
        feat_new = F.normalize(feats_bxD.to(self.device, dtype=torch.float32), dim=1)  # safety

        # If memory empty, just fill up to capacity
        if self._size == 0:
            self.bootstrap(pts_new, feat_new)
            return

        M = self._size
        N = pts_new.shape[0]

        # chunked nearest neighbor (k=1)
        best_dist = torch.empty((N,), dtype=torch.float32, device=self.device).fill_(float("inf"))
        best_idx = torch.empty((N,), dtype=torch.int64, device=self.device)

        # To reduce memory: compute cdist in chunks over new points
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            dists = torch.cdist(pts_new[s:e], self.xyz[:M])  # (e-s, M)
            vals, idxs = dists.min(dim=1)
            best_dist[s:e] = vals
            best_idx[s:e] = idxs

        # MERGE where close
        merge_mask = best_dist < r_merge
        if merge_mask.any():
            idx_mem = best_idx[merge_mask]
            # EMA on features then renormalize
            merged = alpha * feat_new[merge_mask] + (1.0 - alpha) * self.feat[idx_mem]
            self.feat[idx_mem] = F.normalize(merged, dim=1)
            # Optional: small xyz EMA to reduce jitter (comment out if you want exact geometry)
            self.xyz[idx_mem] = 0.3 * pts_new[merge_mask] + 0.7 * self.xyz[idx_mem]
            self.age[idx_mem] = 0  # refreshed

        # INSERT/REPLACE for non-merged
        add_mask = ~merge_mask
        if add_mask.any():
            pts_add = pts_new[add_mask]
            feat_add = feat_new[add_mask]
            K = pts_add.shape[0]

            # free slots
            free = self.capacity - self._size
            take_free = min(free, K)
            if take_free > 0:
                i0 = self._size
                i1 = i0 + take_free
                self.xyz[i0:i1] = pts_add[:take_free]
                self.feat[i0:i1] = feat_add[:take_free]
                self.age[i0:i1] = 0
                self._size = i1

            # need replacements?
            remaining = K - take_free
            if remaining > 0:
                # Can't replace more points than we have in memory
                # Use random sampling if there are more candidates than slots
                max_replaceable = self._size
                n_replace = min(remaining, max_replaceable)

                ## random permutation for random sampling
                # if remaining > max_replaceable:
                #     perm = torch.randperm(remaining, device=self.device)[:n_replace]
                #     pts_to_add = pts_add[perm]
                #     feat_to_add = feat_add[perm]

                pts_to_add = pts_add[take_free : take_free + n_replace]
                feat_to_add = feat_add[take_free : take_free + n_replace]

                # replace the OLDEST entries
                replace_idx = torch.topk(self.age[: self._size], k=n_replace, largest=True).indices
                self.xyz[replace_idx] = pts_to_add
                self.feat[replace_idx] = feat_to_add
                self.age[replace_idx] = 0

        # Everyone ages by 1
        self.age[: self._size].add_(1)

    # ---------------------------- Query ------------------------------
    @torch.no_grad()
    def query(
        self,
        points_qxn3: torch.Tensor,
        k: int = 8,
        *,
        w_spatial_sigma: float | None = 0.04,  # meters; None disables spatial weighting
        w_time_tau: float | None = 50.0,  # frames; None disables temporal decay
        chunk: int = 12768,
        # use_cosine_weight: bool = False,  # if you also have per-query feats, see below
        # query_feats_qxD: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return blended features for query points using kNN and weights.

        If query_feats_qxD is provided and use_cosine_weight=True, we weight by
        cosine(query_feat, mem_feat) as well (identity lock). Otherwise skip it.

        Returns:
            (Q, D) L2-normalized features.
        """
        Q = points_qxn3.shape[0]
        pts_q = points_qxn3.to(self.device, dtype=torch.float32)
        M = self._size

        if M == 0:
            raise RuntimeError("query() called on empty memory")

        out = torch.empty((Q, self.feature_dim), dtype=torch.float32, device=self.device)

        # Precompute for temporal weights
        if w_time_tau is not None:
            # Smaller age => larger weight; exp(-age/tau)
            w_time = torch.exp(-self.age[:M].float() / float(w_time_tau))  # (M,)
        else:
            w_time = None

        # Chunked kNN
        for s in range(0, Q, chunk):
            e = min(s + chunk, Q)
            d = torch.cdist(pts_q[s:e], self.xyz[:M])  # (q, M)
            # kNN indices
            k_eff = min(k, M)
            vals, idxs = torch.topk(d, k=k_eff, largest=False, dim=1)  # (q, k)

            # Base weights: spatial gaussian on distance
            weights = torch.ones_like(vals)
            if w_spatial_sigma is not None:
                sigma = float(w_spatial_sigma)
                weights = torch.exp(-(vals**2) / (2 * sigma * sigma))  # (q, k)

            # Optional temporal decay
            if w_time is not None:
                weights = weights * w_time[idxs]  # broadcast (q,k)

            # # Optional cosine weight with provided query features (identity stabilization)
            # if use_cosine_weight and query_feats_qxD is not None:
            #     qf = F.normalize(query_feats_qxD[s:e].to(self.device, dtype=torch.float32), dim=1)  # (q, D)
            #     mem_feats = self.feat[idxs]  # (q, k, D)
            #     cos = torch.einsum("qd,qkd->qk", qf, mem_feats)  # (q, k)
            #     cos = cos.clamp(min=0.0) ** 1.0  # ignore opposite directions; optional sharpen
            #     weights = weights * (cos + 1e-6)

            # Normalize weights and blend
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # (q, k)
            blended = torch.einsum("qk,qkd->qd", weights, self.feat[idxs])  # (q, D)
            out[s:e] = F.normalize(blended, dim=1)

        return out
