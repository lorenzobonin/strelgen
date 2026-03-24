# Necessary since some predicted trajectories end up with unrealistic values

def smooth_stop_poly(
    traj: torch.Tensor,
    max_step: float = 10.0,
    default_horizon: int = 5,
    exponent: float = 2.0,
    agent_types: list | torch.Tensor | None = None,
    horizon_map: dict | None = None,
    max_velocity: float | None = None,
):
    """
    Polynomial slowdown after detecting jumps.

    Args:
        traj: (N, T, 2) tensor (world coords)
        max_step: threshold for detecting a jump between consecutive steps
        default_horizon: number of steps over which to decelerate
        exponent: polynomial exponent (2.0 => quadratic easing)
        agent_types: optional (N,) array of agent type keys (used with horizon_map)
        horizon_map: optional dict mapping agent_type -> horizon
        max_velocity: optional clamp on the magnitude of the "last valid" velocity

    Returns:
        corrected: (N, T, 2) tensor with smoothed stops
    """
    assert traj.ndim == 3 and traj.shape[2] == 2
    device = traj.device
    corrected = traj.clone()
    N, T, _ = corrected.shape
    if T <= 1:
        return corrected

    # compute distances between consecutive steps
    diffs = corrected[:, 1:] - corrected[:, :-1]      # (N, T-1, 2)
    dists = torch.norm(diffs, dim=-1)                 # (N, T-1)
    jumps = dists > max_step                          # (N, T-1)
    has_jump = jumps.any(dim=1)                       # (N,)

    if not has_jump.any():
        return corrected

    # first jump index per trajectory (index of transition: between k and k+1)
    first_k = torch.argmax(jumps.int(), dim=1)       # returns 0 if none - we will mask
    traj_idxs = torch.nonzero(has_jump).squeeze(1)    # only trajectories that actually have jumps

    for n in traj_idxs:
        n = int(n.item())
        # ensure the argmax corresponds to a true jump (argmax returns 0 even if no True)
        k = int(first_k[n].item())
        if not jumps[n, k]:
            # fallback to explicit search (shouldn't happen often)
            true_inds = jumps[n].nonzero(as_tuple=True)[0]
            if true_inds.numel() == 0:
                continue
            k = int(true_inds[0].item())

        t0 = k + 1                 # index of the first bad position (p_{k+1})
        last_valid_idx = t0 - 1    # index of last valid position (p_k)

        p0 = corrected[n, last_valid_idx].clone()
        # compute "last valid" velocity (if available)
        if last_valid_idx > 0:
            v0 = corrected[n, last_valid_idx] - corrected[n, last_valid_idx - 1]
        else:
            v0 = torch.zeros_like(p0)

        # optional clamp on v0 magnitude
        if (max_velocity is not None) and (torch.norm(v0) > 0):
            v_norm = torch.norm(v0)
            if v_norm > max_velocity:
                v0 = v0 * (max_velocity / (v_norm + 1e-8))

        # choose horizon (possibly agent-dependent)
        h = default_horizon
        if (agent_types is not None) and (horizon_map is not None):
            key = agent_types[n]
            # support tensors and strings: convert to python if needed
            try:
                key = key.item()
            except Exception:
                pass
            h = horizon_map.get(key, default_horizon)

        # apply polynomial easing for indices last_valid_idx + 1 .. last_valid_idx + h
        for i in range(1, h + 1):
            idx = last_valid_idx + i
            if idx >= T:
                break
            factor = 1.0 - (i / float(h)) ** float(exponent)
            corrected[n, idx] = p0 + factor * v0

        # freeze position after the horizon (so original future bad points are overwritten)
        last_idx = min(T - 1, last_valid_idx + h)
        if last_idx + 1 < T:
            # broadcast the frozen position into the tail
            frozen = corrected[n, last_idx].unsqueeze(0).expand(T - last_idx - 1, 2)
            corrected[n, last_idx + 1 :] = frozen

    return corrected

def smooth_stop_poly_batched(
    traj: torch.Tensor,
    max_step: float = 10.0,
    default_horizon: int = 5,
    exponent: float = 2.0,
    agent_types: list | torch.Tensor | None = None,
    horizon_map: dict | None = None,
    max_velocity: float | None = None,
):
    """
    Supports:
        (N, T, 2)
        (N, S, T, 2)
    """

    # ==========================================================
    # --- CASE 1: MULTI-SAMPLE INPUT → recurse per sample ---
    # ==========================================================
    if traj.ndim == 4:
        N, S, T, C = traj.shape
        assert C == 2

        corrected = traj.clone()

        for s in range(S):
            corrected[:, s] = smooth_stop_poly(
                corrected[:, s],
                max_step=max_step,
                default_horizon=default_horizon,
                exponent=exponent,
                agent_types=agent_types,
                horizon_map=horizon_map,
                max_velocity=max_velocity,
            )

        return corrected

    # ==========================================================
    # --- CASE 2: ORIGINAL SINGLE-SAMPLE LOGIC ---
    # ==========================================================
    assert traj.ndim == 3 and traj.shape[2] == 2

    corrected = traj.clone()
    N, T, _ = corrected.shape

    if T <= 1:
        return corrected

    # compute distances between consecutive steps
    diffs = corrected[:, 1:] - corrected[:, :-1]      # (N, T-1, 2)
    dists = torch.norm(diffs, dim=-1)                 # (N, T-1)
    jumps = dists > max_step                          # (N, T-1)
    has_jump = jumps.any(dim=1)                       # (N,)

    if not has_jump.any():
        return corrected

    # first jump index per trajectory
    first_k = torch.argmax(jumps.int(), dim=1)
    traj_idxs = torch.nonzero(has_jump).squeeze(1)

    for n in traj_idxs:
        n = int(n.item())

        k = int(first_k[n].item())
        if not jumps[n, k]:
            true_inds = jumps[n].nonzero(as_tuple=True)[0]
            if true_inds.numel() == 0:
                continue
            k = int(true_inds[0].item())

        t0 = k + 1
        last_valid_idx = t0 - 1

        p0 = corrected[n, last_valid_idx].clone()

        if last_valid_idx > 0:
            v0 = corrected[n, last_valid_idx] - corrected[n, last_valid_idx - 1]
        else:
            v0 = torch.zeros_like(p0)

        # optional velocity clamp
        if (max_velocity is not None) and (torch.norm(v0) > 0):
            v_norm = torch.norm(v0)
            if v_norm > max_velocity:
                v0 = v0 * (max_velocity / (v_norm + 1e-8))

        # choose horizon
        h = default_horizon
        if (agent_types is not None) and (horizon_map is not None):
            key = agent_types[n]
            try:
                key = key.item()
            except Exception:
                pass
            h = horizon_map.get(key, default_horizon)

        # polynomial easing
        for i in range(1, h + 1):
            idx = last_valid_idx + i
            if idx >= T:
                break
            factor = 1.0 - (i / float(h)) ** float(exponent)
            corrected[n, idx] = p0 + factor * v0

        # freeze tail
        last_idx = min(T - 1, last_valid_idx + h)
        if last_idx + 1 < T:
            frozen = corrected[n, last_idx].unsqueeze(0).expand(T - last_idx - 1, 2)
            corrected[n, last_idx + 1:] = frozen

    return corrected


# filter invalid fixed agents and return cleaned trajectories + mask of valid agents
def clean_and_filter_agents(full_world):
    """
    full_world: [N, T, 2]

    returns:
      world_valid: [N_valid, T, 2]   (only valid agents, cleaned)
      agent_mask:  [N]               (True = kept agent)
    """
    N, T, _ = full_world.shape
    device = full_world.device

    # 1) timestep validity mask
    valid = (full_world.abs().sum(-1) != 0)  # [N, T]

    # 2) agent-level validity
    agent_mask = valid.any(dim=1)             # [N]

    # 3) remove fully invalid agents
    world = full_world[agent_mask]             # [N_valid, T, 2]
    valid = valid[agent_mask]                  # [N_valid, T]

    # early exit
    if world.numel() == 0:
        return world, agent_mask

    Nv = world.shape[0]
    time = torch.arange(T, device=device)

    # 4) forward fill indices
    last_valid = torch.where(
        valid,
        time.unsqueeze(0),
        torch.full((Nv, T), -1, device=device)
    )
    last_valid = torch.cummax(last_valid, dim=1).values

    # 5) backward fill indices
    next_valid = torch.where(
        valid,
        time.unsqueeze(0),
        torch.full((Nv, T), T, device=device)
    )
    next_valid = torch.cummin(next_valid.flip(1), dim=1).values.flip(1)

    # 6) choose valid index per timestep
    idx = torch.where(last_valid >= 0, last_valid, next_valid)
    idx = idx.clamp(0, T - 1)

    # 7) gather filled trajectories
    idx = idx.unsqueeze(-1).expand(-1, -1, 2)
    world_valid = torch.gather(world, dim=1, index=idx)

    return world_valid, agent_mask

def clean_and_filter_agents_batched(full_world: torch.Tensor):
    """
    Clean invalid timesteps and remove fully invalid agents.

    Supports:
        [N, T, 2]       (single sample)
        [N, S, T, 2]    (multi-sample)

    Returns:
        world_valid:
            [N_valid, T, 2]        if single-sample
            [N_valid, S, T, 2]     if multi-sample
        agent_mask: [N]  (True = kept agent)
    """

    # ==========================================================
    # -------- MULTI-SAMPLE CASE  [N, S, T, 2]
    # ==========================================================
    if full_world.ndim == 4:

        N, S, T, C = full_world.shape
        assert C == 2

        device = full_world.device

        # ------------------------------------------------------
        # timestep validity per sample
        # ------------------------------------------------------
        valid = (full_world.abs().sum(-1) != 0)        # [N, S, T]

        # ------------------------------------------------------
        # agent valid if ANY timestep in ANY sample is valid
        # (HPC-safe: no tuple dim)
        # ------------------------------------------------------
        agent_mask = valid.any(dim=2).any(dim=1)       # [N]

        # ------------------------------------------------------
        # filter agents
        # ------------------------------------------------------
        world = full_world[agent_mask]                 # [N_valid, S, T, 2]
        valid = valid[agent_mask]                      # [N_valid, S, T]

        if world.numel() == 0:
            return world, agent_mask

        Nv = world.shape[0]
        time = torch.arange(T, device=device)

        # ------------------------------------------------------
        # forward fill indices
        # ------------------------------------------------------
        last_valid = torch.where(
            valid,
            time.view(1, 1, T),
            torch.full((Nv, S, T), -1, device=device)
        )
        last_valid = torch.cummax(last_valid, dim=2).values  # along time

        # ------------------------------------------------------
        # backward fill indices
        # ------------------------------------------------------
        next_valid = torch.where(
            valid,
            time.view(1, 1, T),
            torch.full((Nv, S, T), T, device=device)
        )
        next_valid = torch.cummin(next_valid.flip(2), dim=2).values.flip(2)

        # ------------------------------------------------------
        # choose valid index
        # ------------------------------------------------------
        idx = torch.where(last_valid >= 0, last_valid, next_valid)
        idx = idx.clamp(0, T - 1)

        # ------------------------------------------------------
        # gather cleaned trajectories
        # ------------------------------------------------------
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, 2)      # [Nv, S, T, 2]
        world_valid = torch.gather(world, dim=2, index=idx)

        return world_valid, agent_mask

    # ==========================================================
    # -------- SINGLE-SAMPLE CASE  [N, T, 2]
    # ==========================================================
    elif full_world.ndim == 3:

        N, T, C = full_world.shape
        assert C == 2

        device = full_world.device

        # timestep validity
        valid = (full_world.abs().sum(-1) != 0)        # [N, T]

        # agent validity
        agent_mask = valid.any(dim=1)                  # [N]

        # filter
        world = full_world[agent_mask]                 # [N_valid, T, 2]
        valid = valid[agent_mask]                      # [N_valid, T]

        if world.numel() == 0:
            return world, agent_mask

        Nv = world.shape[0]
        time = torch.arange(T, device=device)

        # forward fill
        last_valid = torch.where(
            valid,
            time.unsqueeze(0),
            torch.full((Nv, T), -1, device=device)
        )
        last_valid = torch.cummax(last_valid, dim=1).values

        # backward fill
        next_valid = torch.where(
            valid,
            time.unsqueeze(0),
            torch.full((Nv, T), T, device=device)
        )
        next_valid = torch.cummin(next_valid.flip(1), dim=1).values.flip(1)

        # choose index
        idx = torch.where(last_valid >= 0, last_valid, next_valid)
        idx = idx.clamp(0, T - 1)

        # gather
        idx = idx.unsqueeze(-1).expand(-1, -1, 2)
        world_valid = torch.gather(world, dim=1, index=idx)

        return world_valid, agent_mask

    # ==========================================================
    # -------- INVALID INPUT
    # ==========================================================
    else:
        raise ValueError(
            f"clean_and_filter_agents_batched expects shape "
            f"[N,T,2] or [N,S,T,2], got {full_world.shape}"
        )