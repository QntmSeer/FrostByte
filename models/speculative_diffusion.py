import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeculativeDiffusionModel(nn.Module):
    """
    Speculative Speculative Decoding (SSD) wrapper for diffusion models.
    Inspired by Tanishq Kumar et al. (ICLR 2026 / Saguaro).
    Uses concurrent PyTorch CUDA streams to execute target verification and
    future draft pre-speculation in parallel, hiding the drafting latency.
    """
    def __init__(self, target_diffusion, draft_diffusion):
        """
        Args:
            target_diffusion: An instance of DiffusionModel (large, slow).
            draft_diffusion: An instance of DiffusionModel (small, fast).
        """
        super().__init__()
        self.target_diffusion = target_diffusion
        self.draft_diffusion = draft_diffusion
        
        assert len(self.target_diffusion.betas) == len(self.draft_diffusion.betas), \
            "Target and Draft models must have the same number of timesteps."
            
        self.timesteps = self.target_diffusion.timesteps
        self.register_buffer('betas', self.target_diffusion.betas)
        self.register_buffer('alphas_cumprod', self.target_diffusion.alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', self.target_diffusion.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', self.target_diffusion.sqrt_one_minus_alphas_cumprod)
        
        # Asynchronous execution streams (initialized lazily based on device)
        self.target_stream = None
        self.draft_stream = None

    def _init_streams(self, device):
        if device.type == 'cuda' and self.target_stream is None:
            self.target_stream = torch.cuda.Stream(device=device)
            self.draft_stream = torch.cuda.Stream(device=device)

    def _get_transition_params(self, x, eps, i):
        beta_t = self.betas[i]
        alpha_bar_t = self.alphas_cumprod[i]
        alpha_t = 1.0 - beta_t
        
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        
        mean = coeff1 * (x - coeff2 * eps)
        sigma = torch.sqrt(beta_t)
        return mean, sigma

    def _sequential_draft(self, x_start, start_t, K, device):
        """
        Helper to draft K steps sequentially.
        """
        x_draft = x_start.clone()
        trajectory = []
        for step_idx in range(K):
            t_val = start_t - step_idx
            t = torch.full((x_start.shape[0],), t_val, device=device, dtype=torch.long)
            
            eps_p = self.draft_diffusion.model(x_draft, t)
            if isinstance(eps_p, list):
                eps_p = torch.cat(eps_p, dim=1)
                
            mean_p, sigma = self._get_transition_params(x_draft, eps_p, t_val)
            
            if t_val > 0:
                x_next_draft = mean_p + sigma * torch.randn_like(x_draft)
            else:
                x_next_draft = mean_p
            x_next_draft.clamp_(-6.0, 6.0)
            
            trajectory.append({
                'x_curr': x_draft.clone(),
                'x_next': x_next_draft.clone(),
                'eps_p': eps_p.clone(),
                'mean_p': mean_p.clone(),
                'sigma': sigma
            })
            x_draft = x_next_draft
        return trajectory

    @torch.no_grad()
    def sample_speculative(self, shape, K=3, device='cpu', async_mode=True):
        """
        Runs Asynchronous Speculative Speculative Decoding (SSD).
        
        Args:
            shape: Tensor shape to generate.
            K: Lookahead window size.
            device: Execution device.
            async_mode: If True, uses dual CUDA streams to parallelize verification
                        and pre-speculation. If False, runs sequentially.
        """
        device = torch.device(device)
        self._init_streams(device)
        
        B = shape[0]
        x = torch.randn(shape, device=device)
        
        total_steps = self.timesteps
        i = total_steps - 1
        
        # Initialize first draft trajectory sequentially
        active_trajectory = self._sequential_draft(x, i, min(K, i + 1), device)
        
        # Stats tracking
        cache_hits = 0
        cache_misses = 0
        total_steps_evaluated = 0
        
        while i >= 0:
            curr_K = len(active_trajectory)
            
            if curr_K <= 1:
                # Fallback to standard step-by-step
                t = torch.full((B,), i, device=device, dtype=torch.long)
                eps_q = self.target_diffusion.model(x, t)
                if isinstance(eps_q, list):
                    eps_q = torch.cat(eps_q, dim=1)
                mean_q, sigma = self._get_transition_params(x, eps_q, i)
                x = mean_q + sigma * torch.randn_like(x) if i > 0 else mean_q
                x.clamp_(-6.0, 6.0)
                i -= 1
                if i >= 0:
                    active_trajectory = self._sequential_draft(x, i, min(K, i + 1), device)
                continue

            # --- 1. ASYNCHRONOUS DRAFT & VERIFY (SSD core) ---
            # Define target inputs
            stacked_x = torch.cat([traj['x_curr'] for traj in active_trajectory], dim=0)
            stacked_t = torch.cat([
                torch.full((B,), i - step_idx, device=device, dtype=torch.long)
                for step_idx in range(curr_K)
            ], dim=0)
            
            pre_spec_trajectory = None
            next_start_t = i - curr_K
            
            # Use PyTorch CUDA streams for asynchronous concurrency on GPU
            if device.type == 'cuda' and async_mode:
                # Stream A: Target verification (heavy)
                with torch.cuda.stream(self.target_stream):
                    stacked_eps_q = self.target_diffusion.model(stacked_x, stacked_t)
                    if isinstance(stacked_eps_q, list):
                        stacked_eps_q = torch.cat(stacked_eps_q, dim=1)
                
                # Stream B: Pre-speculate next trajectory assuming active one is accepted (hiding draft latency)
                if next_start_t >= 0:
                    with torch.cuda.stream(self.draft_stream):
                        x_last_draft = active_trajectory[-1]['x_next']
                        pre_spec_K = min(K, next_start_t + 1)
                        pre_spec_trajectory = self._sequential_draft(x_last_draft, next_start_t, pre_spec_K, device)
                
                # Synchronize streams with the main thread
                torch.cuda.current_stream().wait_stream(self.target_stream)
                torch.cuda.current_stream().wait_stream(self.draft_stream)
            else:
                # Synchronous / CPU fallback
                stacked_eps_q = self.target_diffusion.model(stacked_x, stacked_t)
                if isinstance(stacked_eps_q, list):
                    stacked_eps_q = torch.cat(stacked_eps_q, dim=1)
                if next_start_t >= 0:
                    x_last_draft = active_trajectory[-1]['x_next']
                    pre_spec_K = min(K, next_start_t + 1)
                    pre_spec_trajectory = self._sequential_draft(x_last_draft, next_start_t, pre_spec_K, device)
            
            # --- 2. REJECTION & RESAMPLING ---
            eps_q_list = torch.chunk(stacked_eps_q, curr_K, dim=0)
            rejected = False
            accepted_idx = -1
            
            for step_idx in range(curr_K):
                t_val = i - step_idx
                traj = active_trajectory[step_idx]
                eps_q = eps_q_list[step_idx]
                
                mean_q, sigma = self._get_transition_params(traj['x_curr'], eps_q, t_val)
                x_next = traj['x_next']
                mean_p = traj['mean_p']
                
                diff_q = ((x_next - mean_q) ** 2).view(B, -1).sum(dim=-1)
                diff_p = ((x_next - mean_p) ** 2).view(B, -1).sum(dim=-1)
                denom = max(sigma ** 2, 1e-12)
                
                log_ratio = -0.5 * (diff_q - diff_p) / denom
                alpha = torch.exp(log_ratio).clamp(max=1.0)
                
                u = torch.rand(B, device=device)
                step_accepted = torch.all(u < alpha).item()
                
                if step_accepted:
                    x = x_next
                    accepted_idx = step_idx
                else:
                    # Step rejected! Sample corrected x_next from residual
                    corrected_x = torch.zeros_like(x)
                    for b_idx in range(B):
                        sample_corrected = False
                        for _ in range(50):
                            z_cand = torch.randn_like(x[b_idx])
                            cand = mean_q[b_idx] + sigma * z_cand
                            c_diff_q = ((cand - mean_q[b_idx]) ** 2).sum()
                            c_diff_p = ((cand - mean_p[b_idx]) ** 2).sum()
                            c_log_ratio = -0.5 * (c_diff_p - c_diff_q) / denom
                            c_alpha = torch.exp(c_log_ratio).clamp(max=1.0)
                            
                            u_cand = torch.rand(1, device=device).item()
                            if u_cand > c_alpha.item():
                                corrected_x[b_idx] = cand
                                sample_corrected = True
                                break
                                
                        if not sample_corrected:
                            corrected_x[b_idx] = mean_q[b_idx] + sigma * torch.randn_like(x[b_idx])
                            
                    x = corrected_x
                    x.clamp_(-6.0, 6.0)
                    rejected = True
                    break
            
            # --- 3. CACHE HIT/MISS & Active Trajectory update ---
            total_steps_evaluated += curr_K
            if not rejected:
                # Cache Hit! Entire drafted trajectory was accepted.
                cache_hits += 1
                i -= curr_K
                if pre_spec_trajectory is not None:
                    # Promote pre-speculate cache to active trajectory (0 drafting latency!)
                    active_trajectory = pre_spec_trajectory
                else:
                    active_trajectory = []
            else:
                # Cache Miss! Part of the trajectory was rejected.
                cache_misses += 1
                i -= (accepted_idx + 2)
                # Discard pre-speculate cache, draft a new trajectory from corrected state sequentially
                if i >= 0:
                    active_trajectory = self._sequential_draft(x, i, min(K, i + 1), device)
                else:
                    active_trajectory = []
                    
        stats = {
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': (cache_hits / (cache_hits + cache_misses)) if (cache_hits + cache_misses) > 0 else 0,
            'total_speculations': total_steps_evaluated
        }
        return x, stats
