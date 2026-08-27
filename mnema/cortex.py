import numpy as np

class BennaFusiSynapse:
    """
    4-variable coupled physical chain (m=4) per synapse.
    u_1 is the effective visible weight; deeper variables act as memory reservoirs.
    """
    def __init__(self, in_features: int, out_features: int, m: int = 4, g0: float = 0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.g0 = g0

        # Capacitances: C_k = C_1 * 2^(k-1)
        self.C = np.array([2.0**k for k in range(m)], dtype=np.float32)
        
        # Conductances between beakers: g_{k, k+1} = g0 * 2^(-(2k-1))
        self.g = np.array([g0 * (2.0**(-(2*k - 1))) for k in range(1, m)], dtype=np.float32)

        # Synaptic variables u [m, out_features, in_features] initialized with small weights
        self.u = np.zeros((m, out_features, in_features), dtype=np.float32)
        self.u[0] = np.random.normal(0, 0.05, size=(out_features, in_features)).astype(np.float32)

    @property
    def weight(self) -> np.ndarray:
        return self.u[0]  # Visible weight is u_1

    def update(self, delta_w: np.ndarray, instrument=None):
        """
        Applies plastic update delta_w to u_1 and equilibrates the beaker chain.
        """
        # Inject fast change into u_1
        self.u[0] += delta_w / self.C[0]

        # Coupled diffusion between levels
        for k in range(self.m - 1):
            flow = self.g[k] * (self.u[k+1] - self.u[k])
            self.u[k] += flow / self.C[k]
            self.u[k+1] -= flow / self.C[k+1]

        if instrument is not None:
            instrument.weight_updates += int(np.count_nonzero(delta_w))
            instrument.sram_write_b += self.m * self.in_features * self.out_features * 4


class SlowCortex:
    """
    Shallow Spiking Cortex with Adaptive LIF (ALIF) dynamics & local 3-factor credit assignment.
    """
    def __init__(self, in_features=16384, hidden_dim=256, out_dim=10, tau_m=0.02, tau_adapt=0.2, v_th=1.0, beta=1.8):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Membrane constants
        self.dt = 0.001
        self.alpha = np.exp(-self.dt / tau_m)
        self.rho = np.exp(-self.dt / tau_adapt)
        self.v_th = v_th
        self.beta = beta
        self.rho_e = 0.9
        self.eta = 1e-3

        # Synaptic projections with Benna-Fusi complex synapses
        self.syn_fc = BennaFusiSynapse(in_features, out_dim)

        # Neuron states
        self.v = np.zeros(out_dim, dtype=np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.eligibility = np.zeros((out_dim, in_features), dtype=np.float32)

    def reset_state(self):
        self.v.fill(0.0)
        self.b.fill(0.0)
        self.eligibility.fill(0.0)

    def forward(self, active_indices: np.ndarray, instrument=None) -> np.ndarray:
        """
        Event-driven ALIF forward pass from sparse separator binary codes.
        """
        # Sparse synops: slice only active input columns
        w_active = self.syn_fc.weight[:, active_indices]  # [out_dim, k]
        syn_input = w_active.sum(axis=1)

        # ALIF update equations
        effective_th = self.v_th + self.beta * self.b
        self.v = self.alpha * self.v + syn_input
        
        # Spike generation
        spikes = (self.v >= effective_th).astype(np.float32)
        self.v[spikes > 0] = 0.0  # Reset
        self.b = self.rho * self.b + spikes

        # Update local eligibility traces: e_ij[t] = rho_e * e_ij[t-1] + psi_i * s_j
        # SuperSpike surrogate derivative: psi = 1 / (1 + |v - v_th|)^2
        psi = 1.0 / (1.0 + np.abs(self.v - effective_th))**2
        self.eligibility *= self.rho_e
        self.eligibility[:, active_indices] += psi[:, None]

        if instrument is not None:
            instrument.synops += len(active_indices) * self.out_dim
            instrument.neuron_updates += self.out_dim
            instrument.sram_read_b += len(active_indices) * self.out_dim * 4

        return self.v

    def learn(self, target_label: int, instrument=None):
        """
        Local 3-factor learning rule: Delta w = eta * L_i * e_ij
        """
        target_vec = np.zeros(self.out_dim, dtype=np.float32)
        target_vec[target_label] = 1.0
        
        # Broadcast third factor (error / dopamine signal L_i)
        error_signal = target_vec - self.v
        delta_w = self.eta * (error_signal[:, None] * self.eligibility)

        self.syn_fc.update(delta_w, instrument=instrument)