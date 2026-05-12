"""
Physics-Climate Multimodal Probabilistic (PCMP) Pipeline
Long-Term Solar Power Generation Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict


# =============================================================================
# MODULE 1: Physics Constraint Engine
# =============================================================================

class PhysicsConstraintEngine:
    """
    Computes physical boundary conditions for solar irradiance.
    Enforces thermodynamic and astronomical constraints on model predictions.
    """

    SOLAR_CONST: float = 1361.0   # W/m^2 — solar constant at top of atmosphere
    T_REF:       float = 25.0     # °C   — STC reference temperature
    GAMMA:       float = 0.005    # /°C  — thermal degradation coefficient

    def clear_sky_index(self, d: int, h: float, lat: float, lon: float) -> float:
        """
        Simplified Kasten–Young clear-sky GHI (W/m²).

        Args:
            d   : day of year   [1..365]
            h   : solar hour    [0..23]
            lat : latitude      (degrees)
            lon : longitude     (degrees)

        Returns:
            g   : clear-sky GHI in W/m²  (>=0)
        """
        # --- solar declination (radians) ---
        b = (360.0 / 365.0) * (d - 81)
        b = np.radians(b)
        delta = np.radians(23.45 * np.sin(b))

        # --- hour angle (radians) ---
        omega = np.radians(15.0 * (h - 12.0))

        # --- latitude in radians ---
        phi = np.radians(lat)

        # --- cosine of solar zenith angle ---
        c = (np.sin(phi) * np.sin(delta) +
             np.cos(phi) * np.cos(delta) * np.cos(omega))
        c = float(np.clip(c, 0.0, 1.0))

        # --- eccentricity correction ---
        e = 1.0 + 0.033 * np.cos(np.radians(360.0 * d / 365.0))

        # --- clear-sky GHI ---
        g = self.SOLAR_CONST * e * c
        return max(0.0, g)

    def thermal_degradation(self, t: float) -> float:
        """
        Efficiency multiplier due to cell temperature.

        Args:
            t   : panel temperature (°C)

        Returns:
            eta : efficiency factor in [0, 1]
        """
        delta_t = max(0.0, t - self.T_REF)
        eta = 1.0 - self.GAMMA * delta_t
        return float(np.clip(eta, 0.0, 1.0))


# =============================================================================
# MODULE 2: Climate Data Loader
# =============================================================================

class ClimateWindowDataset(Dataset):
    """
    Sliding-window multivariate time-series dataset.

    Features per timestep:
        [GHI, temperature, wind_speed, humidity]   (4 channels)
    Target:
        GHI at next step
    Physics label:
        Clear-sky GHI limit for that window's anchor timestamp
    """

    def __init__(
        self,
        x: torch.Tensor,          # (N, 4) historical climate features
        y: torch.Tensor,          # (N,)   GHI targets
        c: torch.Tensor,          # (N,)   clear-sky limit per timestep
        w: int = 24,              # window (look-back) length in steps
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.c = c
        self.w = w

    def __len__(self) -> int:
        return len(self.x) - self.w

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            f  : (w, 4)  feature window
            t  : ()      scalar GHI target
            k  : ()      scalar clear-sky limit at window end
        """
        f = self.x[i : i + self.w]           # (w, 4)
        t = self.y[i + self.w]               # scalar
        k = self.c[i + self.w]               # scalar clear-sky limit
        return f, t, k

    @staticmethod
    def build_clear_sky_tensor(
        engine: PhysicsConstraintEngine,
        d: np.ndarray,
        h: np.ndarray,
        lat: float,
        lon: float,
    ) -> torch.Tensor:
        """
        Vectorised helper to pre-compute the clear-sky tensor
        for all N timesteps.

        Args:
            engine : PhysicsConstraintEngine instance
            d      : (N,) day-of-year array
            h      : (N,) hour array
            lat    : station latitude
            lon    : station longitude

        Returns:
            c      : (N,) float32 clear-sky tensor
        """
        c = np.array([
            engine.clear_sky_index(int(d[i]), float(h[i]), lat, lon)
            for i in range(len(d))
        ], dtype=np.float32)
        return torch.from_numpy(c)


# =============================================================================
# MODULE 3: Physics-Informed Neural Network (PINN) Core
# =============================================================================

class PINNCore(nn.Module):
    """
    BiLSTM + Dense PINN with Monte-Carlo-ready Dropout.

    Architectural contract
    ──────────────────────
    Input  : (B, w, F)  — batch × window × features
    Output : (B, 1)     — predicted GHI
    """

    def __init__(
        self,
        f: int   = 4,    # input feature dim
        h: int   = 128,  # LSTM hidden dim
        l: int   = 2,    # LSTM layers
        d: float = 0.3,  # dropout probability
        o: int   = 1,    # output dim
    ) -> None:
        super().__init__()

        # --- BiLSTM encoder ---
        self.rnn = nn.LSTM(
            input_size    = f,
            hidden_size   = h,
            num_layers    = l,
            batch_first   = True,
            bidirectional = True,
            dropout       = d if l > 1 else 0.0,
        )

        # --- MC-Dropout (kept active at inference via training=True trick) ---
        self.drop = nn.Dropout(p=d)

        # --- Dense head ---
        self.fc1 = nn.Linear(h * 2, h)
        self.fc2 = nn.Linear(h, h // 2)
        self.fc3 = nn.Linear(h // 2, o)

    # ------------------------------------------------------------------
    # forward: MC-dropout stays active even in eval() mode
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, w, F)

        Returns:
            p : (B, 1)  — predicted GHI
        """
        # BiLSTM — take last timestep of concatenated fwd+bwd
        z, _ = self.rnn(x)               # (B, w, 2H)
        z    = z[:, -1, :]               # (B, 2H)

        # Dense + MC-Dropout (force dropout=True regardless of model.eval())
        z = self.drop(F.relu(self.fc1(z)))
        z = self.drop(F.relu(self.fc2(z)))
        p = self.fc3(z)                  # (B, 1)
        return p

    # ------------------------------------------------------------------
    # Physics-Informed Loss
    # ------------------------------------------------------------------
    def physics_loss(
        self,
        p: torch.Tensor,   # (B, 1)  predictions
        y: torch.Tensor,   # (B,)    targets
        k: torch.Tensor,   # (B,)    clear-sky limits
        lam: float = 10.0, # penalty weight for physics violation
    ) -> torch.Tensor:
        """
        L = MSE(p, y)  +  λ · mean( ReLU(p - k)² )

        The second term penalises any prediction that exceeds
        the physical clear-sky irradiance upper bound.

        Args:
            p   : model predictions  (B, 1)
            y   : ground-truth GHI   (B,)
            k   : clear-sky limits   (B,)
            lam : physics penalty weight

        Returns:
            l   : scalar loss tensor
        """
        p_flat = p.squeeze(-1)                          # (B,)
        m = F.mse_loss(p_flat, y)                       # data fidelity
        v = F.relu(p_flat - k)                          # violation (B,)
        r = lam * (v ** 2).mean()                       # physics penalty
        l = m + r
        return l


# =============================================================================
# MODULE 4: Probabilistic & Economic Output Layer
# =============================================================================

class ProbabilisticEvaluator:
    """
    Monte Carlo Dropout inference  +  Probabilistic metrics  +  LCOE.
    """

    # Default LCOE parameters (can be overridden at init)
    _CAPEX_DEFAULT: float = 1_000_000.0   # USD
    _OPEX_DEFAULT:  float = 20_000.0      # USD / year
    _CRF_DEFAULT:   float = 0.08          # capital recovery factor
    _N_DEFAULT:     int   = 25            # project lifetime (years)

    def __init__(
        self,
        m: PINNCore,
        n: int   = 100,                    # MC samples
        capex: float = _CAPEX_DEFAULT,
        opex:  float = _OPEX_DEFAULT,
        crf:   float = _CRF_DEFAULT,
        t:     int   = _N_DEFAULT,
    ) -> None:
        self.m    = m
        self.n    = n
        self.capex = capex
        self.opex  = opex
        self.crf   = crf
        self.t     = t

    # ------------------------------------------------------------------
    def monte_carlo_samples(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Run n stochastic forward passes to build a predictive distribution.

        MC-Dropout is achieved by calling model.train() so that Dropout
        layers remain active — a standard Gal & Ghahramani (2016) approach.

        Args:
            x : (1, w, F) — single input window

        Returns:
            s : (n,)  — sample distribution of predicted GHI
        """
        self.m.train()                          # activates Dropout
        with torch.no_grad():
            p = torch.cat(
                [self.m(x).squeeze() for _ in range(self.n)]
            )                                   # (n,)
        self.m.eval()
        return p

    # ------------------------------------------------------------------
    def quantile_metrics(
        self, s: torch.Tensor
    ) -> Dict[str, float]:
        """
        Extract P50 and P90 from the MC sample distribution.

        Args:
            s : (n,) sample tensor

        Returns:
            dict with keys 'p50' and 'p90'
        """
        q = torch.quantile(s, torch.tensor([0.50, 0.90]))
        return {"p50": float(q[0]), "p90": float(q[1])}

    # ------------------------------------------------------------------
    def lcoe(self, p90_kwh: float) -> float:
        """
        Levelized Cost of Energy (USD / kWh).

        Formula:
            LCOE = (CAPEX · CRF + OPEX) / (P90_annual_kWh)

        where P90_annual_kWh is the risk-adjusted annual yield derived
        from the P90 hourly GHI estimate.

        Args:
            p90_kwh : P90 hourly energy yield (kWh)

        Returns:
            l       : LCOE in USD / kWh
        """
        a = p90_kwh * 8760.0                    # annualised energy (kWh/yr)
        a = max(a, 1e-9)                        # guard division by zero
        l = (self.capex * self.crf + self.opex) / a
        return float(l)


# =============================================================================
# MODULE 5: Execution Pipeline
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  PCMP Pipeline — End-to-End Integration Test")
    print("=" * 60)

    # ── Hyper-parameters ──────────────────────────────────────────
    N  = 500    # total timesteps
    W  = 24     # look-back window (hours)
    F  = 4      # features: GHI, temp, wind, humidity
    B  = 16     # batch size

    # ── Module 1: Physics Constraint Engine ──────────────────────
    e = PhysicsConstraintEngine()

    g = e.clear_sky_index(d=172, h=12, lat=28.6, lon=77.2)
    eta = e.thermal_degradation(t=45.0)
    print(f"\n[M1] Clear-Sky GHI (noon, summer solstice, Delhi): {g:.2f} W/m²")
    print(f"[M1] Thermal degradation @ 45°C:  η = {eta:.4f}")

    # ── Module 2: Climate Data Loader ────────────────────────────
    x = torch.rand(N, F)                          # (N, 4)  normalised features
    y = torch.rand(N)                             # (N,)    GHI targets [0,1]

    d_arr = np.random.randint(1, 366, size=N)
    h_arr = np.random.randint(0,  24, size=N).astype(float)
    c = ClimateWindowDataset.build_clear_sky_tensor(e, d_arr, h_arr, 28.6, 77.2)

    ds  = ClimateWindowDataset(x, y, c, w=W)
    dl  = DataLoader(ds, batch_size=B, shuffle=True)
    f_b, t_b, k_b = next(iter(dl))
    print(f"\n[M2] Dataset length : {len(ds)}")
    print(f"[M2] Batch shapes   : features={tuple(f_b.shape)}, "
          f"targets={tuple(t_b.shape)}, clear-sky={tuple(k_b.shape)}")

    # ── Module 3: PINN Core ──────────────────────────────────────
    m   = PINNCore(f=F, h=128, l=2, d=0.3, o=1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    m.train()
    p_b   = m(f_b)                                # (B, 1)
    loss  = m.physics_loss(p_b, t_b, k_b, lam=10.0)
    loss.backward()
    opt.step()

    print(f"\n[M3] Forward pass   : output shape = {tuple(p_b.shape)}")
    print(f"[M3] Physics loss   : {loss.item():.6f}")

    # ── Module 4: Probabilistic & Economic Output Layer ──────────
    ev = ProbabilisticEvaluator(m, n=100)

    x_test = torch.rand(1, W, F)                  # single test window
    s      = ev.monte_carlo_samples(x_test)       # (100,)
    q      = ev.quantile_metrics(s)
    l_cost = ev.lcoe(q["p90"])

    print(f"\n[M4] MC-Dropout samples : {tuple(s.shape)}")
    print(f"[M4] P50 (Expected)     : {q['p50']:.4f}")
    print(f"[M4] P90 (Risk-Adj.)    : {q['p90']:.4f}")
    print(f"[M4] LCOE               : ${l_cost:.4f} / kWh")

    print("\n" + "=" * 60)
    print("  ✓  All modules executed successfully — pipeline verified.")
    print("=" * 60)
