import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from jax import jit, vmap
from matplotlib.colors import LogNorm

jax.config.update("jax_enable_x64", True)


def _single_qubit_x_rotation(psi, qubit, angle, n_qubits):
    shape = (2,) * n_qubits
    psi = psi.reshape(shape)

    cos_a = jnp.cos(angle / 2).astype(jnp.complex128)
    sin_a = jnp.sin(angle / 2).astype(jnp.complex128)

    # Move target qubit to axis 0 for easy slicing
    psi = jnp.moveaxis(psi, qubit, 0)

    psi0 = psi[0]  # amplitude slice where qubit=|0>
    psi1 = psi[1]  # amplitude slice where qubit=|1>

    new0 = cos_a * psi0 - 1j * sin_a * psi1
    new1 = -1j * sin_a * psi0 + cos_a * psi1

    psi = jnp.stack([new0, new1], axis=0)
    psi = jnp.moveaxis(psi, 0, qubit)
    return psi.reshape(-1)


def _two_qubit_zz_rotation(psi, q0, q1, angle, n_qubits):
    dim = 2**n_qubits
    indices = jnp.arange(dim)

    bit_q0 = (indices >> (n_qubits - 1 - q0)) & 1
    bit_q1 = (indices >> (n_qubits - 1 - q1)) & 1

    parity = bit_q0 ^ bit_q1  # 0 or 1
    phase = jnp.where(parity == 0, jnp.exp(1j * angle), jnp.exp(-1j * angle))

    return psi * phase.astype(jnp.complex128)


def _trotter_step(psi, J1_dt_half, J2_dt_half, h_dt, n_qubits, pbc):
    for i in range(n_qubits - 1):
        psi = _two_qubit_zz_rotation(psi, i, i + 1, J1_dt_half, n_qubits)
    if pbc and n_qubits > 2:
        psi = _two_qubit_zz_rotation(psi, n_qubits - 1, 0, J1_dt_half, n_qubits)

    for i in range(n_qubits - 2):
        psi = _two_qubit_zz_rotation(psi, i, i + 2, -J2_dt_half, n_qubits)
    if pbc and n_qubits > 3:
        psi = _two_qubit_zz_rotation(psi, n_qubits - 2, 0, -J2_dt_half, n_qubits)
        psi = _two_qubit_zz_rotation(psi, n_qubits - 1, 1, -J2_dt_half, n_qubits)

    for i in range(n_qubits):
        psi = _single_qubit_x_rotation(psi, i, 2.0 * h_dt, n_qubits)

    for i in range(n_qubits - 2):
        psi = _two_qubit_zz_rotation(psi, i, i + 2, -J2_dt_half, n_qubits)
    if pbc and n_qubits > 3:
        psi = _two_qubit_zz_rotation(psi, n_qubits - 2, 0, -J2_dt_half, n_qubits)
        psi = _two_qubit_zz_rotation(psi, n_qubits - 1, 1, -J2_dt_half, n_qubits)

    # ── NN ZZ half-step (reverse) ──────────────────────────────────────────
    for i in range(n_qubits - 1):
        psi = _two_qubit_zz_rotation(psi, i, i + 1, J1_dt_half, n_qubits)
    if pbc and n_qubits > 2:
        psi = _two_qubit_zz_rotation(psi, n_qubits - 1, 0, J1_dt_half, n_qubits)

    return psi


def trotter_step_jax(psi, J2, h, dt, n_qubits, pbc=True):
    nn_angle = dt / 2.0
    nnn_angle = J2 * dt / 2.0
    x_angle = -2.0 * h * dt

    def nn_layer(psi, angle):
        for i in range(n_qubits - 1):
            psi = _two_qubit_zz_rotation(psi, i, i + 1, angle, n_qubits)
        if pbc and n_qubits > 2:
            psi = _two_qubit_zz_rotation(psi, n_qubits - 1, 0, angle, n_qubits)
        return psi

    def nnn_layer(psi, angle):
        for i in range(n_qubits - 2):
            psi = _two_qubit_zz_rotation(psi, i, i + 2, angle, n_qubits)
        if pbc and n_qubits > 3:
            psi = _two_qubit_zz_rotation(psi, n_qubits - 2, 0, angle, n_qubits)
            psi = _two_qubit_zz_rotation(psi, n_qubits - 1, 1, angle, n_qubits)
        return psi

    def x_layer(psi, angle):
        for i in range(n_qubits):
            psi = _single_qubit_x_rotation(psi, i, angle, n_qubits)
        return psi

    psi = nn_layer(psi, +nn_angle)  # NN forward half
    psi = nnn_layer(psi, -nnn_angle)  # NNN forward half  (+ in H → - in exp)
    psi = x_layer(psi, x_angle)  # transverse full step
    psi = nnn_layer(psi, -nnn_angle)  # NNN backward half
    psi = nn_layer(psi, +nn_angle)  # NN backward half
    return psi


def build_initial_state(n_qubits, state_type="ferro"):

    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=np.complex128)

    if state_type == "ferro":
        # |↑↑...↑> = |00...0> in Z convention
        psi[0] = 1.0

    elif state_type == "antiferro":
        # |↑↓↑↓...> = |010101...>
        idx = sum(1 << (n_qubits - 1 - i) for i in range(1, n_qubits, 2))
        psi[idx] = 1.0

    elif state_type == "antiphase":
        # |↑↑↓↓↑↑↓↓...> — period-4 pattern
        idx = sum(1 << (n_qubits - 1 - i) for i in range(n_qubits) if (i // 2) % 2 == 1)
        psi[idx] = 1.0

    elif state_type == "plus":
        # |+>^N  — uniform superposition
        psi[:] = 1.0 / np.sqrt(dim)

    else:
        raise ValueError(f"Unknown state_type: {state_type!r}")

    return jnp.array(psi)


def make_echo_fn(n_qubits, dt, n_steps_max, psi0, pbc=True):
    # We sample the echo at these step counts
    sample_steps = np.unique(
        np.round(np.linspace(max(1, n_steps_max // 8), n_steps_max, 8)).astype(int)
    )
    sample_set = set(sample_steps.tolist())
    n_samples = len(sample_steps)

    sample_steps_jax = jnp.array(sample_steps, dtype=jnp.int32)

    def echo_fn(kappa, h):
        J2 = kappa

        def scan_body(carry, _):
            psi, step_idx, echoes = carry

            psi_new = trotter_step_jax(psi, J2, h, dt, n_qubits, pbc)

            # Compute echo at this step
            echo = jnp.abs(jnp.dot(psi0.conj(), psi_new)) ** 2

            # Conditionally store: if step_idx+1 is in sample_steps
            # We use a mask: compare against all sample_steps
            step_num = step_idx + 1  # 1-indexed
            mask = sample_steps_jax == step_num  # shape (n_samples,)
            echoes = echoes + mask * echo  # broadcast store

            return (psi_new, step_idx + 1, echoes), None

        init_carry = (psi0, jnp.int32(0), jnp.zeros(n_samples, dtype=jnp.float64))
        (psi_final, _, echoes), _ = lax.scan(
            scan_body, init_carry, None, length=n_steps_max
        )
        return echoes  # shape (n_samples,)

    return jit(echo_fn), sample_steps


def compute_phase_diagram(
    n_qubits=8,
    kappa_vals=None,
    h_vals=None,
    t_total=6.0,
    dt=0.15,
    pbc=True,
    init_state="ferro",
):

    n_steps_max = max(1, int(t_total / dt))
    nk, nh = len(kappa_vals), len(h_vals)

    print(f"\n{'='*60}")
    print(f" ANNNI Phase Diagram  |  N={n_qubits}  dt={dt}  t_max={t_total}")
    print(f" Grid: {nk} × {nh} = {nk*nh} points")
    print(f" Trotter steps per point: {n_steps_max}")
    print(f" Init state: {init_state}    PBC: {pbc}")
    print(f"{'='*60}\n")

    # Build reference state once
    psi0 = build_initial_state(n_qubits, init_state)

    # Build JIT-compiled echo function
    echo_fn, sample_steps = make_echo_fn(n_qubits, dt, n_steps_max, psi0, pbc)

    batched_over_h = vmap(echo_fn, in_axes=(None, 0))

    # Output arrays
    rate_avg = np.zeros((nk, nh))
    rate_max = np.zeros((nk, nh))
    echo_final = np.zeros((nk, nh))

    # Warm up JIT on first kappa value
    print("JIT compiling... (first kappa row may take a few seconds)")
    t0 = time.perf_counter()

    h_jax = jnp.array(h_vals)

    for ki, kappa in enumerate(kappa_vals):
        t_row = time.perf_counter()

        # echoes_row: (nh, n_samples)
        echoes_row = np.array(batched_over_h(float(kappa), h_jax))

        # Rate function: -(1/N) * log(echo)
        rates_row = -np.log(echoes_row + 1e-10) / n_qubits  # (nh, n_samples)

        rate_avg[ki] = rates_row.mean(axis=1)
        rate_max[ki] = rates_row.max(axis=1)
        echo_final[ki] = echoes_row[:, -1]  # last sample time ≈ t_total

        elapsed = time.perf_counter() - t_row
        total_elapsed = time.perf_counter() - t0
        eta = total_elapsed / (ki + 1) * (nk - ki - 1)
        print(
            f"  [{ki+1:3d}/{nk}]  κ={kappa:.3f}  "
            f"rate_avg∈[{rate_avg[ki].min():.3f},{rate_avg[ki].max():.3f}]  "
            f"row={elapsed:.2f}s  ETA={eta:.1f}s"
        )

    print(f"\nTotal scan time: {time.perf_counter() - t0:.1f}s")
    return rate_avg, rate_max, echo_final, kappa_vals, h_vals


def plot_phase_diagram(kappa_vals, h_vals, rate_avg, rate_max, echo_final, n_qubits):
    fig = plt.figure(figsize=(17, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    K, H = np.meshgrid(kappa_vals, h_vals, indexing="ij")

    panels = [
        (rate_avg, r"Time-averaged $\langle\lambda(t)\rangle$", "inferno", "log"),
        (rate_max, r"Peak $\max_t\lambda(t)$  [DQPT signal]", "inferno", "log"),
        # (echo_final, r"Final-time echo $\mathcal{L}(t_{\rm max})$","inferno_r", "lin"),
    ]

    for idx, (data, title, cmap, scale) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])

        if scale == "log":
            vmin = max(float(data.min()), 1e-3)
            im = ax.pcolormesh(
                K,
                H,
                data,
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=float(data.max())),
                shading="auto",
            )
        else:
            im = ax.pcolormesh(
                K, H, data, cmap=cmap, vmin=0.0, vmax=1.0, shading="auto"
            )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$\kappa$", fontsize=12)
        ax.set_ylabel(r"$h$", fontsize=12)
        ax.set_title(title, fontsize=11, pad=9)

        # ── Phase boundary guides ──────────────────────────────────────────
        k_fm = kappa_vals[kappa_vals <= 0.5]
        k_ap = kappa_vals[kappa_vals >= 0.5]
        ax.plot(
            k_fm,
            np.clip(1.0 - 2 * k_fm, 0, None),
            "w--",
            lw=1.3,
            alpha=0.75,
            label=r"FM$\to$PM",
        )
        ax.plot(k_ap, (k_ap - 0.5), "w:", lw=1.3, alpha=0.75, label=r"AP$\to$PM")

        # ── Phase labels ───────────────────────────────────────────────────
        kw = dict(
            color="white",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
            ha="center",
        )
        ax.text(0.17, 0.20, "FM", **kw)
        ax.text(0.78, 0.10, "AP", **kw)
        ax.text(0.50, 0.83, "PM", **kw)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.4)

    fig.suptitle(
        f"ANNNI Phase Diagram — Loschmidt Echo  "
        f"(N={n_qubits}, 2nd-order Trotter, JAX+vmap)",
        fontsize=13,
        y=1.02,
    )
    plt.savefig("annni_phase_diagram.png", dpi=150, bbox_inches="tight")
    print("Saved annni_phase_diagram.png")
    plt.close()


N_QUBITS = 16  # increase to 10–12 for sharper phase boundaries
DT = 0.4  # Trotter step  (2nd-order: error ~ dt^3 per step)
T = 5.0  # total evolution time
INIT_STATE = "plus"

# Grid resolution — 30×30 is fast; 50×50 gives publication quality
KAPPA_VALS = np.linspace(0.0, 1, 30)
H_VALS = np.linspace(0.0, 2, 30)

# ── Phase diagram ───────────────────────────────────────────────────────────
rate_avg, rate_max, echo_final, kv, hv = compute_phase_diagram(
    n_qubits=N_QUBITS,
    kappa_vals=KAPPA_VALS,
    h_vals=H_VALS,
    t_total=T,
    dt=DT,
    init_state=INIT_STATE,
)

plot_phase_diagram(kv, hv, rate_avg, rate_max, echo_final, N_QUBITS)

print("\nAll done.")
