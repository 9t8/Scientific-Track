from itertools import product as iproduct

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from matplotlib.colors import LogNorm

pipeline = qml.CompilePipeline(
    qml.transforms.commute_controlled,
    qml.transforms.cancel_inverses(recursive=True),
    qml.transforms.merge_rotations,
)


def zz_gate(theta, wire_i, wire_j):
    """exp(i·theta·Z_i Z_j)  via  CNOT – RZ(-2θ) – CNOT."""
    qml.CNOT(wires=[wire_i, wire_j])
    qml.DepolarizingChannel(0.05, wires=wire_j)
    qml.RZ(-2.0 * theta, wires=wire_j)
    qml.CNOT(wires=[wire_i, wire_j])
    qml.DepolarizingChannel(0.05, wires=wire_j)


def annni_trotter_step(n_qubits, J1, J2, h, dt, pbc=True):
    half = dt / 2.0

    def nn_layer(angle):
        for i in range(n_qubits - 1):
            zz_gate(angle, i, i + 1)
        if pbc and n_qubits > 2:
            zz_gate(angle, n_qubits - 1, 0)

    def nnn_layer(angle):
        for i in range(n_qubits - 2):
            zz_gate(angle, i, i + 2)
        if pbc and n_qubits > 3:
            zz_gate(angle, n_qubits - 2, 0)
            zz_gate(angle, n_qubits - 1, 1)

    nn_layer(+J1 * half)  # NN forward half
    nnn_layer(-J2 * half)  # NNN forward half
    for i in range(n_qubits):  # full transverse-field step
        qml.RX(-2.0 * h * dt, wires=i)
    nnn_layer(-J2 * half)  # NNN backward half
    nn_layer(+J1 * half)  # NN backward half


# ── Initial state preparation ──────────────────────────────────────────────────


def prepare_initial_state(n_qubits, state_type="ferro"):

    if state_type == "ferro":
        pass  # |0...0> is default

    elif state_type == "antiferro":
        for i in range(1, n_qubits, 2):
            qml.PauliX(wires=i)

    elif state_type == "antiphase":
        for i in range(n_qubits):
            if (i // 2) % 2 == 1:
                qml.PauliX(wires=i)

    elif state_type == "plus":
        for i in range(n_qubits):
            qml.Hadamard(wires=i)


def make_echo_circuit(n_qubits, J1, J2, h, dt, n_steps, init_state, pbc=True):

    dev = qml.device("default.mixed", wires=n_qubits)

    @pipeline
    @qml.qjit
    @qml.qnode(dev, interface="jax")
    def circuit():
        # Prepare reference state
        prepare_initial_state(n_qubits, init_state)
        # Time-evolve forward
        for _ in range(n_steps):
            annni_trotter_step(n_qubits, J1, J2, h, dt, pbc)
        # Measure overlap with initial state via computational basis
        return qml.state()

    return circuit


def loschmidt_echo(n_qubits, J1, J2, h, dt, n_steps, init_state="ferro", pbc=True):

    dev = qml.device("default.mixed", wires=n_qubits)

    @pipeline
    # @qml.qjit
    @qml.qnode(dev, interface="jax")
    def ref_circuit():
        prepare_initial_state(n_qubits, init_state)
        return qml.state()

    @pipeline
    # @qml.qjit
    @qml.qnode(dev, interface="jax")
    def evolved_circuit():
        prepare_initial_state(n_qubits, init_state)
        for _ in range(n_steps):
            annni_trotter_step(n_qubits, J1, J2, h, dt, pbc)
        return qml.state()

    psi0 = ref_circuit()
    psi_t = evolved_circuit()
    overlap = np.abs(np.dot(psi0.conj(), psi_t)) ** 2
    return float(overlap[0][0])


def rate_function(echo, t, n_qubits):

    return -np.log(echo + 1e-10) / n_qubits


def compute_phase_diagram(
    n_qubits=8,
    kappa_vals=None,
    h_vals=None,
    t_total=6.0,
    dt=0.1,
    pbc=True,
    init_state="ferro",
    J1=1.0,
):

    if kappa_vals is None:
        kappa_vals = np.linspace(0.0, 1.2, 25)
    if h_vals is None:
        h_vals = np.linspace(0.0, 2.5, 25)

    n_steps_total = int(t_total / dt)
    # Sample at several time slices and average
    sample_steps = np.unique(
        np.linspace(max(1, n_steps_total // 8), n_steps_total, 6, dtype=int)
    )

    nk, nh = len(kappa_vals), len(h_vals)
    rate_avg = np.zeros((nk, nh))
    rate_max = np.zeros((nk, nh))
    echo_final = np.zeros((nk, nh))

    total = nk * nh
    print(
        f"Scanning {nk}×{nh} = {total} points, {n_qubits} qubits, "
        f"t∈[0,{t_total}], dt={dt}"
    )
    print(f"Trotter steps per point: up to {n_steps_total}\n")

    for ki, kappa in enumerate(kappa_vals):
        J2 = kappa * J1
        for hi, h in enumerate(h_vals):
            rates = []
            for ns in sample_steps:
                echo = loschmidt_echo(n_qubits, J1, J2, h, dt, ns, init_state, pbc)
                lam = rate_function(echo, ns * dt, n_qubits)
                rates.append(lam)
                if ns == sample_steps[-1]:
                    echo_final[ki, hi] = echo

            rate_avg[ki, hi] = np.mean(rates)
            rate_max[ki, hi] = np.max(rates)

        done = (ki + 1) * nh
        print(
            f"  [{done:4d}/{total}]  kappa={kappa:.3f}  "
            f"rate_avg range: [{rate_avg[ki].min():.3f}, {rate_avg[ki].max():.3f}]"
        )

    return rate_avg, rate_max, echo_final


def compute_echo_timeseries(
    n_qubits, J1, kappa, h, t_max=8.0, dt=0.1, init_state="ferro", pbc=True
):
    """Compute L(t) at every Trotter step up to t_max."""
    J2 = kappa * J1
    n_steps = int(t_max / dt)
    times, echoes = [], []

    dev = qml.device("default.mixed", wires=n_qubits)

    @pipeline
    @qml.qjit
    @qml.qnode(dev, interface="jax")
    def ref_circuit():
        prepare_initial_state(n_qubits, init_state)
        return qml.state()

    psi0 = ref_circuit()

    @pipeline
    @qml.qjit
    @qml.qnode(dev, interface="jax")
    def step_circuit(ns):
        prepare_initial_state(n_qubits, init_state)
        for _ in range(ns):
            annni_trotter_step(n_qubits, J1, J2, h, dt, pbc)
        return qml.state()

    for ns in range(1, n_steps + 1):
        psi_t = step_circuit(ns)
        echo = float(np.abs(np.dot(psi0.conj(), psi_t)) ** 2)
        times.append(ns * dt)
        echoes.append(echo)

    return np.array(times), np.array(echoes)


def plot_phase_diagram(kappa_vals, h_vals, rate_avg, rate_max, echo_final, n_qubits):
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    K, H = np.meshgrid(kappa_vals, h_vals, indexing="ij")
    cmap = "inferno"

    titles = [
        (rate_avg, r"Time-averaged rate function $\langle\lambda(t)\rangle$", None),
        (rate_max, r"Max rate function $\max_t\,\lambda(t)$  (DQPT signal)", None),
        (echo_final, r"Final-time echo $\mathcal{L}(t_{\rm max})$", None),
    ]

    axes = []
    for idx, (data, title, norm) in enumerate(titles):
        ax = fig.add_subplot(gs[idx])
        axes.append(ax)

        if idx < 2:
            # rate functions: log scale helps show transitions
            vmin = max(data.min(), 1e-3)
            im = ax.pcolormesh(
                K,
                H,
                data,
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=data.max()),
                shading="auto",
            )
        else:
            im = ax.pcolormesh(
                K, H, data, cmap=cmap + "_r", vmin=0, vmax=1, shading="auto"
            )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$\kappa = J_2/J_1$", fontsize=12)
        ax.set_ylabel(r"$h/J_1$", fontsize=12)
        ax.set_title(title, fontsize=11, pad=8)

        # Annotate known phase boundaries
        # FM–PM boundary (approx): h_c ≈ 1 - kappa for kappa < 0.5
        k_fm = kappa_vals[kappa_vals <= 0.5]
        ax.plot(k_fm, 1.0 - k_fm, "w--", lw=1.2, alpha=0.7, label="FM/PM (approx)")

        # AP–PM boundary (approx): h_c ≈ 1 + kappa for kappa > 0.5
        k_ap = kappa_vals[kappa_vals >= 0.5]
        ax.plot(k_ap, k_ap - 0.5, "w:", lw=1.2, alpha=0.7, label="AP/PM (approx)")

        # Phase labels
        ax.text(
            0.18,
            0.25,
            "FM",
            color="white",
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.75,
            0.25,
            "AP",
            color="white",
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )

        if idx == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.4)

    fig.suptitle(
        f"ANNNI Phase Diagram via Loschmidt Echo  (N={n_qubits} qubits, 2nd-order Trotter)",
        fontsize=13,
        y=1.02,
    )

    plt.savefig("annni_phase_diagram.png", dpi=150, bbox_inches="tight")
    print("\nSaved: annni_phase_diagram.png")
    plt.close()


def plot_timeseries(
    n_qubits,
    J1,
    representative_points,
    dt=0.05,
    t_max=8.0,
    init_state="ferro",
    pbc=True,
):

    fig, axes = plt.subplots(
        2,
        len(representative_points),
        figsize=(4.5 * len(representative_points), 7),
        sharex="col",
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(representative_points)))

    for col, (label, kappa, h) in enumerate(representative_points):
        print(f"  Computing timeseries: {label} (κ={kappa}, h={h})")
        times, echoes = compute_echo_timeseries(
            n_qubits, J1, kappa, h, t_max, dt, init_state, pbc
        )
        rates = -np.log(echoes + 1e-10) / n_qubits

        ax_e = axes[0, col]
        ax_r = axes[1, col]
        c = colors[col]

        ax_e.plot(times, echoes, color=c, lw=1.5)
        ax_e.fill_between(times, 0, echoes, alpha=0.15, color=c)
        ax_e.set_ylim(-0.05, 1.05)
        ax_e.set_ylabel(r"$\mathcal{L}(t)$", fontsize=11)
        ax_e.set_title(
            f"{label}\n" r"$\kappa$" f"={kappa}, " r"$h$" f"={h}", fontsize=10
        )
        ax_e.axhline(0, color="gray", lw=0.5)
        ax_e.grid(alpha=0.3)

        ax_r.plot(times, rates, color=c, lw=1.5)
        ax_r.set_ylabel(r"$\lambda(t) = -\frac{1}{N}\ln\mathcal{L}$", fontsize=11)
        ax_r.set_xlabel(r"$t\,J_1$", fontsize=11)
        ax_r.grid(alpha=0.3)

        # Mark non-analyticities (local maxima in rate fn = DQPT candidates)
        from scipy.signal import argrelmax

        (peaks,) = argrelmax(rates, order=2)
        if len(peaks):
            ax_r.scatter(
                times[peaks],
                rates[peaks],
                color="red",
                s=30,
                zorder=5,
                label="DQPT?",
                marker="v",
            )
            ax_r.legend(fontsize=8)

    axes[0, 0].set_title(axes[0, 0].get_title(), fontsize=10)
    fig.suptitle(
        f"Loschmidt Echo Dynamics  (N={n_qubits}, 2nd-order Trotter, dt={dt})",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig("annni_echo_timeseries.png", dpi=150, bbox_inches="tight")
    print("Saved: annni_echo_timeseries.png")
    plt.close()


N_QUBITS = 8  # system size (8 is tractable; go to 10-12 with patience)
DT = 0.4  # Trotter step (smaller = more accurate, more gates)
T_TOTAL = 5.0  # total evolution time for phase diagram scan
INIT_STATE = "ferro"  # reference state

# Phase diagram grid resolution (increase for publication quality)
KAPPA_VALS = np.linspace(0.0, 1, 15)
H_VALS = np.linspace(0.0, 2, 15)

# ── Phase diagram ───────────────────────────────────────────────────────────
rate_avg, rate_max, echo_final = compute_phase_diagram(
    n_qubits=N_QUBITS,
    kappa_vals=KAPPA_VALS,
    h_vals=H_VALS,
    t_total=T_TOTAL,
    dt=DT,
    init_state=INIT_STATE,
)

plot_phase_diagram(KAPPA_VALS, H_VALS, rate_avg, rate_max, echo_final, N_QUBITS)

print("\nDone. Outputs: annni_phase_diagram.png, annni_echo_timeseries.png")
