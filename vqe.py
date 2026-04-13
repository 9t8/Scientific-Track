import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def get_H(num_spins, k, h):
    """Construction function the ANNNI Hamiltonian (J=1)"""

    # Interaction between spins (neighbouring):
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H - (qml.PauliZ(i) @ qml.PauliZ(i + 1))

    # Interaction between spins (next-neighbouring):
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliZ(i) @ qml.PauliZ(i + 2))

    # Interaction of the spins with the magnetic field
    for i in range(0, num_spins):
        H = H - h * qml.PauliX(i)

    return H


def hva_layer(params, n_qubits, j1=1.0, j2=0.5, h=1.0, periodic=True, p=0.0):
    """
    One HVA layer for the ANNNI model.
    params: array of 3 angles [theta_nn, theta_nnn, theta_x]
    """
    theta_nn, theta_nnn, theta_x = params

    for i in range(0, n_qubits - 1, 2):
        qml.IsingZZ(2 * theta_nn, wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.IsingZZ(2 * theta_nn, wires=[i, i + 1])

    if periodic and n_qubits > 2:
        qml.IsingZZ(2 * theta_nn, wires=[n_qubits - 1, 0])

    # --- Block 2: Next-nearest-neighbor ZZ ---
    # e^{-i * theta_nnn * sum ZiZi+2}
    # Even sublattice: (0,2), (4,6), ...
    for i in range(0, n_qubits - 2, 2):
        qml.IsingZZ(2 * theta_nnn * j2, wires=[i, i + 2])
    # Odd sublattice: (1,3), (5,7), ...
    for i in range(1, n_qubits - 2, 2):
        qml.IsingZZ(2 * theta_nnn * j2, wires=[i, i + 2])
    if periodic and n_qubits > 3:
        qml.IsingZZ(2 * theta_nnn * j2, wires=[n_qubits - 2, 0])
        qml.IsingZZ(2 * theta_nnn * j2, wires=[n_qubits - 1, 1])

    # --- Block 3: Transverse field X ---
    # e^{-i * theta_x * sum Xi} = product of Rx rotations
    if abs(h) > 1e-12:
        for i in range(n_qubits):
            qml.RX(2 * theta_x * h, wires=i)
    else:
        # Replace with a free RX that isn't scaled by h,
        # so it retains gradient signal even at h=0
        for i in range(n_qubits):
            qml.RX(theta_x, wires=i)


def do_vqe(n_qubits, k, h, p):
    hamiltonian = get_H(n_qubits, k, h)
    n_layers = 6
    dev = qml.device("lightning.gpu", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        # create_ansatz(params, n_qubits)

        for l in range(n_layers):
            hva_layer(params[l], n_qubits, 1.0, k, h, True, p)
        return qml.expval(hamiltonian)

    @qml.qnode(dev)
    def state_circuit(params, n_qubits):
        # create_ansatz(params, n_qubits)

        for l in range(n_layers):
            hva_layer(params[l], n_qubits, 1.0, k, h, True, p)
        return qml.state()

    max_iters = 100
    tolerance = 1e-04

    # create an optimizer
    opt = qml.AdamOptimizer(stepsize=0.1)

    # energy is a list that stores all the estimates for the ground-state energy
    energy = []

    # execute the VQE optimization loop
    params = pnp.array(
        pnp.random.uniform(-0.1, 0.1, size=(n_layers, 3)), requires_grad=True
    )
    for i in range(max_iters):
        params, prev_energy = opt.step_and_cost(circuit, params)
        energy.append(prev_energy)

        if i > 1:
            if np.abs(energy[-2] - energy[-1]) < tolerance:
                break

    return state_circuit(params, n_qubits)
