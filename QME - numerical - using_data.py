import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor


def reduced_density_matrix(psi, N, A_subsys):
    
    psi_tensor = psi.reshape([2] * N)

    # Move subsystem spins to the first axes
    psi_tensor = np.moveaxis(
        psi_tensor,
        A_subsys,
        range(len(A_subsys))
    )

    # Subsystem dimension = 2^(number of spins)
    dim_A = 2 ** len(A_subsys)

    # Reshape into subsystem x environment
    M = psi_tensor.reshape(dim_A, -1)

    # Trace out environment
    rho_A = M @ M.conj().T

    return rho_A


def trace_distance_squared(rho1, rho2):
    """
    D^2 = [1/2 ||rho1 - rho2||_1]^2
    """

    Delta = rho1 - rho2

    # Remove small numerical non-Hermiticity
    Delta = 0.5 * (Delta + Delta.conj().T)

    eigenvalues = np.linalg.eigvalsh(Delta)

    D = 0.5 * np.sum(np.abs(eigenvalues))

    return D ** 2


def main():

    
    #Parameters
    

    N = 14
    A_subsys = [5]  #Sixth spin

    theta_lister = [0.0]
    theta_list = np.array(theta_lister) * 2*np.pi

    tlist = np.linspace(0, 5, 101)

    spectrum_filename = "H_spectrum_14.npz"

    
    #Load saved H eigenspectrum


    with np.load(spectrum_filename) as spectrum:


        eigvalsH = np.asarray(
            spectrum["eigvalsH"],
            dtype=float
        )

        eigvecsH = np.asarray(spectrum["eigvecsH"])

    dim = 2**N

    if eigvalsH.shape != (dim,):
        raise ValueError(
            f"Expected {dim} eigenvalues, got {eigvalsH.shape}"
        )

    if eigvecsH.shape != (dim, dim):
        raise ValueError(
            f"Expected eigenvectors of shape {(dim, dim)}, "
            f"got {eigvecsH.shape}"
        )



    
    #Loop over initial states
    

    Delta_All = []

    for theta in theta_list:

        print("\n==========================================")
        print(f"theta = {theta:.6f} rad")
        print(f"theta/2pi = {theta / (2* np.pi):.4f}")
        print("==========================================")

        
        #Initial state
        

        ket0 = basis(2, 0)
        ket1 = basis(2, 1)

        single = (
            np.cos(theta / 2) * ket0
            + np.sin(theta / 2) * ket1
        )

        psi0 = tensor([single] * N)

        psi0_array = psi0.full().ravel()

        print("Norm(psi0) =", np.linalg.norm(psi0_array))

        
        #Weights
       
        overlaps = eigvecsH.conj().T @ psi0_array

        rho_mumu = np.abs(overlaps)**2

        print("Total spectral weight:", np.sum(rho_mumu))

        # Initial energy expectation value
        E_initial = np.sum(eigvalsH * rho_mumu)

        print(f"Initial energy = {E_initial:.12f}")

     
        #Reconstruct time-evolved states
        

        rhoA_t_list = []

        for t in tlist:

            # |psi(t)> = sum_mu c_mu exp(-i E_mu t) |E_mu>
            phases = np.exp(-1j * eigvalsH * t)

            psi_t = eigvecsH @ (overlaps * phases)

            #Reduced density matrix of the selected spin
            rhoA_t = reduced_density_matrix(
                psi_t,
                N,
                A_subsys
            )

            rhoA_t_list.append(rhoA_t)

        
        #Initial subsystem purity
        

        rhoA_initial = rhoA_t_list[0]

        initial_purity = np.trace(
            rhoA_initial @ rhoA_initial
        ).real

        print("Initial Tr(rho_A^2) =", initial_purity)

        
        #Finite-time averaged diagonal ensemble
        

        rhoA_DE = np.mean(
            np.asarray(rhoA_t_list),
            axis=0
        )

        DE_purity = np.trace(
            rhoA_DE @ rhoA_DE
        ).real

        print("Tr(rho_A,DE^2) =", DE_purity)

        
        #Numerical trace distance squared
        

        Delta_t = np.array([
            trace_distance_squared(rhoA, rhoA_DE)
            for rhoA in rhoA_t_list
        ])

        Delta_All.append(Delta_t)

        
        # print D^2(t)
        # print("\nNumerical trace distance squared:")
        # print(f"{'Time':>12} {'D^2(t)':>18}")

        # for t, value in zip(tlist, Delta_t):
        #     print(f"{value:18.12e}")

        
        #Save numerical data
        

        output_filename = (
            f"D_squared_theta_{theta / np.pi:.2f}pi.txt"
        )

        np.savetxt(
            output_filename,
            np.column_stack((tlist, Delta_t)),
            header="time D_squared",
            fmt="%.12e"
        )
        #This is the txt file used in the comparison code with the analytical formula and ΔE cutoff window
        np.savetxt(
            "D2.txt",
            Delta_t,
            fmt="%.12e"
        )

        print(f"\nSaved results to {output_filename}")

    
    #Plot D^2(t)

    plt.figure(figsize=(8, 5))

    for i, theta in enumerate(theta_list):

        plt.plot(
            tlist,
            Delta_All[i],
            linewidth=2,
            label=rf"$\theta={theta / (2*np.pi):.2f}\times 2\pi$"
        )

    plt.xlabel("Time")
    plt.ylabel(r"$D^2(\rho_A(t),\rho_{A,\mathrm{DE}})$")
    plt.title("Numerical trace distance squared")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        "D_squared_from_saved_spectrum.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
