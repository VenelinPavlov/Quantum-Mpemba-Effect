import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, qeye, sigmax, sigmaz, Qobj
import zipfile
import os

filename = "H_spectrum_14.npz"

with open(filename, "rb") as f:
    data = f.read()

print("File size:", len(data))

# # Look for ZIP local-file headers.
# signature = b"PK\x03\x04"
# positions = []
# start = 0

# while True:
#     pos = data.find(signature, start)
#     if pos == -1:
#         break
#     positions.append(pos)
#     start = pos + 4

# print("Local ZIP headers found:", len(positions))
# print("First positions:", positions[:20])

# # Also look for the ZIP end-of-central-directory record.
# print(
#     "End-of-central-directory found:",
#     data.rfind(b"PK\x05\x06")
# )

def main():

    
    #Parameters
    
    theta = 0.00 * 2*np.pi

    N = 14

    tlist = np.linspace(0, 5, 101)

    E_cut_low = 0
    E_cut_high = 11.0

    
    #Load numerical result

    D2_numerical = np.loadtxt("D2.txt")
    D2_numerical = np.asarray(D2_numerical).reshape(-1)

    if len(D2_numerical) != len(tlist):
        raise ValueError(
            f"Numerical data has {len(D2_numerical)} points, "
            f"but tlist has {len(tlist)} points."
        )

    
    #Load H spectrum
    
    spectrum = np.load("H_spectrum_14.npz")

    
    eigvalsH = np.asarray(spectrum["eigvalsH"], dtype=float)
    eigvecsH = np.asarray(spectrum["eigvecsH"])


    #initial state
    

    ket0 = basis(2, 0)
    ket1 = basis(2, 1)

    single = (
        np.cos(theta / 2) * ket0
        + np.sin(theta / 2) * ket1
    )

    psi0 = tensor([single] * N)

    psi0_array = psi0.full().ravel()


    
    overlaps = eigvecsH.conj().T @ psi0_array
    rho_mumu = np.abs(overlaps)**2

    print("Total spectral weight:", np.sum(rho_mumu))

    
    #Initial energy
    

    E_initial = np.sum(eigvalsH * rho_mumu)

    print(f"Initial energy: {E_initial:.12f}")

    
    #Energy window and renormalization
    

    energy_mask = (
        (eigvalsH >= E_cut_low)
        & (eigvalsH <= E_cut_high)
    )

    rho_mumu_cut = np.where(energy_mask, rho_mumu, 0.0)

    retained_weight = np.sum(rho_mumu_cut)

    print("Retained spectral weight:", retained_weight)

    if retained_weight <= 0:
        raise ValueError(
            "No spectral weight in the selected energy window."
        )

    rho_mumu_new = rho_mumu_cut / retained_weight

    print("Normalized retained weight:", np.sum(rho_mumu_new))

    
    #Energy variance of the filtered initial state
    

    E_mean_filtered = np.sum(
        eigvalsH * rho_mumu_new
    )

    E2_mean_filtered = np.sum(
        eigvalsH**2 * rho_mumu_new
    )

    VarE_filtered = (
        E2_mean_filtered - E_mean_filtered**2
    )

    DeltaE_filtered = np.sqrt(VarE_filtered)

    print("\nFiltered energy statistics:")
    print(f"<H>      = {E_mean_filtered:.12f}")
    print(f"<H^2>    = {E2_mean_filtered:.12f}")
    print(f"Var(H)   = {VarE_filtered:.12e}")
    print(f"Delta E' = {DeltaE_filtered:.12e}")


    
    #Omega(t) = sum_mu p_mu exp(-i E_mu t)
    

    Omega = np.exp(
        -1j * np.outer(tlist, eigvalsH)
    ) @ rho_mumu_new

    Omega_squared = np.abs(Omega)**2


    prefactor= D2_numerical[0]
    #Analytical 
    #D2_an(t) = 1/4 |Omega(t)|^4 * trace_part^2

    D2_analytical = (
        0.25
        * Omega_squared**2
        * prefactor*4



    )

    
    #Save analytical data

    np.savetxt(
        "D2_analytical.txt",
        D2_analytical
    )

    np.savetxt(
        "comparison_data.txt",
        np.column_stack([
            tlist,
            D2_numerical,
            D2_analytical
        ]),
        header="time D2_numerical D2_analytical"
    )


    #Plot
    
    plt.figure(figsize=(9, 5.5))

    plt.plot(
        tlist,
        D2_numerical,
        linewidth=2,
        label="Numerical"
    )

    plt.plot(
        tlist,
        D2_analytical,
        "--",
        linewidth=2,
        label="Analytical"
    )

    plt.xlabel("Time")
    plt.ylabel(r"$D$")
    plt.title("Numerical vs analytical result")
    plt.xlim(0,1)

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        "comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    
    # Plot rho_mumu as a function of energy
   

    plt.figure(figsize=(9, 5.5))

    # Sort energies and corresponding weights
    sort_index = np.argsort(eigvalsH)
    E_mu_sorted = eigvalsH[sort_index]
    rho_mumu_sorted = rho_mumu[sort_index]

    np.savetxt(
        "rho_mumu_energy.txt",
        E_mu_sorted
    )

    np.savetxt(
        "rho_mumu_values.txt",
        rho_mumu_sorted
    )

    print("Saved energy data to: rho_mumu_energy.txt")
    print("Saved rho_mumu data to: rho_mumu_values.txt")

    # Plot vertical lines and dots
    markerline, stemlines, baseline = plt.stem(
        E_mu_sorted,
        rho_mumu_sorted
    )

    plt.setp(markerline, markersize=3, alpha=0.7)
    plt.setp(stemlines, linewidth=0.8, alpha=0.7)
    baseline.set_visible(False)

    plt.xlabel(r"Energy $E_\mu$")
    plt.ylabel(r"$\rho_{\mu\mu}=|\langle E_\mu|\psi_0\rangle|^2$")
    plt.title(r"Initial-state spectral weights $\rho_{\mu\mu}$")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        "rho_mumu_vs_energy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

   

if __name__ == "__main__":
    main()
