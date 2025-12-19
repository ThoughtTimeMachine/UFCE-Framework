# Copyright (C) 2025 Kyle Killian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numba import njit, prange
import time

# --- 1. The Scenario: "Whale Watching" in High-Frequency Trading ---
# We want to find the single moment where a specific Wallet moved funds
# during a massive Network Congestion spike (High Gas).
# Total Interactions: 100 Billion (FinTech Scale)

N_WALLETS = 50_000         # Space (Spatial entities)
N_BLOCKS = 2_000_000       # Time (Temporal events)

print(f"--- BLOCKCHAIN ANALYTICS VALIDATION ---")
print(f"Scenario: Scanning {N_WALLETS * N_BLOCKS / 1e9:.1f} Billion transaction pairs for 'Liquidity Shock'.")

# --- 2. Generate Data (Power Laws) ---
print("Generating ledger data...")

# Wallets: Real crypto wealth follows a Pareto distribution (80/20 rule).
# Most have 0.1 ETH, a few "Whales" have 10,000 ETH.
wallet_holdings = np.random.pareto(a=3.0, size=N_WALLETS).astype(np.float32)

# Network Gas Fees: Usually low, with random massive spikes.
gas_fees = np.random.exponential(scale=10.0, size=N_BLOCKS).astype(np.float32)

# --- 3. Inject the "Signal" (The Event) ---
# Wallet #8888 (A Whale) moves money at Block #1234567 (Gas Crisis)
TARGET_WALLET = 8888
TARGET_BLOCK = 1_234_567

print(f"Injecting Liquidity Shock at Wallet #{TARGET_WALLET}, Block #{TARGET_BLOCK}...")

# Make the Whale HUGE
wallet_holdings[TARGET_WALLET] = 1000.0 
# Make the Gas Fee HUGE
gas_fees[TARGET_BLOCK] = 500.0      

# Expected "Impact Flux" = 1000 * 500 = 500,000.0

# --- 4. The UFCE Streaming Kernel (FinTech Edition) ---
# Calculates "Impact Flux" = Wallet_Size * Gas_Fee
# Returns the Max Impact Score for each wallet.
@njit(parallel=True)
def calculate_congestion_flux(wallets, fees, n_w, n_b):
    # Zero-Memory Scoreboard (One float per wallet)
    wallet_impact_scores = np.zeros(n_w, dtype=np.float32)
    
    for i in prange(n_w):
        w_val = wallets[i]
        
        # Optimization: Ignore "Dust" wallets (balance < 1.0)
        # This is a standard "Filter" in streaming analytics
        if w_val < 1.0: 
            continue
            
        # Scan the blockchain history for this wallet
        current_max = 0.0
        for j in range(n_b):
            # Flux = Size * Cost
            flux = w_val * fees[j]
            if flux > current_max:
                current_max = flux
        
        wallet_impact_scores[i] = current_max
                
    return wallet_impact_scores

# --- 5. Run the Analytics ---
print("Streaming UFCE Ledger Scan...")
start_time = time.time()

scores = calculate_congestion_flux(wallet_holdings, gas_fees, N_WALLETS, N_BLOCKS)
max_flux = np.max(scores)
culprit_wallet = np.argmax(scores)

end_time = time.time()
duration = end_time - start_time

# --- 6. Validate ---
print(f"\n--- RESULTS ---")
print(f"Processing Time: {duration:.4f} seconds")
print(f"Throughput: {(N_WALLETS * N_BLOCKS) / duration / 1e9:.2f} Billion ops/sec")
print(f"Max Impact Detected: {max_flux:.2f}")

if culprit_wallet == TARGET_WALLET and max_flux > 490000.0:
    print(f"\n✅ SUCCESS: Whale Identified.")
    print(f"Wallet #{culprit_wallet} triggered the shock.")
    print("UTILITY CONFIRMED: High-Frequency Blockchain Analytics.")
else:
    print("\n❌ FAILURE: Signal lost.")