def calculate_custom_reward(state, action, w_energy, w_position, w_balance, w_stab=None):
    if w_stab is None:
        w_stab = w_balance

    sin_th, cos_th, omega = state
    
    position_reward = -cos_th 
    
    M = 0.07618
    g = 9.80155
    L = 0.04206
    J = 0.00024421
    
    # Energy in Joules
    E_kin = 0.5 * J * (omega ** 2)
    E_pot = -M * g * L * cos_th
    E_current = E_kin + E_pot
    
    # Target energy (upright, at rest)
    E_target = M * g * L
    
    # Normalize penalty by max possible energy difference (bottom to top)
    energy_penalty = -abs(E_current - E_target) / (2 * E_target)
    
    # --- Strict Speed Limit for "Unnecessary Kinetic Energy" ---
    # We allow a 10% margin above E_target before aggressively penalizing
    E_excess = max(0.0, E_current - 1.1 * E_target)
    # Normalize excess penalty quadratically
    excess_penalty = -(E_excess / E_target) ** 2
    energy_penalty += excess_penalty 
    
    if cos_th < -0.8: 
        balance_bonus = w_position * (1.0 - min(1.0, abs(omega) / 6.0))
        stab_penalty = w_position * (0.15 * (omega ** 2) + 0.5 * (action ** 2))
    else:
        balance_bonus = 0.0
        stab_penalty = 0.0

    return (w_position * position_reward) + (w_energy * energy_penalty) + (w_balance * balance_bonus) - (w_stab * stab_penalty)
