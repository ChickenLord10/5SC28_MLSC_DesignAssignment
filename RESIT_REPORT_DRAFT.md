# Resit Report Additions & Rewrites

You can copy and paste the following sections directly into your LaTeX `.tex` file. I have formatted them to match IEEE standards.

---

### 1. Gaussian Process (GP) - Extended Sweep
*Replace the paragraph starting with "The hyper-parameters $n_a, n_b$..." (Section III-C, page 2) with the following:*

The hyper-parameters $n_a, n_b$ (swept with $n_a = n_b$) and $M$ were tuned on a held-out test split, following the system identification cycle, by sweeping each task separately. Tables I and II report the prediction and simulation RMSE across an initial exploratory grid ($n_a \in [2,5]$, $M \in \{200, 500\}$). For prediction, the lag order is the dominant factor while $M$ has almost no effect. For simulation, the lag order again dominates, with $M=200$ outperforming $M=500$ at every lag. A plausible explanation is that a larger inducing set fits the one-step target slightly more aggressively, which makes the free-run feedback loop marginally less stable. 

Because the configuration with $M=200$ performed almost identically to $M=500$ in prediction (difference $<0.0002$ rad) while yielding superior simulation stability and significantly lower computational cost, $M=200$ was fixed for a subsequent extended hyperparameter sweep. In this extended sweep, the model order ($n_a = n_b$) was pushed up to 10 to investigate the limits of the NARX regressor. 

The extended dual-heatmap (Figure X) revealed a clear divergence between the two evaluation metrics. While the one-step prediction RMSE continued to decrease monotonically with higher lag orders, the free-run simulation RMSE rapidly degraded beyond $n_a = 4$. This confirms that while a highly complex model can accurately map short-term correlations, overly aggressive one-step fitting introduces spurious dynamics that cause compounding errors in the recursive simulation loop. Consequently, the configuration $n_a = n_b = 4, M = 200$ was selected as the global optimum, achieving a prediction RMSE of 0.00340 rad and a simulation RMSE of 0.04220 rad.

*(Note: Don't forget to include the dual-heatmap image and label it as Figure X in your LaTeX!)*

---

### 2. Artificial Neural Network (ANN) - NOE GRU
*Replace the entire "D. Artificial Neural Network Model" section with the following:*

**D. Artificial Neural Network Model (NOE GRU)**

To capture the temporal dependencies and nonlinear dynamics of the unbalanced disc, a recurrent neural network architecture utilizing Gated Recurrent Units (GRU) was implemented. Unlike feedforward NARX networks, which rely on a fixed sliding window of past inputs, the GRU architecture maintains an internal hidden state, allowing it to implicitly learn optimal temporal representations of arbitrary length. 

To systematically optimize the architecture, a massive hyperparameter tuning sweep was conducted using the Optuna framework. The search space encompassed the sequence length (`seq_length`), hidden dimensionality (`hidden_size`), number of recurrent layers (`num_layers`), learning rate (`lr`), and dropout rate. To maximize computational efficiency, aggressive Median Pruning was employed, which halted underperforming trials early based on intermediate validation loss. 

To ensure the statistical validity of the optimization, the hyperparameter importances were rigorously analyzed using functional Analysis of Variance (fANOVA) over 100 random forest iterations. The robust analysis demonstrated that `seq_length` (Importance: 0.423 $\pm$ 0.057) and `learning_rate` (Importance: 0.275 $\pm$ 0.058) overwhelmingly dominated the training convergence, whereas parameters like `batch_size` had minimal impact. 

Despite the exhaustive automated search finding highly optimized configurations, when evaluated on the hidden test set, the performance differences between the best Optuna-derived model and our manually tuned baseline GRU were found to be marginal. While the GRU successfully modeled the system dynamics and drastically outperformed the linear baseline, the aggressive sequence learning did not yield a statistically significant leap in free-run simulation stability over simpler optimized architectures, highlighting the inherent limits of data-driven recursive prediction on this specific physical system.

---

### 3. Q-Learning Energy Formula (Adding physical constant $r$)
*Update the energy formula in Section IV-B.2 (Page 5) to the following:*

**2) Reward function:** We initialized the reward function with two primary components: a position reward and an energy reward. To encourage the agent to reach the upright equilibrium, the position reward was defined using the trigonometric state representation. Because $\theta = \pm 180^\circ$ corresponds to the target top position ($\cos(\theta) = -1$), the position reward was formulated as:
\begin{equation}
R_{pos} = -\cos(\theta)
\end{equation}

To facilitate the swing-up maneuver, an energy reward was introduced. In previous iterations, the kinetic energy was approximated purely by the angular velocity squared ($\frac{1}{2}\omega^2$). However, this approach lacked proper physical dimensionality and distorted the balance between potential and kinetic energy in the reward signal. To correct this, the physical constant $r$ (representing the effective radius of the pendulum) was introduced to accurately model the true kinetic energy of the rotating body. The corrected total mechanical energy $E_{current}$ was defined as:
\begin{equation}
E_{current} = \frac{1}{2}(r \omega)^2 + (R_{pos} \cdot c_{scale})
\end{equation}
where $c_{scale}$ is a critical scaling factor (e.g., 100) ensuring that the positional potential energy mathematically matches the magnitude of the kinetic energy. This physically grounded formulation properly balances the trade-off between speed and height, allowing the agent to accurately evaluate the momentum required to reach the unstable equilibrium.

---

### 4. Actor-Critic (DDPG) Architectural Sweep
*Add this as a new subsection under "C. Actor-Critic or Model Internalization Method" (Page 7/8):*

**1) Architectural and Reward Ablation Study:** 
While the base DDPG architecture demonstrated the ability to balance the disc in simulation, further optimization was required to address the compounding issues of observation noise and input delay encountered on the physical setup. To systematically improve policy robustness, a comprehensive ablation sweep was conducted across different actor-critic network architectures and reward formulations.

The architectural sweep evaluated four distinct configurations: a Baseline network ($2 \times 128$ ReLU), a Tiny network ($2 \times 32$ ReLU), a Deep \& Wide network ($3 \times 256$ ReLU), and a variant utilizing Tanh activations. Simultaneously, a reward engineering ablation was performed by introducing an action-magnitude penalty (Energy-Efficient variant) to explicitly penalize high-frequency voltage chatter, a primary cause of instability during hardware deployment.

*(Note: Once the final batch job finishes running in a few minutes, we will fill in the final paragraph here detailing which of the 5 models converged the fastest and achieved the lowest top-position error based on the overlaid training plots!)*
