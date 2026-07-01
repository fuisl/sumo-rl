# Thesis Pseudocode

This page collects thesis-style LaTeX pseudocode for the current canonical
graph topology generator and the DCRNN-based PPO variants in this repository.
The snippets are written to match the current code paths in:

- `sumo_rl/models/graph.py`
- `sumo_rl/models/dcrnn.py`
- `sumo_rl/agents/ppo/rllib_module.py`
- `sumo_rl/agents/ppo/learner.py`

The snippets below assume the LaTeX packages `algorithm` and `algpseudocode`.

## Shared notation

Use the following notation consistently across the algorithms:

- `B`: batch size
- `H`: graph-history length
- `N`: number of graph nodes
- `F`: node-feature dimension
- `d_h`: DCRNN hidden dimension
- `X \in \mathbb{R}^{B \times H \times N \times F}`: graph-history observation
- `A \in \mathbb{R}^{N \times N}`: fixed directed adjacency matrix
- `\mathcal{S}`: diffusion supports derived from `A`
- `z_i`: latent for ego agent or traffic signal `i`
- `\pi_i`: policy head for agent `i`
- `V_i`: value head for agent `i`

The DCRNN variants in this repo use graph communication only through diffusion
over the fixed adjacency matrix. They do not add a separate GAT layer after the
DCRNN encoder.

## Graph topology generator

```latex
\begin{algorithm}[t]
\caption{\textsc{ExtractTLSTopology}: TLS Communication Graph Extraction from SUMO \texttt{.net.xml}}
\begin{algorithmic}[1]
\Require SUMO network file $f_{\text{net}}$
\Ensure Topology object with workers, directed TLS edges, undirected TLS edges, edge weights, and super-edge metadata
\State Parse all junction positions from $f_{\text{net}}$
\State Parse all non-internal road edges into a catalog with source, target, length, travel time, lane count, and polyline shape
\State Parse connection elements into legal edge-to-edge transitions
\State Identify traffic-signal junctions and their TLS IDs
\State Initialize empty map $\texttt{bestByPair}$ from ordered TLS pair $(u,v)$ to best super-edge
\ForAll{traffic signals $u$}
    \State Let $j_u$ be the junction of $u$
    \ForAll{outgoing road edges $e_0$ from $j_u$}
        \State Initialize min-priority queue with $e_0$ keyed by accumulated travel time
        \State Record $\texttt{best\_cost}[e_0] \gets \tau(e_0)$ and $\texttt{pred}[e_0] \gets \varnothing$
        \While{queue is not empty}
            \State Pop edge $e$ with smallest accumulated travel time
            \State Let $j$ be the target junction of $e$
            \If{$j$ is a TLS junction and $j \neq j_u$}
                \State Reconstruct legal road-edge path $P$ by following predecessor links from $e$
                \ForAll{traffic signals $v$ attached to junction $j$}
                    \State Aggregate path statistics:
                    \Statex \hspace{\algorithmicindent} $\ell \gets \sum_{e' \in P} \ell(e')$, \quad $\tau \gets \sum_{e' \in P} \tau(e')$, \quad $\texttt{lanes} \gets \max_{e' \in P} \texttt{lanes}(e')$
                    \If{$u \neq v$ and $\big((u,v) \notin \texttt{bestByPair}$ or $\tau < \texttt{bestByPair}[(u,v)].\tau\big)$}
                        \State Store super-edge $(u,v,\ell,\tau,\texttt{lanes},P)$ in $\texttt{bestByPair}[(u,v)]$
                    \EndIf
                \EndFor
                \State \textbf{break}
            \EndIf
            \ForAll{legal successor edges $e'$ of $e$}
                \State $\texttt{cost}' \gets \texttt{best\_cost}[e] + \tau(e')$
                \If{$\texttt{cost}' < \texttt{best\_cost}[e']$}
                    \State Update $\texttt{best\_cost}[e']$ and $\texttt{pred}[e'] \gets e$
                    \State Push $e'$ into the queue
                \EndIf
            \EndFor
        \EndWhile
    \EndFor
\EndFor
\State $\texttt{super\_edges} \gets$ all values stored in $\texttt{bestByPair}$
\State $\texttt{directed\_edges} \gets$ unique ordered pairs $(u,v)$ from $\texttt{super\_edges}$
\State $\texttt{edges} \gets$ unique undirected pairs $\{u,v\}$ from $\texttt{super\_edges}$
\ForAll{super-edges $(u,v,\ell,\tau,\dots)$}
    \State Accumulate undirected weight $w_{\{u,v\}} \gets w_{\{u,v\}} + 1/\tau$
\EndFor
\State Return topology object with workers, $\texttt{directed\_edges}$, $\texttt{edges}$, $W$, and $\texttt{super\_edges}$
\end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}[t]
\caption{\textsc{BuildTrafficSignalGraph}: Convert TLS Topology to the Repo's Model Graph}
\begin{algorithmic}[1]
\Require Ordered traffic signals $\mathcal{T}$, optional net file $f_{\text{net}}$, self-loop flag $\delta_{\text{self}}$, feature layout $\ell$
\Ensure Graph object $G = (\texttt{ts\_ids}, \texttt{ts\_index}, A, \texttt{edge\_index}, \texttt{metadata})$
\State Normalize feature layout $\ell$
\State Build deterministic node list $\texttt{ts\_ids} \gets (t_0, t_1, \dots, t_{|\mathcal{T}|-1})$
\State Build index map $\texttt{ts\_index}[t_j] \gets j$ for all $t_j \in \texttt{ts\_ids}$
\State Compute graph metadata:
\Statex \hspace{\algorithmicindent} $\texttt{max\_lanes} \gets \max\big(1, \max_{t \in \mathcal{T}} |\texttt{lanes}(t)|\big)$
\Statex \hspace{\algorithmicindent} $\texttt{max\_green\_phases} \gets \max\big(1, \max_{t \in \mathcal{T}} \texttt{num\_green\_phases}(t)\big)$
\Statex \hspace{\algorithmicindent} $N \gets |\texttt{ts\_ids}|$
\If{$f_{\text{net}}$ is missing or topology extraction fails}
    \State Fall back to the legacy lane-link graph builder
\EndIf
\State $\texttt{topology} \gets \textsc{ExtractTLSTopology}(f_{\text{net}})$
\State Initialize adjacency matrix $A \gets \mathbf{0}_{N \times N}$
\ForAll{$(u, v) \in \texttt{topology.directed\_edges}$}
    \If{$u \in \texttt{ts\_index}$ and $v \in \texttt{ts\_index}$}
        \State $A[\texttt{ts\_index}[u], \texttt{ts\_index}[v]] \gets 1$
    \EndIf
\EndFor
\If{$\delta_{\text{self}} = \textbf{true}$}
    \State Set all diagonal entries of $A$ to $1$
\EndIf
\State $\texttt{edge\_index} \gets \textsc{NonZeroIndices}(A)$
\State Return graph object with:
\Statex \hspace{\algorithmicindent} $\texttt{ts\_ids}$, $\texttt{ts\_index}$, $A$, $\texttt{edge\_index}$,
\Statex \hspace{\algorithmicindent} $\texttt{num\_nodes}=N$, $\texttt{max\_lanes}$, $\texttt{max\_green\_phases}$,
\Statex \hspace{\algorithmicindent} $\texttt{feature\_layout}=\ell$, and topology source ``tls\_super\_edges''
\end{algorithmic}
\end{algorithm}
```

The first algorithm mirrors the richer topology extractor in
`sumo_rl/agents/fgs/topology.py`, including multi-hop road-path contraction and
undirected inverse-travel-time weights. The second algorithm shows how the
current preferred constructor in `build_traffic_signal_graph(...)` consumes only
the directed TLS pairs, converts them into a binary directed adjacency matrix,
and then adds optional self-loops for the model.

## DCRNN

```latex
\begin{algorithm}[t]
\caption{DCRNN Encoder Forward and Backward Pass}
\begin{algorithmic}[1]
\Require Graph-history tensor $X \in \mathbb{R}^{B \times H \times N \times F}$, adjacency $A$, number of layers $L$, hidden dimension $d_h$, max diffusion step $K$
\Ensure Final node embeddings $E \in \mathbb{R}^{B \times N \times d_h}$
\State Build diffusion supports $\mathcal{S} \gets \textsc{DiffusionSupports}(A)$
\State Transpose sequence view $\tilde{X} \gets \textsc{TransposeToTimeMajor}(X) \in \mathbb{R}^{H \times B \times N \times F}$
\State $\texttt{current\_inputs} \gets \tilde{X}$
\For{$\ell = 1$ to $L$}
    \State Initialize hidden state $h^{(\ell)}_0 \gets \mathbf{0} \in \mathbb{R}^{B \times N d_h}$
    \For{$t = 1$ to $H$}
        \State $x_t^{(\ell)} \gets \texttt{current\_inputs}[t]$
        \State Compute graph-convolutional gates
        \Statex \hspace{\algorithmicindent} $[r_t, u_t] \gets \sigma(\textsc{DiffusionGraphConv}_{\text{gate}}(x_t^{(\ell)}, h^{(\ell)}_{t-1}; \mathcal{S}, K))$
        \State Compute candidate state
        \Statex \hspace{\algorithmicindent} $\hat{h}^{(\ell)}_t \gets \tanh(\textsc{DiffusionGraphConv}_{\text{cand}}(x_t^{(\ell)}, r_t \odot h^{(\ell)}_{t-1}; \mathcal{S}, K))$
        \State Update recurrent state
        \Statex \hspace{\algorithmicindent} $h^{(\ell)}_t \gets u_t \odot h^{(\ell)}_{t-1} + (1-u_t) \odot \hat{h}^{(\ell)}_t$
        \State Store output $o^{(\ell)}_t \gets h^{(\ell)}_t$
    \EndFor
    \State $\texttt{current\_inputs} \gets \textsc{Stack}(o^{(\ell)}_1, \dots, o^{(\ell)}_H)$
\EndFor
\State Final node embeddings $E \gets \textsc{ReshapeNodes}(o^{(L)}_H) \in \mathbb{R}^{B \times N \times d_h}$
\Statex
\Statex \textbf{Backward pass:}
\Require Downstream gradient $\nabla_E \mathcal{L}$ from a task loss $\mathcal{L}$
\State Backpropagate $\nabla_E \mathcal{L}$ through the final reshape and final-time-step selection
\For{$\ell = L$ down to $1$}
    \For{$t = H$ down to $1$}
        \State Propagate gradients through the recurrent update for $h^{(\ell)}_t$
        \State Propagate gradients through candidate and gate diffusion convolutions
        \State Accumulate gradients on diffusion weights, biases, and recurrent states
        \State Propagate gradients through diffusion-support multiplications across all diffusion orders up to $K$
    \EndFor
\EndFor
\State Update all DCRNN parameters with the optimizer
\end{algorithmic}
\end{algorithm}
```

This pseudocode focuses on the encoder/module itself. Diffusion is the graph
message-passing mechanism, while the DCGRU recurrence carries temporal
dependencies across the graph-history window.

## PPO with private DCRNN backbone: `ppo_dcrnn_mlp`

```latex
\begin{algorithm}[t]
\caption{PPO-DCRNN-MLP Forward and Backward Pass}
\begin{algorithmic}[1]
\Require Agent-specific graph-history batch $X_i \in \mathbb{R}^{B \times H \times N \times F}$, private backbone parameters $\theta^{\text{enc}}_i$, policy-head parameters $\theta^{\pi}_i$, value-head parameters $\theta^{V}_i$
\Ensure Policy logits $\ell_i$, value predictions $\hat{v}_i$, and optional cached embeddings $z_i$
\Statex \textbf{Forward pass:}
\If{pre-encoder MLP is enabled}
    \State Apply the same node-wise MLP to every node feature in $X_i$
    \State Obtain encoded graph-history $\bar{X}_i$
\Else
    \State $\bar{X}_i \gets X_i$
\EndIf
\State $(E_i, F_i^{\text{latest}}) \gets \textsc{EncodeGraphWithDCRNN}(\bar{X}_i)$
\State Select ego hidden state $e_i \gets E_i[:, i, :]$
\State Select latest ego node features $f_i \gets F_i^{\text{latest}}[:, i, :]$
\State Form ego latent $z_i \gets [e_i \,\|\, f_i]$
\State Policy logits $\ell_i \gets \pi_i(z_i)$
\State Value prediction $\hat{v}_i \gets V_i(z_i)$
\State Return $\ell_i$, $\hat{v}_i$, and optionally cached embedding $z_i$
\Statex
\Statex \textbf{Backward pass:}
\State Compute PPO ratio $r_i = \frac{\pi_{\theta_i}(a_i \mid s_i)}{\pi_{\theta_i^{\text{old}}}(a_i \mid s_i)}$
\State Compute clipped policy loss
\Statex \hspace{\algorithmicindent} $\mathcal{L}^{\text{clip}}_i = - \mathbb{E} \left[\min \left(r_i \hat{A}_i, \textsc{Clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}_i \right)\right]$
\State Compute value loss $\mathcal{L}^{V}_i = \mathbb{E}\left[(\hat{v}_i - \hat{R}_i)^2\right]$
\State Optionally compute entropy bonus $\mathcal{H}_i$
\State Form total loss
\Statex \hspace{\algorithmicindent} $\mathcal{L}_i = \mathcal{L}^{\text{clip}}_i + c_v \mathcal{L}^{V}_i - c_e \mathcal{H}_i$
\State Backpropagate $\nabla \mathcal{L}_i$ through the policy head and value head into the same private DCRNN backbone
\State Update $\theta^{\text{enc}}_i$, $\theta^{\pi}_i$, and $\theta^{V}_i$ with the optimizer
\end{algorithmic}
\end{algorithm}
```

This matches the current `PPODCRNNTorchRLModule` behavior: each policy owns its
own DCRNN backbone, and both policy and value losses send gradients into that
same private encoder.

## PPO with shared DCRNN backbone: `ppo_dcrnn_shared_mlp`

```latex
\begin{algorithm}[t]
\caption{Shared-Backbone PPO-DCRNN-MLP Forward and Backward Pass}
\begin{algorithmic}[1]
\Require Per-agent graph-history batches $\{X_i\}_{i=1}^M$, shared backbone parameters $\theta^{\text{enc}}$, per-agent policy heads $\{\theta^{\pi}_i\}_{i=1}^M$, per-agent value heads $\{\theta^{V}_i\}_{i=1}^M$
\Ensure Per-agent logits $\{\ell_i\}$, value predictions $\{\hat{v}_i\}$, and embeddings $\{z_i\}$
\Statex \textbf{Forward pass:}
\If{all participating agents share the same observation batch $X$}
    \State $(E, F^{\text{latest}}) \gets \textsc{EncodeGraphWithDCRNN}(X)$ \Comment{encode once}
    \For{$i = 1$ to $M$}
        \State Select agent hidden state $e_i \gets E[:, i, :]$
        \State Select latest agent features $f_i \gets F^{\text{latest}}[:, i, :]$
        \State Form latent $z_i \gets [e_i \,\|\, f_i]$
        \State $\ell_i \gets \pi_i(z_i)$
        \State $\hat{v}_i \gets V_i(z_i)$
    \EndFor
\Else
    \For{$i = 1$ to $M$}
        \State $z_i \gets \textsc{ForwardForAgent}(X_i, i; \theta^{\text{enc}})$
        \State $\ell_i \gets \pi_i(z_i)$
        \State $\hat{v}_i \gets V_i(z_i)$
    \EndFor
\EndIf
\State Return $\{(\ell_i, \hat{v}_i, z_i)\}_{i=1}^M$
\Statex
\Statex \textbf{Backward pass:}
\For{$i = 1$ to $M$}
    \State Compute clipped PPO policy loss $\mathcal{L}^{\text{clip}}_i$
    \State Compute value loss $\mathcal{L}^{V}_i$
    \State Optionally compute entropy bonus $\mathcal{H}_i$
    \State Form agent loss $\mathcal{L}_i = \mathcal{L}^{\text{clip}}_i + c_v \mathcal{L}^{V}_i - c_e \mathcal{H}_i$
\EndFor
\State Aggregate multi-agent loss $\mathcal{L}_{\text{shared}} = \sum_{i=1}^M \mathcal{L}_i$
\State Backpropagate $\nabla \mathcal{L}_{\text{shared}}$ through all agent heads
\State Merge the resulting encoder-side gradients into the single shared DCRNN backbone
\State Register one optimizer over the deduplicated parameter set
\Statex \hspace{\algorithmicindent} $\{\theta^{\text{enc}}\} \cup \{\theta^{\pi}_i, \theta^{V}_i\}_{i=1}^M$
\State Update the shared backbone and all heads with one optimizer step
\end{algorithmic}
\end{algorithm}
```

This matches the current shared PPO design:

- one parent-owned DCRNN backbone
- one graph encode when the per-agent observation batches are identical
- separate policy and value heads per traffic signal
- one learner optimizer with deduplicated shared parameters
