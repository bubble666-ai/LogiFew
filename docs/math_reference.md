# LogiFew Mathematical Notes

## 1. Loss Components
- **Entailment loss**  
  \[
  L_{\text{entail}} = \text{MSE}\left(p, y\right) = \frac{1}{N} \sum_{i=1}^N \left(p_i - y_i\right)^2
  \]
  where \(p_i\) is the predicted entailment probability and \(y_i \in \{0, \tfrac{1}{2}, 1\}\) encodes `no`, `unknown`, `yes`.

- **Consistency regularizer**  
  \[
  L_{\text{cons}} = \lambda_{\text{cons}} \left( \frac{1}{N} \sum_{i=1}^N \max(0, p_i - 1)^2 + \max(0, -p_i)^2 \right)
  \]
  pushing probabilities into \([0, 1]\).

- **Contrastive rule loss**  
  \[
  L_{\text{contrast}} = -\beta_{\text{contrast}} \, \frac{1}{N} \sum_{i=1}^N \log \sigma\left(\text{sim}(q_i, r_i) - \text{sim}(q_i, \tilde{r}_i)\right)
  \]
  where \(q_i\) is the query embedding, \(r_i\) the true rule embedding, \(\tilde{r}_i\) a corrupted rule.

- **Total loss**  
  \[
  L_{\text{total}} = L_{\text{entail}} + L_{\text{cons}} + L_{\text{contrast}}
  \]

## 2. Metrics
- **Exact Deduction Accuracy (EDA)**  
  \[
  \text{EDA} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\left[\hat{y}_i = y_i\right]
  \]

- **Proof Validity Rate (PVR)**  
  \[
  \text{PVR} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\left[\text{prove}(\mathcal{P}_i)\right]
  \]
  where \(\mathcal{P}_i\) is the proof trace for example \(i\).

- **Logical Consistency Score (LCS)**  
  \[
  \text{LCS} = 1 - \left|p_i - s_i\right|
  \]
  averaged over all samples; \(s_i\) is the symbolic inference probability.

- **Rule Induction F1 (RIF1)**  
  Treat discovered rules as set \(D\), gold rules as \(G\):
  \[
  \text{RIF1} = \frac{2|D \cap G|}{|D| + |G|}
  \]

- **Data Efficiency Ratio (DER)**  
  \[
  \text{DER} = \frac{\text{EDA}}{\text{Number of training samples}}
  \]

## 3. Memory Attention
Given query embedding \(q \in \mathbb{R}^d\), memory keys \(K \in \mathbb{R}^{m \times d}\), values \(V \in \mathbb{R}^{m \times d_v}\):
\[
\alpha = \text{softmax}\left(\frac{q K^\top}{\sqrt{d}}\right), \quad z = \alpha V
\]

## 4. Suggested References
- Bottou, L. “Large-Scale Machine Learning with Stochastic Gradient Descent,” COMPSTAT 2010.  
- van den Broeck, G. et al. “Probabilistic Soft Logic,” AAAI 2011.  
- Evans, R., & Grefenstette, E. “Learning Explanatory Rules from Noisy Data,” JAIR 2018.  
- Cingillioglu, N., & Russo, A. “DeepLogic: End-to-End Differentiable Logical Reasoning,” NIPS Workshop 2018.  
- Han, W. et al. “Neural Theorem Provers,” NeurIPS 2018.  
- T5: Raffel, C. et al. “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” JMLR 2020.  
- CLEVRER: Yi, K. et al. “CLEVRER: Collision Events for Video Representation and Reasoning,” ICLR 2020.
