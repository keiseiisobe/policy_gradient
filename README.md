
## $\theta^* = \arg\max_{\theta}E{\tau \sim \pi_{\theta}(\tau)}[\sum_{t}r(s_{t}, a_{t})]$

### Flow
1. sample $\tau$ from environment
2. $\nabla_{\theta}J(\theta) \approx \sum_{i}\sum_{t}\nabla_{\theta}log\pi_{\theta}(a_{t}^{i}, t_{t}^{i})\sum_{t}r(s_{t}, a_{t})$
3. $\theta &larr; \theta + \alpha\nabla_{\theta}J(\theta)$
4. iterate


### References:
- http://karpathy.github.io/2016/05/31/rl/
- https://gymnasium.farama.org/index.html
- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- cs182 in https://www.youtube.com/@rail7462/playlists
