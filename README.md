
#### Sample Result

Basic system is defined as

$$
\dot{x}(t) = A x(t) + B u(t),
$$

where $A$ is stable, symmetric and metlzer, B is positive

The distance function is defined as 

$$
d(i,j) = \frac{|x_i - x_j|}{a_{ij}}
$$

The clustering cost function is defined as 

$$
Q = \frac{\sum_{i,j} g(A_{ij}) \cdot \delta(C(i), C(j))}{\sum_{i,j} g(A_{ij})}
$$

where $g: A \rightarrow \mathbb{R}_+$ is a monotonic function encourages nodes share smaller weights to be grouped in a cluster, so we also assume $g' < 0$. $\delta(C(i), C(j)) = 1$ if nodes $i$ and $j$ are in the same community; 0 otherwise.

In this particular experiment, the cost function is taken as $g(d) = 1/d$.

Transfer function is defined as

$$
f(s) := (sI_n - A)^{-1} b
$$

Error is computed as follows

$$
|f(s)-f^{(k)}(s)|_{H_\infty}  = f(0) - f^{(k)}(0)
$$

![Error_log](https://github.com/user-attachments/assets/59188097-9e94-4a9c-baba-16f4c8ca5ef5)
