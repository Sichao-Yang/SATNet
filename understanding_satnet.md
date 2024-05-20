```python
class SATNet(nn.Module):
    '''Apply a SATNet layer to complete the input probabilities.

    Args:
        n: Number of input variables.
        m: Rank of the clause matrix.
        aux: Number of auxiliary variables.

        max_iter: Maximum number of iterations for solving
            the inner optimization problem.
            Default: 40
        eps: The stopping threshold for the inner optimizaiton problem.
            The inner Mixing method will stop when the function decrease
            is less then eps times the initial function decrease.
            Default: 1e-4
        prox_lam: The diagonal increment in the backward linear system
            to make the backward pass more stable.
            Default: 1e-2
        weight_normalize: Set true to perform normlization for init weights.
            Default: True
```

这里的 aux 我理解，就是sat问题里除了原本的variables之外，为了方便求解而定义的变量，在satnet里定义这个主要是为了定义S矩阵的位宽。

kernel function:

```cpp
  // this function is for both Algo2 (forward pass) and 3 (backward pass),
  // in backward pass, U is mapped to V, V is mapped to Vproj (to
  // calculate P), phi is mapped to W, dg is mapped to g
  float delta = 0;
  for (int i, i_ = 0; (i = index[i_]); i_++) {
    const float Sii = Snrms[i];
    const float *__restrict__ Si = S + i * m;

    for (int kk = 0; kk < k; kk++)
      g[kk] = sdot(Si, W + kk * m, m);
    saxpy(g, -Sii, V + i * k, k);

    float gnrmi;
    if (is_forward) {
      // algo2 line7: vo=-go/norm(go)
      gnrmi = snrm2(g, k);
      sscal(g, -1, k);
    } else {
      gnrmi = gnrm[i] + prox_lam;
      float c = sdot(Vproj + i * k, g, k) + dz[i] * Vproj[i * k];
      sscal(g, -1, k);
      saxpy(g, c, Vproj + i * k, k);
      g[0] -= dz[i];
    }
    sscal(g, 1 / gnrmi, k);
```

这一块儿是对应paper里的algorithm 3里的前两个公式，我们来理解下实现上的转换
$$
\begin{aligned}
& \text { compute } \mathrm{d} g_o=\Psi s_o-\left\|s_o\right\|^2 u_o-\partial \ell / \partial v_o  \\
& \text { compute } u_o=-P_o \mathrm{~d} g_o /\left\|g_o\right\|
\end{aligned}
$$
其中需要用到公式：（v的dim：kx1, 所以vvT：kxk）
$$
P_o \equiv I_k-v_o v_o^T \text { for each } o \in \mathcal{O} \text {. }
$$
lets remove forward part and convert to the coresponding symbols:

```cpp
// i is the output index
	const float Sii = Snrms[i];
    const float *__restrict__ Si = S + i * m;
	// dg: kx1, S: nxm, Phi: kxm
    for (int kk = 0; kk < k; kk++)
      dg[kk] = sdot(Si, Phi + kk * m, m);
    saxpy(dg, -Sii, U + i * k, k);
	// v: nxk
  	float c = sdot(V + i * k, dg, k) + dz[i] * V[i * k];
  	sscal(dg, -1, k);
  	saxpy(dg, c, V + i * k, k);
  	dg[0] -= dz[i];
	sscal(dg, 1 / gnrmi, k);
```

我们看到在第一part（5-7）行，我们计算的是：
$$
dg'=\Psi s_o-\left\|s_o\right\|^2 u_o
$$
它只是dg的第一项，$-\partial \ell / \partial v_o $还没有加进来。第13行是一个归一化，没啥重要的，所以我们重点要看9-12行的转换：

首先第九行，$v_o^Ta+dz_o*v_o[0]$，注意这里的后半部分很有意思，因为 $\partial \ell / \partial v_o$ 的公式 $\frac{\partial \ell}{\partial v_o}==\left(\frac{\partial \ell}{\partial z_o}\right) \frac{1}{\pi \sin \left(\pi z_o\right)} v_{\top}$ 实际上是一个相当于一个标量乘上 $v_{\top}$，而我们知道我们定义的 $v_\top$是[1,0,0,0,0] 这样的向量，所以实际上第九行做的就是：
$$
c=v_o^T dg'+v_o^T\frac{\partial \ell}{\partial v_o}
$$
然后第10行就是对a取反，最后第11行是ax+y，a=c，x=v_o, y=-dg'，可得：
$$
(v_o^Tdg'+v_o^T\frac{\partial \ell}{\partial v_o})v_o-dg'
$$
最后再在第0号元素上减去个dz[i]，等价于减去 $\frac{\partial \ell}{\partial v_o}$ 向量: 
$$
(v_o^Tdg'+v_o^T\frac{\partial \ell}{\partial v_o})v_o-dg' - \frac{\partial \ell}{\partial v_o}
$$
我们可以来对比下把原始公式展开后的样子：
$$
u_o=-P_o \mathrm{~d} g_o &= -(I_k-v_o v_o^T)\mathrm{~d} g_o \\
&= -(I_k-v_o v_o^T) (dg'+\frac{\partial \ell}{\partial v_o})\\
&=(v_o v_o^T-I)(dg'+\frac{\partial \ell}{\partial v_o})\\
&=v_o v_o^Tdg'-dg'+v_o v_o^T\frac{\partial \ell}{\partial v_o}-\frac{\partial \ell}{\partial v_o}\\
&=v_o(v_o^Tdg'+v_o^T\frac{\partial \ell}{\partial v_o})-dg'-\frac{\partial \ell}{\partial v_o}
$$
最后一个加减号对不上呢？好奇怪。我检查了下，把cpp里的符号改过来后，也能学，但是loss更大些