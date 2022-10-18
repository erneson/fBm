# Intro

Brownian motion in one variable constitutes the simplest random fractal, and also it is at the heart of all the following generalizations. Small particles of solid
matter suspended in a liquid can be seen under a microscope to move about in an irregular and erratic way. This was observed by the botanist R. Brown around 1827.
The modeling of this movement is one of the great topics of statistical mechanics. Occasionally Brownian motion is also referred to as "brown noise". In one dimension
we obtain a random process $X(t)$ , i.e. a function $X$ of a real variable $t$ (time) whose values are random variables $X(t_l)$, $X(t_2)$, etc.

# 1. Fractional Brownian motion: Approximation by spectral synthesis

## 1.1 The spectral representation of random functions

The spectral synthesis method (also known as the Fourier filtering method) for generating fBm is based on the spectral representation of samples of the process
$X(t)$. Since the Fourier transform of $X$ generally is undefined we first restrict $X(t)$ to a finite time interval, say $0 < t < T$ :

$$ \begin{equation}
    X(t, T) =
     \begin{cases}
       X(t) & \text{if 0 < $t$ < $T$}\\
       0 & \text{otherwise}       
     \end{cases}       
\end{equation} $$

We thus have

$$ X(t, T) = \int^{\infty}_{-\infty} F(f, T)e^{2\pi itf}df $$

where $F(f, T)$ is the Fourier transform of $X(t, T)$

$$ F(f, T) = \int^{T}_{0} X(t)e^{-2\pi ift}dt $$

Now $|F(f,T)|^{2}df$ is the contribution to the total energy of $X(t,T)$ from those components with frequencies between $f$ and $f + df$. The average power of $X$
contained in the interval $[0, T]$ is then given by

$$ \frac{1}{T} \int^{\infty}_{-\infty} |F(f,T)|^{2}df $$

and the power spectral density of $X(t,T)$ is

$$ S(f,T) = \frac{1}{T}|F(f,T)|^2 $$

The spectral density of $X$ is then obtained in the limit as $T \to \infty$

$$ S(f) = \lim_{T\to\infty} \frac{1}{T}|F(f,T)|^2 $$

The interpretation of $S(f)$ is the following: $S(f)df$ is the average of the contribution to the total power from components in $X(t)$ with frequencies between
$f$ and $f + df$. $S(f)$ is a nonnegative and even function. We may think of $X(t)$ as being decomposed into a sum of infinitely many sine and cosine terms of 
frequencies $f$ whose powers (and amplitudes) are determined by the spectral density $S(f)$.

## 1.2 The spectral exponent $\beta$ in fractional Brownian motion

The underlying idea of spectral synthesis is that a prescription of the right kind of spectral density $S(f)$ will give rise to fBm with an exponent 0 < $H$ < 1.
If the random function $X(t)$ contains equal power for all frequencies $f$. this process is called white noise in analogy with the white light made up of radiations
of all wave lengths. If $S(f)$ is proportional to 1/ $f^2$ we obtain the usual brown noise or Brownian motion. In general, a process $X(t)$ with a spectral density
proportional to 1/ $f^{\beta}$ corresponds to fBm with $H = \frac{\beta -1}{2}$:

Equation 1.1

$$ S(f) \propto \frac{1}{f^\beta} \sim \text{fBm with } \beta = 2H + 1 $$ 

Choosing $\beta$ between 1 and 3 will generate a graph of fBm with a fractal dimension of

$$ D_f = 2 - H = \frac{5-\beta}{2} $$

Let us pause for a moment to explain this relationship between $\beta$ and $H$. Mathematically it can be derived from the fact that the mean square increments
(which are proportional to $\Delta t^{2H}$ for fBm with exponent $H$) are directly related to the autocorrelation function of $X$, which in turn defines the spectral
density by means of a Fourier transform via the Wiener-Khintchine relation. In place of this "pure" approach we propose a simple and more heuristic argument for
the relation $\beta = 2H + 1$. We start out by restating the fundamental property of fBm: If $X(t)$ denotes fBm with exponent 0 < $H$ < 1 then the properly
rescaled random function

$$ Y(t) = \frac{1}{r^H}X(rt) $$

for a given $r > 0$ has the same statistical properties as $X$. Thus it also has the same spectral density. From this basic observation we can deduce the important
result (1.1) using only the above definitions and some elementary calculus as follows.

Let us fix some $r>0$, set

$$ \begin{equation}
    Y(t, T) =
     \begin{cases}
       Y(t)=\frac{1}{r^H}X(rt), & \text{if 0 < $t$ < $T$}\\
       0, & \text{otherwise}       
     \end{cases}       
\end{equation} $$

and adopt the notation 

$$ \begin{matrix} 
F_{X}(t,T), F_{Y}(t,T) & \text{Fourier transforms of $X(t,T), Y(t,T)$,}\\
S_{X}(f,T), S_{Y}(f,T) & \text{spectral densities of $X(t,T), Y(t,T)$,}\\
S_{X}(f), S_{Y}(f) & \text{spectral densities of $X(t), Y(t)$,}     
\end{matrix} $$

We compute

$$ F_{Y}(f,T) = \int_{0}^{T} Y(t)e^{-2\pi ift}dt = \frac{1}{r^H} \int_{0}^{rT} X(s)e^{-2\pi i\frac{f}{r}s}\frac{ds}{r} $$

where we have substituted $\frac{s}{r}$ for $t$ and $\frac{ds}{r}$ for $dt$ in the second integral. Thus, clearly

$$ F_{Y}(f,T) = \frac{1}{r^{H+1}}F_{X}(\frac{f}{r}, rT) $$

Now it follows for the spectral density of $Y(t,T)$

$$ S_{Y}(f,T) = \frac{1}{r^{2H+1}} \frac{1}{rT} |F_{X}(\frac{f}{r}, rT)|^2 $$

and in the limit as $T \to \infty$ or equivalently as $rT \to \infty$ we conclude

$$ S_{Y}(f) = \frac{1}{r^{2H+1}} S_{X}(\frac{f}{r}) $$

Since $Y$ is just a properly rescaled version of $X$, their spectral densities must coincide, thus, also

$$ S_{X}(f) = \frac{1}{r^{2H+1}} S_{X}(\frac{f}{r}) $$

Now we formally set $f = 1$ and replace 1/ $r$ again by $f$ to finally obtain the desired result (1.1)

$$ S_{X}(f) \propto \frac{1}{f^{2H+1}} = \frac{1}{f^{\beta}} $$

## 1.3 The Fourier filtering method | one-dimension

For the practical algorithm we have to translate the above into conditions on the coefficients $a_{k}$ of the discrete fourier transform

Equation 1.2

$$ \overline{X}(t) = \sum_{k=0}^{N-1} a_{k}e^{2\pi ikt} $$ 

The coefficients $a_{k}$ are in a 1:1 correspondence with the complex values $\overline{X}(t_k), t_k = \frac{k}{N}, k = 0, 1, ..., N - 1$. The condition to be imposed on the coefficients in order to obtain $S(f) \propto \frac{1}{f^\beta}$ now becomes

Equation 1.3

$$ E(|a_k|^2) \propto \frac{1}{k^\beta} $$

since $k$ essentially denotes the frequency in (1.2). This relation (1.3) holds for $0 < k < \frac{N}{2}$. For $k\geqslant \frac{N}{2}$ we must have $a_k = \overline{a_{N-k}}$
because $\overline{X}$ is a real function. The method thus simply consists of randomly choosing coefficients subject to the expectation in (1.3) and then computing the inverse
Fourier transform to obtain $X$ in the time domain. Since the process $X$ need only have real values it is actually sufficient to sample real random variables $A_k$ and $B_k$
under the constraint

$$ E(A^2_k + B^2_k) \propto \frac{1}{k^\beta} $$ 

and then set

$$ \overline{X}(t) = \sum_{k=1}^{N/2} (A_{k} \cos kt + B_k \sin kt) $$

In contrast to some of the previous algorithms discussed this method is not recursive and also does not proceed in stages of increasing spatial resolution. We may, however,
interpret the addition of more and more random Fourier coefficients $a_{k}$ satisfying (1.3) as a process of adding higher frequencies, thus, increasing the resolution in the
frequency domain.

The advantage of this straight forward method is that it is the purest interpretation of the concept of fractional Brownian motion. Artifacts such as those occurring in the midpoint displacement methods are not apparent. However, due to the nature of Fourier transforms, the generated samples are periodic. This is sometimes annoying, and in this case one can compute twice or four times as many points as actually needed and then discard a part of the sequence.

# 2. Extensions to higher dimensions

## 2.1 Definitions

In this section we discuss how one can generalize the displacement methods and the spectral synthesis methods to two and three dimensions. The generalization of fractional Brownian motion itself is straight forward. It is a multidimensional process (a random field) $X(t_{1}, t_{2}, ..., t_{n})$ with the properties:

  1. The increments $X(t_{1}, t_{2}, ..., t_{n}) - X(s_{1}, s_{2}, ..., s_{n})$ are Gaussian with mean 0

  2. The variance of the increments $X(t_{1}, t_{2}, ..., t_{n}) - X(s_{1}, s_{2}, ..., s_{n})$ depends only on the distance

$$ \sqrt{\sum_{i=1}^{n} (t_{i} - s_{i})^2} $$

and in fact is proportional to the 2H-th power of the distance, where the parameter $H$ again satisfies $0 < H < 1$. Thus,

Equation 2.1

$$ E(|X(t_{1}, t_{2}, ..., t_{n}) - X(s_{1}, s_{2}, ..., s_{n})|^2) \propto (\sum_{i=1}^{n} (t_{i} - s_{i})^2)^H $$ 

The random field $X$ again has stationary increments and is isotropic, i.e. all points $(t_{1}, t_{2}, ..., t_{n})$ and all directions are statistically equivalent. In the
frequency domain we have for the spectral density

Equation 2.2

$$ S(f_{1}, ..., f_{n}) \propto \frac{1}{(\sqrt{\sum_{i=1}^{n} f_{i}^{2}})^{2H + n}} $$

This fact can be deduced in the exact same fashion as in the last section for the $\frac{1}{f^\beta}$ law in the one-dimensional case. Therefore we skip these details here. This ensures that $X$ restricted to any straight line will be a $\frac{1}{f^\beta}$ noise corresponding to $2H = \beta - 1$. In analogy with formula (1.1) the fractal dimension of the
graph of a sample of $X(t_{1}, t_{2}, ..., t_{n})$ is

Equation 2.3

$$ D = n + 1- H $$

## 2.2 The Fourier filtering method | two-dimension

We now proceed to the extension of the Fourier filtering method to higher dimensions. In two dimensions the spectral density $S$ generally will depend on two frequency variables
$u$ and $\upsilon$ corresponding to the $x$ and $y$ directions. But since all directions in the $xy$-plane are equivalent with respect to statistical properties, $S$ depends only
on $\sqrt{u^{2} + v^{2}}$ . If we cut the surface along a straight line in the $xy$-plane, we expect for the spectral density $S$ of this fBm in only one dimension a power law 
1/ $f^\beta$ as before. This requirement implies (see Equation 2.2) for the two-dimensional spectral density to behave like

$$ S(u,\upsilon) = \frac{1}{(u^{2}+\upsilon^{2})^{H+1}} $$

The two-dimensional discrete inverse Fourier transform is

$$ X(x, y) = \sum_{k=0}^{N-1} \sum_{l=0}^{N-1} a_{kl}e^{2\pi i(kx+ly)} $$

for $x, y = 0, \frac{1}{N}, \frac{2}{N}, ..., \frac{N-1}{N}$, and thus, we specify for the coefficients $a_{kl}$

$$ E(|a_{kl}|^2) \propto \frac{1}{(k^{2} + l^{2})^{H+1}} $$

Since the constructed function $X$ is a real function we must also satisfy a conjugate symmetry condition, namely

$$ \begin{matrix} 
a_{N-i, N-j} = \overline{a_{i,j}} & \text{for $i, j > 0$}\\
a_{0, N-j} = \overline{a_{0,j}} & \text{for $j > 0$}\\
a_{N-i, 0} = \overline{a_{i,0}} & \text{for $i > 0$}\\
a_{0, 0} = \overline{a_{0,0}}     
\end{matrix} $$

The fractal dimension $D_f$ of the surface will be $D_f = 3 - H$. The algorithm then consists simply of choosing the Fourier coefficients accordingly and then performing a two-dimensional inverse transform.

In the sequence of Figure 1 and 2 we show how the spectral synthesis method "adds detail" by improving the spectral representation of the random fractal, i.e. by allowing more and more Fourier coefficients to be employed. The resolution in the algorithm *SpectralSynthesisFM2D()* was $N$ = 64 but in the top image of Figure 1 only the first $2^{2} = 4$ coefficients were used. In the other pictures we allowed  $4^{2} = 16$ (middle) and  $8^{2} = 64$ (bottom) non-zero coefficients. In Figure 2 we increase the number of Fourier coefficients to 
$16^{2}$ (top), $32^{2}$ (middle) and $64^{2}$ (bottom).

Figure 1
![1](/images/spectral_synthesis_of_a_mountain.png)


Figure 2
![2](/images/spectral_synthesis_of_a_mountain2.png)

Source: The Science of Fractal Images, edited by H. Peitgen and D. Saupe (Springer, New York, 1988).