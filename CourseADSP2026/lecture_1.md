# Modern Digital Signal Processing
## Chapter 1: Discrete-Time Signal Processing — Undergraduate Review

---

## Table of Contents
1. [Digital Signals and DSP Overview](#1-digital-signals-and-dsp-overview)
2. [Filter Structures and Design](#2-filter-structures-and-design)
3. [Transforms for Discrete-Time Signals](#3-transforms-for-discrete-time-signals)
4. [Special Sequences and Corresponding Filters](#4-special-sequences-and-corresponding-filters)

---

## Course Overview

This course extends undergraduate DSP to cover stochastic signals and advanced processing methods. The main topics are:

- Foundations of discrete-time signal processing *(this chapter — review)*
- Discrete random signal analysis (orthogonal transforms, parameter estimation)
- Linear prediction and lattice filters
- Linear modeling of random signals
- Power spectral estimation
- Optimal linear filtering: Wiener and Kalman filters
- Adaptive filters

Optional/supplementary topics: multirate DSP, time-frequency analysis, spatial/MIMO processing, blind signal processing.

---

## 1. Digital Signals and DSP Overview

### 1.1 What is a Digital Signal?

A **digital signal** is a discrete-time, discrete-amplitude representation of a physical quantity. Common examples include:

- **Speech signals** — continuous pressure waves sampled at 8–44.1 kHz
- **Image signals** — 2D arrays of pixel intensities
- **Communication signals** — baseband sequences such as MPSK and M-QAM constellations (e.g., QPSK, 8-PSK, 16-QAM)
- **Radar signals** — pulsed or frequency-diverse waveforms

A **digital signal** is a discrete-time, discrete-amplitude representation of a physical quantity. Three representative examples are shown below:

> **[PLACEHOLDER: Speech signal waveform]**
> *Figure 1.1: A speech signal — continuous pressure waves sampled at a fixed rate,
> shown as a discrete sequence in the time domain*

> **[PLACEHOLDER: MPSK and M-QAM constellation diagrams]**
> *Figure 1.2: Baseband communication signals — QPSK (M=4), 8-PSK (M=8), and
> 16-QAM constellation maps. Each point represents a symbol carrying K bits.*

> **[PLACEHOLDER: Radar signal waveform (frequency diversity)]**
> *Figure 1.3: A frequency-diverse radar signal — shows how the carrier frequency
> shifts across pulses to improve target discrimination*

Other common digital signals include image signals (2D pixel arrays), audio, and biomedical recordings such as ECG.

### 1.2 Applications of Signal Processing

DSP is foundational to virtually every modern technology domain:

- **Wireless communications** — signal generation, compression, modulation/demodulation, channel estimation, equalization, source/channel coding (e.g., OFDM systems)
- **Multimedia** — audio/video filtering, enhancement, compression, recognition, synthesis
- **Radar and sonar** — filtering, detection, feature extraction, tracking, target recognition
- **Biomedical engineering** — diagnostic monitoring, remote medicine
- **Geophysics and meteorology** — seismic recording, weather signal processing
- **Automatic control**

> **[PLACEHOLDER: OFDM system block diagram]**
> *Figure: OFDM transceiver — transmitter chain: encoding → interleaving → modulation → pilot insertion → S/P → IFFT → P/S → CP insertion → DAC → RF TX; receiver chain is symmetric*

### 1.3 General DSP System Pipeline

A continuous-time bandlimited signal $x(t)$ is:
1. **Anti-aliasing filtered** (lowpass filter, cutoff at $f_s/2$)
2. **Sampled** by an A/D converter at rate $f_s \geq 2B$ (Nyquist–Shannon theorem)
3. **Processed** by a discrete-time system with impulse response $h(n)$ via convolution
4. **Reconstructed** by a D/A converter if a continuous output is needed

> **[PLACEHOLDER: Speech digitization diagram showing time-domain and frequency-domain views at each stage]**
> *Figure: Digitization pipeline for speech — input → anti-aliasing LPF → A/D converter → sampled signal, shown in both time and frequency domains*

This gives rise to two fundamental problems that form the backbone of DSP:

1. **Describing discrete-time systems**: difference equations, time-domain responses, frequency-domain responses, structural realizations → **Filters**
2. **Describing and analyzing discrete-time signals efficiently**: z-transform, DTFT, DFT/FFT → **Transforms**

---

## 2. Filter Structures and Design

### 2.1 FIR Filter Implementations

An FIR (Finite Impulse Response) filter of order $M$ has transfer function:

$$H(z) = \sum_{n=0}^{M} h(n) z^{-n}$$

with frequency response:

$$H(e^{j\omega}) = \sum_{n=0}^{M} h(n) e^{-j\omega n}$$

#### 2.1.1 Direct Form (Transversal Filter)

The output is a weighted sum of the current and past $M$ inputs:

$$y(n) = \sum_{m=0}^{M} h(m)\, x(n - m)$$

This is implemented as a tapped delay line. Computationally straightforward but requires $M+1$ multiplications per output sample.

> **[PLACEHOLDER: Transversal (direct form) FIR filter signal flow graph]**
> *Figure: Direct-form FIR — shift register of $z^{-1}$ delays, each output tapped and multiplied by $h[m]$, summed to produce $y[n]$*

#### 2.1.2 Cascade (Factored) Form

$H(z)$ is factored into second-order sections (biquads):

$$H(z) = \prod_{k=1}^{M_s} H_k(z), \quad H_k(z) = b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}$$

Each section is implemented as a small FIR. This is numerically more robust than the direct form for high-order filters.

> **[PLACEHOLDER: Cascade FIR filter structure diagram]**
> *Figure: Cascade implementation — series of second-order FIR sections*

#### 2.1.3 Recursive (IIR-based) Realization and Comb Filter

A uniform averaging filter (moving average):

$$h(n) = \begin{cases} 1/N & n = 0, 1, \ldots, N-1 \\ 0 & \text{otherwise} \end{cases}$$

has transfer function:

$$H(z) = \frac{1}{N}\sum_{n=0}^{N-1}z^{-n} = \frac{1}{N} \cdot \frac{1 - z^{-N}}{1 - z^{-1}}$$

This can be decomposed as $H(z) = H_1(z) \cdot H_2(z)$ where:
- $H_1(z) = \frac{1-z^{-N}}{N}$ — an FIR **comb filter** (zeros at $N$-th roots of unity)
- $H_2(z) = \frac{1}{1-z^{-1}}$ — a first-order IIR accumulator

The pole-zero cancellation preserves the FIR nature, while the recursive structure reduces computation from $N$ to 2 operations per sample.

#### 2.1.4 Frequency Sampling Form

When the DFT values $H(k)$ are known, $H(z)$ can be written as:

$$H(z) = \frac{1 - z^{-N}}{N} \sum_{k=0}^{N-1} \frac{H(k)}{1 - W_N^{-k} z^{-1}}, \quad W_N = e^{-j2\pi/N}$$

This combines a comb filter ($1 - z^{-N}$) cascaded with a bank of first-order resonators tuned to the DFT frequencies. Useful when the desired frequency response is specified at discrete frequencies.

> **[PLACEHOLDER: Frequency sampling FIR structure diagram]**
> *Figure: Frequency-sampling realization — comb filter feeding a parallel bank of first-order resonators*

---

### 2.2 IIR Filter Implementations

An IIR (Infinite Impulse Response) filter has both poles and zeros:

$$H(z) = \frac{\sum_{k=0}^{N} a_k z^{-k}}{1 + \sum_{k=1}^{N} b_k z^{-k}}$$

#### 2.2.1 Direct Form I

Implements the difference equation directly with two separate delay chains — one for the input $x(n)$ (FIR part) and one for the output $y(n)$ (IIR part). Requires $2N$ delay elements.

> **[PLACEHOLDER: Direct Form I IIR filter signal flow graph]**
> *Figure: Direct Form I — left delay chain for input terms $x(n-k)$, right delay chain for feedback terms $y(n-k)$*

#### 2.2.2 Direct Form II (Canonical Form)

Merges the two delay chains by exploiting linearity. Requires only $N$ delay elements — the minimum possible (canonical form). The intermediate state variable is $u(n)$.

$$u(n) = x(n) - \sum_{k=1}^{N} b_k u(n-k)$$
$$y(n) = \sum_{k=0}^{N} a_k u(n-k)$$

> **[PLACEHOLDER: Direct Form II IIR filter signal flow graph]**
> *Figure: Direct Form II — single shared delay chain, feedback on the left, feedforward on the right*

#### 2.2.3 Cascade Form

$H(z)$ is factored into second-order sections:

$$H(z) = \prod_{k=1}^{N/2} \frac{a_{0k} + a_{1k}z^{-1} + a_{2k}z^{-2}}{1 + b_{1k}z^{-1} + b_{2k}z^{-2}}$$

Each biquad is implemented as a Direct Form II section. Cascade form is numerically superior to high-order direct forms because coefficient sensitivity is localized.

> **[PLACEHOLDER: IIR cascade form signal flow graph]**
> *Figure: IIR cascade — series of biquad (second-order) sections*

#### 2.2.4 Parallel Form

$H(z)$ is expanded via partial fractions:

$$H(z) = C + \sum_{k=1}^{N/2} \frac{a_{0k} + a_{1k}z^{-1}}{1 + b_{1k}z^{-1} + b_{2k}z^{-2}}$$

The parallel form generally has the best numerical performance because the poles of each section are independent, and round-off errors do not accumulate.

> **[PLACEHOLDER: IIR parallel form signal flow graph]**
> *Figure: Parallel IIR — constant $C$ plus parallel bank of second-order sections*

---

### 2.3 Filter Design

The goal of filter design is to find coefficients $h(n)$ (FIR) or $\{a_k, b_k\}$ (IIR) such that the frequency response $|H(e^{j\omega})|$ approximates a desired response $|H_d(\nu)|$.

Common ideal magnitude responses:
- **(a) Lowpass**: passband $[0, \nu_1]$, stopband $[\nu_1, 1]$
- **(b) Highpass**: stopband $[0, \nu_1]$, passband $[\nu_1, 1]$
- **(c) Bandpass**: passband $[\nu_1, \nu_2]$
- **(d) Bandstop**: stopband $[\nu_1, \nu_2]$

> **[PLACEHOLDER: Four ideal filter magnitude response plots (lowpass, highpass, bandpass, bandstop)]**
> *Figure: Ideal filter shapes — (a) lowpass, (b) highpass, (c) bandpass, (d) bandstop*

**Practical filters** have transition bands and ripple:

- Passband ripple $\delta_1$: $1 - \delta_1 \leq |H(e^{j\omega})| \leq 1 + \delta_1$ for $\omega \leq \omega_c$
- Stopband attenuation $\delta_2$: $|H(e^{j\omega})| \leq \delta_2$ for $\omega \geq \omega_{st}$

> **[PLACEHOLDER: Practical filter frequency response with passband ripple and stopband attenuation labeled]**
> *Figure: Practical filter spec — passband, transition band, stopband, with ripple tolerances $\delta_1$ and $\delta_2$*

A rough estimate of required filter order for equiripple FIR design:

$$N \approx \frac{-20\log_{10}\sqrt{\delta_1 \delta_2} - 13}{14.6 \cdot F_{st}}$$

where $F_{st} = (\omega_{st} - \omega_c)/(2\pi)$ is the normalized transition bandwidth.

#### 2.3.1 FIR Filter Design Methods

1. **Linear phase property**: FIR filters can have *exactly* linear phase (constant group delay), which is critical for applications like audio and data communications. This requires $h(n)$ to be either **even-symmetric** ($h(n) = h(N-1-n)$) or **odd-symmetric** ($h(n) = -h(N-1-n)$).

2. **Window method**: Start with the ideal (infinite-length) impulse response $h_d(n)$, then truncate using a window $w(n)$:
   $$h(n) = w(n) \cdot h_d(n)$$
   Window choice (rectangular, Hann, Hamming, Blackman, Kaiser) trades off transition width vs. stopband attenuation.

3. **Frequency sampling method**: Specify the desired response at $N$ DFT frequencies, compute the IDFT to get $h(n)$. Transition band samples can be optimized to reduce ripple.

4. **Chebyshev (equiripple) approximation**: Minimize the maximum error $\|H(e^{j\omega}) - H_d(e^{j\omega})\|_\infty$ over specified frequency bands. The Parks-McClellan algorithm (based on the Remez exchange algorithm) gives the optimal solution. Requires computer-aided design tools.

#### 2.3.2 IIR Filter Design Methods

1. **Analog prototype design**: Design a classical analog filter (Butterworth, Chebyshev I/II, Elliptic) in the $s$-domain.
2. **Impulse invariance method**: Map the analog prototype to digital by matching the impulse response at sample points. Prone to aliasing; only suitable for lowpass/bandpass designs.
3. **Bilinear z-transform method**: Map $s \to \frac{2}{T}\frac{1-z^{-1}}{1+z^{-1}}$. No aliasing, but introduces nonlinear frequency warping — pre-warp the critical frequencies to compensate.
4. **Frequency transformation method**: Design a digital lowpass prototype, then apply a digital frequency transformation to obtain highpass, bandpass, or bandstop designs.

---

## 3. Transforms for Discrete-Time Signals

### 3.1 DTFT and z-Transform

The **Discrete-Time Fourier Transform (DTFT)** generalizes the continuous Fourier transform to discrete sequences:

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x(n)\, e^{-j\omega n}$$

$$x(n) = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega})\, e^{j\omega n}\, d\omega$$

The **z-Transform** is the generalization to the complex plane:

$$X(z) = \sum_{n=-\infty}^{\infty} x(n)\, z^{-n}$$

**Key relationship**: The DTFT is the z-transform evaluated on the unit circle:

$$X(e^{j\omega}) = X(z)\big|_{z = e^{j\omega}}$$

This is valid only when the ROC (Region of Convergence) of $X(z)$ includes the unit circle.

#### DTFT Properties

| Property | Time Domain | ↔ | Frequency Domain |
|---|---|---|---|
| Linearity | $ax(n) + by(n)$ | ↔ | $aX(e^{j\omega}) + bY(e^{j\omega})$ |
| Time shift | $x(n - n_0)$ | ↔ | $e^{-j\omega n_0} X(e^{j\omega})$ |
| Frequency shift | $e^{j\omega_0 n} x(n)$ | ↔ | $X(e^{j(\omega - \omega_0)})$ |
| Modulation | $x(n)\cos\omega_0 n$ | ↔ | $\frac{1}{2}[X(e^{j(\omega-\omega_0)}) + X(e^{j(\omega+\omega_0)})]$ |
| Conjugate | $x^*(n)$ | ↔ | $X^*(e^{-j\omega})$ |
| Time reversal | $x(-n)$ | ↔ | $X(e^{-j\omega})$ |
| Convolution | $x(n) * y(n)$ | ↔ | $X(e^{j\omega})Y(e^{j\omega})$ |
| Correlation | $\sum_k x(k)y(n+k)$ | ↔ | $X(e^{j\omega})Y(e^{-j\omega})$ |
| Multiplication | $x(n)y(n)$ | ↔ | $\frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\theta})Y(e^{-j(\omega-\theta)})d\theta$ |
| Freq. differentiation | $nx(n)$ | ↔ | $j\frac{d}{d\omega}X(e^{j\omega})$ |
| Parseval's theorem | $\sum_{n=-\infty}^{\infty} x(n)y^*(n)$ | $=$ | $\frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega})Y^*(e^{j\omega})d\omega$ |

#### z-Transform Properties

| Property | Time Sequence | ↔ | z-Transform | ROC |
|---|---|---|---|---|
| Linearity | $ax(n)+by(n)$ | ↔ | $aX(z)+bY(z)$ | At least $\text{ROC}_x \cap \text{ROC}_y$ |
| Time shift | $x(n-K)$ | ↔ | $z^{-K}X(z)$ | $\text{ROC}_x$ (excl. $z=0$ if $K>0$) |
| z-domain scaling | $\alpha^n x(n)$ | ↔ | $X(z/\alpha)$ | $|\alpha|r_1 < |z| < |\alpha|r_2$ |
| Conjugate | $x^*(n)$ | ↔ | $X^*(z^*)$ | $\text{ROC}_x$ |
| Time reversal | $x(-n)$ | ↔ | $X(1/z)$ | $1/r_2 < |z| < 1/r_1$ |
| Convolution | $x(n)*y(n)$ | ↔ | $X(z)Y(z)$ | At least $\text{ROC}_x \cap \text{ROC}_y$ |
| Correlation | $x(n)*y(-n)$ | ↔ | $X(z)Y(z^{-1})$ | — |
| z-domain diff. | $nx(n)$ | ↔ | $-z\frac{d}{dz}X(z)$ | $\text{ROC}_x$ |
| Initial value | $x(0)$ (causal) | — | $\lim_{z\to\infty} X(z)$ | — |
| Parseval | $\sum x(n)y^*(n)$ | $=$ | $\frac{1}{2\pi j}\oint X(v)Y^*(1/v^*)v^{-1}dv$ | — |

#### Common z-Transform Pairs

| Sequence | z-Transform | ROC |
|---|---|---|
| $\delta(n)$ | $1$ | All $z$ |
| $\alpha^n u(n)$ | $\dfrac{1}{1-\alpha z^{-1}}$ | $\lvert z\rvert > \lvert\alpha\rvert$ |
| $-\alpha^n u(-n-1)$ | $\dfrac{1}{1-\alpha z^{-1}}$ | $\lvert z\rvert < \lvert\alpha\rvert$ |
| $\alpha^n u(n) - \alpha^n u(n-N)$ | $\dfrac{1-\alpha^N z^{-N}}{1-\alpha z^{-1}}$ | $\lvert z\rvert > 0$ |
| $\alpha^{\lvert n\rvert}$ | $\dfrac{1-\alpha^2}{(1-\alpha z^{-1})(1-\alpha z)}$ | $\lvert\alpha\rvert < \lvert z\rvert < 1/\lvert\alpha\rvert$ |

---

### 3.2 DFT (Discrete Fourier Transform)

For a finite-length sequence of length $N$, the DTFT is continuous in frequency. The DFT samples it at $N$ equally spaced points on the unit circle:

$$X(k) = \sum_{n=0}^{N-1} x(n) W_N^{nk}, \quad k = 0, 1, \ldots, N-1$$

$$x(n) = \frac{1}{N}\sum_{k=0}^{N-1} X(k) W_N^{-nk}, \quad n = 0, 1, \ldots, N-1$$

where $W_N = e^{-j2\pi/N}$.

**Key points:**
- The DFT is related to the DFS (Discrete Fourier Series) of a periodically extended sequence
- $X(k)$ represents the spectrum sampled at frequencies $\omega_k = 2\pi k/N$
- The DFT is the workhorse of spectral analysis and fast convolution

Key DFT properties include linearity, circular shift, circular convolution (time-domain multiplication ↔ frequency-domain circular convolution), conjugate symmetry for real inputs, and Parseval's theorem.

---

### 3.3 FFT (Fast Fourier Transform)

The naive DFT computation requires $O(N^2)$ operations. The FFT exploits two properties of $W_N$:

1. **Periodicity**: $W_N^{r+N} = W_N^r$
2. **Symmetry**: $W_N^{r+N/2} = -W_N^r$

Using a **divide-and-conquer** strategy, an $N$-point DFT is split into smaller DFTs, reducing complexity to $O(N \log_2 N)$.

#### Radix-2 Decimation-in-Time (DIT) FFT

Split $x(n)$ into **even-indexed** and **odd-indexed** subsequences:

$$X(k) = \sum_{n \text{ even}} x(n)W_N^{nk} + \sum_{n \text{ odd}} x(n)W_N^{nk} = U(k) + W_N^k V(k)$$

$$X(k+N/2) = U(k) - W_N^k V(k)$$

This "butterfly" structure is applied recursively. For $N = 2^m$:
- **Multiplications**: $\frac{N}{2}\log_2 N$
- **Additions**: $N\log_2 N$

For $N = 1024$: naive DFT needs ~$10^6$ multiplications; FFT needs only ~$5000$.

> **[PLACEHOLDER: N=8 DIT-FFT butterfly diagram (first decomposition stage)]**
> *Figure: 8-point DIT FFT — input split into even/odd groups, each fed to a 4-point DFT, combined via butterfly operations with twiddle factors $W_8^k$*

#### Radix-2 Decimation-in-Frequency (DIF) FFT

Split $x(n)$ into **first half** and **second half**:

$$X(2k) = \sum_{n=0}^{N/2-1}[x(n) + x(n+N/2)]W_{N/2}^{nk}$$
$$X(2k+1) = \sum_{n=0}^{N/2-1}[x(n) - x(n+N/2)]W_N^n W_{N/2}^{nk}$$

#### Bit-Reversal Permutation

In DIT-FFT, the input must be reordered in **bit-reversed** order before the butterfly stages (or equivalently, the output in DIF-FFT is bit-reversed). For example, with $N=8$: index $6$ (binary `110`) maps to index $3$ (binary `011`).

#### FFT Variants

- **Radix-4 FFT**: Groups of 4; reduces multiplicative count further
- **Split-radix FFT**: Best known operation count for power-of-2 $N$
- **Mixed-radix FFT**: Handles arbitrary $N$ by factoring $N = N_1 \times N_2 \times \cdots$

#### FFT for Spectral Analysis

The DFT/FFT maps a time-domain sequence to its frequency content. Practical considerations:
- **Frequency resolution**: $\Delta f = f_s / N$ — more points → finer resolution
- **Spectral leakage**: Caused by finite observation window; mitigated by windowing
- **Zero-padding**: Increases DFT size beyond the signal length for interpolated display (does not add true resolution)

---

## 4. Special Sequences and Corresponding Filters

### 4.1 Allpass Sequences and Allpass Filters

An **allpass filter** has unit magnitude response at all frequencies:

$$|H_{ap}(e^{j\omega})| = 1 \quad \forall\, \omega$$

This implies:

$$H_{ap}(e^{j\omega}) H_{ap}^*(e^{j\omega}) = 1$$
$$h_{ap}(n) * h_{ap}^*(-n) = \delta(n)$$
$$H_{ap}(z) H_{ap}^*(1/z^*) = 1$$

**Pole-zero structure**: For a rational allpass filter, every pole at $z = c$ is paired with a zero at $z = 1/c^*$ (its conjugate reciprocal). Poles inside the unit circle have corresponding zeros outside, and vice versa.

**First-order allpass**:

$$H_{ap1}(z) = \frac{z^{-1} - c^*}{1 - c z^{-1}}$$

**General $N$-th order allpass**:

$$H_{ap2}(z) = \prod_{k=1}^{N} \frac{z^{-1} - \alpha_k^*}{1 - \alpha_k z^{-1}}$$

**General form** (with $A(z) = 1 + a_1 z^{-1} + \cdots + a_N z^{-N}$):

$$H_{ap}(z) = z^M \frac{A^*(1/z^*)}{z^{N-M} A(z)} = \frac{z^{-N} + a_1^* z^{-N+1} + \cdots + a_N^*}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}$$

The numerator is the "conjugate-reversed" polynomial of the denominator.

**Use case**: Allpass filters are used for phase equalization (correcting phase distortion while preserving magnitude) and as building blocks in lattice filters and filter banks.

> **[PLACEHOLDER: Pole-zero diagram for first-order allpass filter]**
> *Figure: Allpass pole at $z=c$ (inside unit circle) with zero at $z=1/c^*$ (outside unit circle)*

---

### 4.2 Minimum-Phase Sequences and Filters

A causal, stable filter $H_m(z)$ is **minimum-phase** if and only if:

1. It is **stable** (ROC includes the unit circle)
2. It is **causal** (right-sided sequence)
3. **All zeros are inside or on the unit circle**

Among all stable, causal filters with the same magnitude response, the minimum-phase filter has the smallest phase lag at every frequency and the fastest energy buildup in its impulse response.

**Key Theorem (Minimum-Phase/Allpass Decomposition)**:

> *Any stable, non-minimum-phase filter $H(z)$ can be uniquely decomposed as:*
> $$H(z) = H_{ap}(z) \cdot H_m(z)$$
> *where $H_m(z)$ is minimum-phase and $H_{ap}(z)$ is allpass.*

**Construction**: 
- All poles and zeros inside or on the unit circle → assign to $H_m(z)$
- For each zero at $z = c$ outside the unit circle: add a zero at $z = 1/c^*$ to $H_m(z)$ (moving it inside), and add a corresponding allpass factor with pole at $z = 1/c^*$ and zero at $z = c$ to $H_{ap}(z)$

> **[PLACEHOLDER: Pole-zero diagram showing decomposition H(z) = H_ap(z) × H_m(z)]**
> *Figure: Decomposition — original H(z) with zeros inside and outside unit circle → allpass H_ap(z) plus minimum-phase H_m(z)*

---

### 4.3 Linear-Phase Sequences and Filters

An FIR filter has **generalized linear phase** if:

$$H(e^{j\omega}) = e^{j\beta} e^{-j\alpha\omega} A(e^{j\omega})$$

where $A(e^{j\omega})$ is real-valued. This requires the impulse response to satisfy a symmetry condition.

The z-transform satisfies:

$$H(z) = \pm z^{-N} H(1/z)$$

**Strict linear phase** (real $h(n)$): even symmetry $h(n) = h(N-1-n)$.

#### Four Types of Linear-Phase FIR Filters

Classified by symmetry type and whether $N$ (length) is odd or even:

| Type | Symmetry | Length $N+1$ | Phase | $H(e^{j\omega})$ form |
|------|----------|----------|-------|-----------------------|
| **I** | Even ($h(n)=h(N{-}1{-}n)$) | **Odd** | $\varphi(\omega) = -\omega\frac{N-1}{2}$ | $\sum_{n=0}^{(N-1)/2} a(n)\cos n\omega$ |
| **II** | Even | **Even** | $\varphi(\omega) = -\omega\frac{N-1}{2}$ | $\sum_{n=1}^{N/2} b(n)\cos[(n-\frac{1}{2})\omega]$ |
| **III** | Odd ($h(n)=-h(N{-}1{-}n)$) | **Odd** | $\varphi(\omega) = -\omega\frac{N-1}{2} - \frac{\pi}{2}$ | $\sum_{n=1}^{(N-1)/2} c(n)\sin n\omega$ |
| **IV** | Odd | **Even** | $\varphi(\omega) = -\omega\frac{N-1}{2} - \frac{\pi}{2}$ | $\sum_{n=1}^{N/2} d(n)\sin[(n-\frac{1}{2})\omega]$ |

> **[PLACEHOLDER: Four-panel figure showing h(n), a/b/c/d(n), and H(ω) for each of the four linear-phase FIR types]**
> *Figure: All four types of linear-phase FIR filters — impulse response symmetry, reduced coefficients, and magnitude response shape*

#### Applicable Filter Types per Class

| Type | Lowpass | Highpass | Bandpass | Bandstop |
|------|---------|----------|----------|----------|
| I | ✓ | ✓ | ✓ | ✓ |
| II | ✓ | ✗ | ✓ | ✗ |
| III | ✗ | ✗ | ✓ | ✗ |
| IV | ✗ | ✓ | ✓ | ✗ |

**Explanation of constraints**:
- **Type II**: forced zero at $\omega = \pi$ (i.e., $z = -1$) due to the half-sample delay structure → cannot be highpass or bandstop
- **Type III**: forced zeros at both $\omega = 0$ and $\omega = \pi$ → only useful for bandpass (e.g., differentiators, Hilbert transformers)
- **Type IV**: forced zero at $\omega = 0$ → cannot be lowpass or bandstop; useful for highpass and Hilbert transformers

---

### 4.4 Conjugate-Symmetric and Positive Semi-Definite Sequences

A sequence $r(n)$ is **Hermitian (conjugate-symmetric)** if:

$$r(n) = r^*(-n)$$

Its DTFT $R(e^{j\omega})$ is **real-valued**:

$$R(e^{j\omega}) = R^*(e^{j\omega}) \in \mathbb{R}$$

Its z-transform satisfies:

$$R(z) = R^*(1/z^*)$$

A Hermitian sequence is **positive semi-definite** if additionally $R(e^{j\omega}) \geq 0$ for all $\omega$. Such sequences are precisely **autocorrelation sequences** — they arise as $r(n) = \sum_k x(k) x^*(k-n)$ for some signal $x(n)$.

**Spectral factorization**: If $R(z)$ is a rational positive semi-definite sequence, it can be factored as:

$$R(z) = H_m(z) H_m^*(1/z^*)$$

where $H_m(z)$ is minimum-phase. On the unit circle:

$$R(e^{j\omega}) = |H_m(e^{j\omega})|^2$$

**Theorem (Zero pairing for positive semi-definite sequences)**:
If $r(n)$ is positive semi-definite with rational z-transform, then zeros on the unit circle occur in **conjugate pairs** (even multiplicity), and zeros off the unit circle occur in **quadruples** $\{z_0,\, z_0^*,\, 1/z_0,\, 1/z_0^*\}$.

This spectral factorization theorem is fundamental to Wiener filter theory, linear prediction, and power spectral estimation covered in later chapters.

---

## Summary: Chapter 1 Key Results

| Concept | Key Property | Application |
|---|---|---|
| FIR direct form | $y(n) = \sum_m h(m)x(n-m)$ | Simple, always stable |
| IIR Direct Form II | Canonical (minimum delays) | Efficient implementation |
| IIR Parallel Form | Best numerical properties | High-order stable filters |
| DTFT | Spectrum on unit circle | Frequency analysis |
| z-Transform | Generalizes DTFT to complex plane | System analysis, filter design |
| DFT | Sampled DTFT for finite sequences | Numerical computation |
| FFT | $O(N\log N)$ DFT algorithm | Fast convolution, spectral analysis |
| Allpass filter | $\|H_{ap}(e^{j\omega})\| = 1$ | Phase equalization |
| Minimum-phase | All zeros inside unit circle | Causal inverse, minimum delay |
| Linear-phase FIR | Symmetric $h(n)$ | Distortion-free filtering |
| Positive semi-definite | $R(e^{j\omega}) \geq 0$ | Autocorrelation, spectral factorization |