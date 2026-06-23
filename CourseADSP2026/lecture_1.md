# Modern Digital Signal Processing
## Chapter 1: Discrete-Time Signal Processing — Undergraduate DSP Review

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005
> Chapters covered: Ch. 1 (Introduction) + Ch. 2 (Fundamentals of Discrete-Time Signal Processing)

## Table of Contents

1. [Part I: Digital Signals and DSP Overview](#part-i-digital-signals-and-dsp-overview)
   - [§1.5 Elementary Sequences (δ, u, αⁿu, e^jω₀n)](#15-elementary-sequences-fundamental-building-blocks)
2. [Part II: Transforms for Discrete-Time Signals](#part-ii-transforms-for-discrete-time-signals)
3. [Part III: Digital Filter Structures and Design](#part-iii-digital-filter-structures-and-design)
4. [Part IV: Special Sequences and Corresponding Filters](#part-iv-special-sequences-and-corresponding-filters)

---

## Notation and Variable Definitions

All symbols used in this chapter are collected below. Where one symbol carries different meanings in different sections, the context is explicitly noted.

### Time, Frequency, and Index Variables

| Symbol | Definition | Unit |
|--------|-----------|------|
| $t$ | Continuous time | s |
| $n$ | Discrete time index ($n \in \mathbb{Z}$) | sample |
| $f$ | Analog cyclic frequency | Hz |
| $f_s = 1/T_s$ | Sampling rate | Hz |
| $f_c$ | Carrier frequency (§1.1.4, §1.3.4) | Hz |
| $f_1,\, f_2$ | Lower/upper band edges of a bandpass signal | Hz |
| $T_s = 1/f_s$ | Sampling period | s/sample |
| $B$ | Signal bandwidth (one-sided baseband) | Hz |
| $\Omega = 2\pi f \in [-\pi/T_s,\,\pi/T_s]$ | Analog angular frequency | rad/s |
| $\omega = \Omega T_s \in [-\pi,\pi]$ | Digital angular frequency | rad/sample |
| $\omega_k = 2\pi k/N$ | $k$-th DFT frequency bin | rad/sample |
| $\omega_c$ | Digital cutoff frequency | rad/sample |
| $\omega_p,\,\omega_s$ | Passband/stopband edge frequencies | rad/sample |
| $\omega_0$ | Fixed oscillation frequency in exponential sequences | rad/sample |
| $\Delta\omega = \omega_s - \omega_p$ | Filter transition bandwidth | rad/sample |
| $k$ | DFT bin index; OFDM subcarrier index; summation index (context determines) | — |
| $l$ | Multipath channel tap index | — |
| $m$ | Image row index (§1.1.2); FIR coefficient index (§3.1.1) | — |
| $q$ | Integer subsampling factor in bandpass sampling theorem (§1.3.3) | — |

### Signals and Sequences

| Symbol | Definition |
|--------|-----------|
| $x_a(t)$ | Continuous-time (analog) input signal |
| $y_a(t)$ | Continuous-time (analog) output signal |
| $p(t) = \sum_{n}\delta(t-nT_s)$ | Ideal Dirac comb (impulse train) used in sampling model |
| $x_s(t) = x_a(t)\cdot p(t)$ | Ideally sampled signal |
| $x(n) = x_a(nT_s)$ | Discrete-time sequence (uniform samples of $x_a$) |
| $y(n)$ | Discrete-time output sequence |
| $h(n)$ | Impulse response of a discrete-time LTI system |
| $h_d(n)$ | Ideal (desired) impulse response in FIR window design |
| $h_{ap}(n)$ | Impulse response of an allpass filter |
| $h_m(n)$ | Impulse response of a minimum-phase filter |
| $r_x(n) = x(n)\ast x^{*}(-n)$ | Autocorrelation sequence of $x(n)$ |
| $\tilde{x}(n) = x(\langle n\rangle_N)$ | Periodic extension of finite-length $x(n)$, period $N$ |
| $\delta(n)$ | Unit impulse: $1$ at $n=0$, $0$ elsewhere |
| $u(n)$ | Unit step: $1$ for $n\ge 0$, $0$ for $n<0$ |
| $\alpha^n u(n)$ | Causal exponential sequence with scalar base $\alpha$ |
| $x(m,n)$ | 2D image signal (intensity at pixel row $m$, column $n$) |
| $x_k(m,n)$ | $k$-th video frame |
| $w(n)$ | **(a)** Window function in FIR design (§3.3.2); **(b)** Additive time-domain noise in channel model (§1.1.4) — context determines |

### Transforms and Spectral Functions

| Symbol | Definition |
|--------|-----------|
| $X_a(f)$ | Continuous Fourier transform of $x_a(t)$ |
| $X_s(f)$ | Fourier transform of sampled signal $x_s(t)$; periodic copies of $X_a(f)$ at spacing $f_s$ |
| $X(e^{j\omega}) = \sum_n x(n)e^{-j\omega n}$ | DTFT of $x(n)$ |
| $X(z) = \sum_n x(n)z^{-n}$ | z-Transform of $x(n)$, $z\in\mathbb{C}$ |
| $X(k) = \sum_{n=0}^{N-1}x(n)W_N^{nk}$ | $k$-th DFT coefficient (§2.3); also QAM symbol on subcarrier $k$ in OFDM (§1.1.4) |
| $\tilde{X}(k)$ | DFS coefficient of periodic sequence $\tilde{x}(n)$ |
| $H(e^{j\omega})$ | System frequency response (DTFT of $h(n)$) |
| $H(z)$ | System transfer function (z-transform of $h(n)$) |
| $W_N = e^{-j2\pi/N}$ | DFT twiddle factor |
| $\mathbf{W}_N = [W_N^{kn}]$ | $N\times N$ DFT matrix with row/column indices $k,n=0,\ldots,N-1$ |
| $R(e^{j\omega})$ | DTFT of autocorrelation $r_x(n)$ |
| $R(z)$ | z-Transform of autocorrelation $r_x(n)$ |
| $P_x(e^{j\omega}) = \lvert X(e^{j\omega})\rvert^2$ | Power spectral density (PSD) of $x(n)$ |
| $\mathbf{R} = [r_x(i-j)]_{i,j}$ | Autocorrelation matrix (Hermitian Toeplitz) |

### Laplace and z-Domain Variables

| Symbol | Definition |
|--------|-----------|
| $s = \sigma + j\Omega$ | Complex Laplace frequency variable |
| $\sigma = \mathrm{Re}(s)$ | Damping factor (Np/s); $\sigma<0$: decaying, $\sigma>0$: growing |
| $z = re^{j\omega}$ | Complex z-domain variable (polar form) |
| $r = \lvert z\rvert = e^{\sigma T_s}$ | Magnitude of $z$ |
| $z = e^{sT_s}$ | s-plane to z-plane mapping |
| $\mathrm{ROC}$ | Region of Convergence: $\{z\in\mathbb{C}:\sum_n\lvert x(n)\rvert\lvert z\rvert^{-n}<\infty\}$ |
| $\langle n\rangle_N = n\bmod N$ | Modulo-$N$ reduction of integer $n$ |

### LTI System and Filter Parameters

| Symbol | Definition |
|--------|-----------|
| $N$ | **(a)** DFT/FFT length; OFDM subcarrier count (§2.3, §1.1.4); **(b)** IIR filter order (§3.0, §3.2); **(c)** Length of FIR filter (§3.1.3); always stated in context |
| $M$ | **(a)** FIR filter order (taps $= M+1$, §3.1); **(b)** Image height in pixels (§1.1.2); **(c)** Modulation constellation size (§1.1.4) |
| $a_k$ | Feedforward (numerator) coefficients of LCCDE |
| $b_k$ | Feedback (denominator) coefficients of LCCDE ($b_0=1$ by convention) |
| $A(z) = \sum_{k=0}^{M}a_k z^{-k}$ | Numerator polynomial of $H(z)$ |
| $B(z) = \sum_{k=0}^{N}b_k z^{-k}$ | Denominator polynomial of $H(z)$ |
| $v(n)$ | Internal state variable in IIR Direct Form II (§3.2.2); distinct from unit step $u(n)$ |
| $\delta_1,\,\delta_2$ | Passband ripple / stopband attenuation (linear scale) |
| $R_p,\,A_s$ | Passband ripple / stopband attenuation (dB) |
| $\alpha$ | **(a)** Base of exponential sequence $\alpha^n u(n)$ and pole of first-order IIR/allpass, $\lvert\alpha\rvert<1$ (§1.5.3, §3.2, §4.1); **(b)** Constant group delay in samples (§4.3.1 only — local meaning, distinct from (a)) |
| $\beta$ | **(a)** Kaiser window shape parameter (§3.3.2); **(b)** Linear-phase FIR phase offset $\beta\in\{0,\pm\pi/2\}$ (§4.3.1 only) |
| $r$ | **(a)** Magnitude $\lvert z\rvert$ of the z-domain variable; **(b)** Decay parameter in sequences $r^n\cos(\omega_0 n)u(n)$ (§2.2.3) |

### Allpass, Minimum-Phase, and Spectral Factorization

| Symbol | Definition |
|--------|-----------|
| $H_{ap}(z)$ | Allpass filter transfer function: $\lvert H_{ap}(e^{j\omega})\rvert=1\ \forall\omega$ |
| $H_m(z)$ | Minimum-phase filter / minimum-phase factor in decomposition |
| $\alpha_k$ | $k$-th pole of allpass filter, $\lvert\alpha_k\rvert<1$ |
| $\tau(\omega) = -d(\angle H)/d\omega$ | Group delay |
| $\sigma^2$ | Scalar gain in spectral factorization $R(z)=\sigma^2 H_m(z)H_m^*(1/z^*)$ |

### OFDM and Multipath Channel Model (§1.1.4)

| Symbol | Definition |
|--------|-----------|
| $N$ | Number of OFDM subcarriers |
| $k$ | Subcarrier index, $k=0,\ldots,N-1$ |
| $X(k)$ | QAM symbol placed on subcarrier $k$ (also: DFT of $x(n)$ in §2.3) |
| $Y(k)$ | Received signal on subcarrier $k$ after FFT |
| $H(k)$ | Channel frequency response at subcarrier $k$ |
| $W(k)$ | Additive noise on subcarrier $k$ (frequency domain) |
| $\hat{X}(k) = Y(k)/H(k)$ | One-tap equalized symbol estimate |
| $h(l)$ | Discrete channel impulse response tap at delay $l$ |
| $L$ | Number of resolvable channel taps |
| $L_{\mathrm{cp}} = L-1$ | Minimum cyclic prefix length |

---

# Part I: Digital Signals and DSP Overview

> 📖 Textbook §1.1 (Random Signals overview); §2.1 (Discrete-Time Signals)

---

## 1.1 Basic Types and Examples of Digital Signals

A **digital signal** is a discrete-time, discrete-amplitude representation of a physical quantity. It arises from sampling and quantizing a continuous-time analog signal.

### 1.1.1 Speech Signals

A speech signal is a **one-dimensional time-domain discrete sequence**. A continuous pressure wave $x_a(t)$ is sampled at rate $f_s$ (typically 8 kHz for telephony, 44.1 kHz for audio) to produce $x(n) = x_a(nT_s)$ where $T_s = 1/f_s$.

> ![Figure 1.1 speech waveform](<./CourseADSP2026/Fig/fig_1_1.png>)
>
> *Figure 1.1: A speech signal sampled at a fixed rate, shown as a discrete sequence in the time domain. Voiced segments (vowels) exhibit quasi-periodic structure; unvoiced segments (fricatives) appear noise-like.*

### 1.1.2 Image Signals

An image signal is a **two-dimensional discrete sequence** $x(m, n)$, where $(m, n)$ indexes the row and column of a pixel. Grayscale images have scalar values; color images have vector values (e.g., RGB). Standard digital image resolutions range from $320 \times 240$ (VGA) to $3840 \times 2160$ (4K).

$$x(m, n), \quad m = 0, 1, \ldots, M-1;\; n = 0, 1, \ldots, N-1$$

> ![Figure 1.2 image pixel](<./CourseADSP2026/Fig/fig_1_2.jpg>)
>
> *Figure 1.2: An image signal — 2D pixel array. Each element $x(m,n)$ stores intensity (grayscale) or a color vector.*

### 1.1.3 Video Signals

A video signal is a **dynamic image sequence**: a series of image frames $x_k(m,n)$ indexed by frame number $k$, captured at a frame rate (typically 25, 30, or 60 fps). Video introduces temporal correlation in addition to spatial correlation, requiring 3D signal processing.

$$\text{Video:} \quad \lbrace x_k(m,n)\rbrace_{k=0}^{K-1}, \quad k = \text{frame index}$$

The large data volume of uncompressed video (e.g., 1080p at 30 fps $\approx$ 1.5 Gbps) motivates video compression standards such as H.264, H.265/HEVC, and AV1.

### 1.1.4 Communication Signals

**Background: why does the channel "do" convolution at all?** It is tempting to picture the channel as a simple wire that just hands the receiver an attenuated, delayed copy of whatever was transmitted — i.e. $y(n) = a\,x(n-d) + w(n)$. That picture is not wrong, but it is the special case of a channel with a *single* propagation path. In practice, the transmitted signal almost always reaches the receiver via *several* distinct physical paths — a direct line-of-sight path plus a handful of reflections (off the ground, walls, the sea surface/seabed for an underwater acoustic link, etc.). Each path $l$ has its own geometric length, and therefore its own propagation delay, and its own attenuation/phase shift from reflection and absorption loss; call that path's net (complex) gain $h(l)$. The receiver does not see any one path in isolation — it sees the *sum* of all of them, each one a delayed, scaled copy of the same transmitted signal:

$$y(n) = \underbrace{h(0)x(n)}_{\text{path }0} + \underbrace{h(1)x(n-1)}_{\text{path }1} + \underbrace{h(2)x(n-2)}_{\text{path }2} + \cdots + w(n) = \sum_l h(l)\,x(n-l) + w(n)$$

This "delay, weight, and add" operation is, by definition, convolution. (A useful mental picture: speaking in a room with an echo — the microphone captures not just your direct voice but several delayed, attenuated copies of it bouncing off the walls and ceiling, all summed together.)

More fundamentally, this is not something the channel "chooses" to do — it is a consequence of two physically reasonable assumptions. Any discrete signal can be decomposed into a weighted sum of shifted unit impulses, $x(n) = \sum_k x(k)\,\delta(n-k)$. If the channel is **linear** (multiple superimposed waves add without interacting — true of EM (Electromagnetic) and acoustic propagation at ordinary power levels) and **time-invariant** (the multipath geometry — delays and gains — does not change appreciably over the duration of one symbol, i.e. a quasi-static channel), then its response to a shifted impulse $\delta(n-k)$ is just a shifted copy of its impulse response, $h(n-k)$, and by linearity its response to the full signal is the sum of these shifted, scaled responses:

$$y(n) = \sum_k x(k)\,h(n-k) = x(n) \ast h(n)$$

In other words, *any* LTI channel — multipath or not — necessarily produces an output that is the convolution of the input with its impulse response $h(n)$; $h(n)$ is simply the packaged record of every propagation path's delay and gain. The OFDM channel model below inherits exactly this structure, with $L$ denoting the number of resolvable channel taps (the discrete-time length of the multipath spread).

*Note: Before the equations below, it helps to have the intuitive picture in mind. An OFDM system does not send one fast stream of bits over a single wide channel — it splits the data into $N$ slower streams and sends each one over its own narrow frequency lane (a "subcarrier"). To put bits onto a single subcarrier, a small group of bits is first mapped to one complex number called a symbol — this mapping is what "modulation" (e.g., QAM) means. A symbol can be pictured as a point on a 2D plane (in-phase/quadrature, or I/Q); the full set of possible points is called a constellation (QPSK has 4 points, 16-QAM has 16, and so on — more points per constellation packs in more bits per symbol, but pushes the points closer together and makes the system more sensitive to noise). Placing one such symbol on each of the $N$ subcarriers and taking an IFFT produces the time-domain sequence $x(n)$ below — it is literally many narrowband QAM signals added together. Finally, the cyclic prefix (CP) is a short copy of the tail of $x(n)$, glued onto its front before transmission. Its job is to absorb the smearing caused by multipath echoes, so that after the receiver discards it, the messy convolution in time collapses into a simple per-subcarrier multiplication in frequency — exactly the clean relationship $Y(k) = H(k)X(k) + W(k)$ shown next. Here $k = 0, 1, \ldots, N-1$ indexes the subcarrier; $X(k)$ is the transmitted symbol placed on subcarrier $k$ before the IFFT (the QAM constellation point carrying that subcarrier's bits); $H(k)$ is the channel's frequency response at subcarrier $k$ — a single complex number (gain and phase) describing how that narrow frequency lane is attenuated and rotated by the physical channel; $W(k)$ is the additive noise on subcarrier $k$ after the FFT (typically modeled as complex Gaussian, inherited from thermal noise in the time domain); and $Y(k)$ is what the receiver actually observes — the noisy, channel-distorted version of $X(k)$. Because the relationship is a simple multiplication rather than a convolution, the receiver can recover an estimate of $X(k)$ by one-tap equalization, $\hat{X}(k) = Y(k)/H(k)$, instead of solving a much harder time-domain deconvolution problem.*

Modern digital communication uses discrete baseband sequences. Key examples:

**OFDM baseband sequence**: An Orthogonal Frequency Division Multiplexing system maps bits onto $N$ subcarriers via modulation (e.g., QAM), applies an $N$-point IFFT to produce the time-domain discrete sequence $x(n)$, and appends a cyclic prefix (CP) to combat multipath.

Explicitly, with $X(k)$ the QAM symbol carried on subcarrier $k$, the transmitted baseband sample sequence is the $N$-point IDFT of the symbol block:

$$\boxed{x(n) = \frac{1}{N}\sum_{k=0}^{N-1} X(k)\, e^{j2\pi kn/N}}, \quad n = 0, 1, \ldots, N-1$$

(the IFFT is simply the fast algorithm that computes this sum). The receiver discards the cyclic prefix, takes an $N$-point FFT of the remaining samples to recover $X(k)$, and the received signal after FFT demodulation is:

$$Y(k) = H(k) X(k) + W(k), \quad k = 0, 1, \ldots, N-1$$

**Why the cyclic prefix works: from linear to circular convolution.** The physical channel can only perform *linear* convolution, $y(n) = \sum_l h(l)\,x(n-l) + w(n)$, whose output is $N+L-1$ samples long and pulls in values from *before* the block — i.e., from the tail of the previous OFDM symbol.

> *Why is the channel restricted to linear convolution in the first place?* The channel is a real, continuously operating physical system — multipath propagation: a direct path plus several delayed, attenuated, phase-shifted echoes. It filters whatever signal is actually flowing through it in continuous time, and it has no concept of an "OFDM symbol block." Chopping the bitstream into length-$N$ blocks is purely a digital bookkeeping convention on the transmitter/receiver side (needed so the IFFT/FFT can be applied), not a physical property of the propagation medium. So when evaluating $y(n)$ for $n$ near the start of the current block, the convolution sum $\sum_l h(l)\,x(n-l)$ genuinely needs samples $x(-1), x(-2), \ldots$ from *before* index $0$ — and the channel, having no notion of periodicity, supplies whatever was *actually transmitted* just before: the tail of the previous OFDM symbol (or, once a CP is inserted, the CP itself). This is exactly linear convolution: it "honestly" pulls in real transmission history. Circular convolution, by contrast, is a purely mathematical construct that *assumes* $x(n)$ repeats with period $N$ — i.e., that the sample "before $x(0)$" is the *current* block's own $x(N-1)$, rather than whatever was genuinely sent earlier. Nothing about real wave propagation enforces that assumption; it becomes true of the channel's output only once we deliberately engineer it — which is precisely the job of the CP, as the worked example below shows.

The DFT, however, only diagonalizes *circular* convolution, $y(n) = \sum_l h(l)\,x(\langle n-l\rangle_N)$, where $x(n)$ is treated as periodic with period $N$ (this property is formalized later via the DFS/circular-convolution theorem in Part II). These are not the same operation. The entire job of the CP is to make the channel's linear convolution behave, once the CP is discarded, *exactly* like a circular convolution of $x(n)$ with $h(n)$.

*Worked example ($N=4$, two-tap channel).* Let the channel have taps $h(0), h(1)$ (length $L=2$), so a CP of length $L_{cp}=L-1=1$ suffices. The transmitted sequence is the CP sample $s(-1)=x(3)$ followed by the block $s(0..3)=x(0..3)$. The channel produces $y(n)=h(0)s(n)+h(1)s(n-1)$; after the receiver discards $s(-1)$:

$$y(0) = h(0)x(0) + h(1)x(3), \qquad y(1) = h(0)x(1) + h(1)x(0),$$
$$y(2) = h(0)x(2) + h(1)x(1), \qquad y(3) = h(0)x(3) + h(1)x(2).$$

Compare with the circular convolution $y_{\text{circ}}(n) = \sum_l h(l)\,x(\langle n-l\rangle_4)$: e.g. $y_{\text{circ}}(0) = h(0)x(0) + h(1)x(\langle -1\rangle_4) = h(0)x(0)+h(1)x(3)$, identical to $y(0)$ above, and likewise for $n=1,2,3$. So the $N$ retained samples are, term for term, the circular convolution of $x(n)$ and $h(n)$ — and the DFT convolution theorem then gives $Y(k)=H(k)X(k)$ directly.

This match depends entirely on the guard samples being an **exact copy** of $x(N-1), x(N-2), \ldots$ — not zero-padding. If $s(-1)$ were $0$ instead of $x(3)$, $y(0)$ would become $h(0)x(0)+h(1)\cdot 0$, which does **not** equal $y_{\text{circ}}(0)=h(0)x(0)+h(1)x(3)$. Only the literal copy of the tail reproduces the periodic wrap-around that circular convolution assumes; this is why it is a *cyclic prefix* and not merely a blank guard interval.

**Two distinct impairments from multipath, fixed by two different mechanisms.** Delay spread in the channel — multiple echoes arriving with different delays — causes two separable problems:

| Impairment | Cause | How the CP fixes it |
|---|---|---|
| **Inter-symbol interference (ISI)** | Echoes carrying the *previous* symbol's tail bleed into the current symbol's receive window | *Guard time*: if $L_{cp} \geq$ the channel's maximum delay spread, the stale energy decays entirely within the discarded CP interval and never reaches the $N$ retained samples. Any guard interval — even zeros — would achieve this. |
| **Inter-carrier interference (ICI) / loss of subcarrier orthogonality** | Within a single symbol, *linear* convolution lacks the periodic boundary condition that the complex exponentials $e^{j2\pi kn/N}$ need to remain eigenfunctions of the channel operator, so energy leaks between subcarriers | *Exact tail copy*: converts the in-block operation into a true circular convolution, under which each $e^{j2\pi kn/N}$ passes through unchanged in shape (only scaled by $H(k)$), so no energy crosses subcarriers. Zero-padding alone would not prevent this. |

The two fixes are independent: sufficient *length* protects against ISI, while the exact *duplication* of the tail protects against ICI. Both are necessary — a long-enough zero guard band would still leave a linear-convolution boundary artifact inside the block, smearing the $N$ orthogonal subcarriers into each other even with no inter-symbol contamination at all. It is the combination of the two that yields the clean, crosstalk-free model $Y(k)=H(k)X(k)+W(k)$ used throughout this section, and with it the simple one-tap equalizer $\hat{X}(k) = Y(k)/H(k)$ in place of a much harder time-domain deconvolution.

**Constellation diagrams**: The modulation alphabet is visualized as a constellation. Common examples:
- **QPSK** ($M=4$): 4 points on a circle, 2 bits/symbol
- **8-PSK** ($M=8$): 8 points on a circle, 3 bits/symbol
- **16-QAM** ($M=16$): 16 points on a square grid, 4 bits/symbol
- **64-QAM** ($M=64$): 64 points, 6 bits/symbol

> ![Figure 1.3](<./CourseADSP2026/Fig/fig_1_3.png>)
>
> *Figure 1.3: Baseband communication signals — constellation maps. Each point represents one symbol carrying $\log_2 M$ bits. Higher-order constellations achieve greater spectral efficiency but require higher SNR.*

### 1.1.5 Radar Signals

Radar signals are specially designed waveforms transmitted and received after reflection from a target. Two common types:

**Linear frequency modulation (LFM / chirp)**: The instantaneous frequency increases linearly with time:

$$x(t) = \text{rect}\!\left(\frac{t}{T}\right) e^{j\pi \mu t^2}, \quad \mu = B/T \text{ (chirp rate)}$$

> **Notation — the rect function**: $\text{rect}(\cdot)$ is the *rectangular (boxcar) window function*,
> $$\text{rect}(u) = \begin{cases} 1, & \lvert u\rvert \le \tfrac{1}{2} \\ 0, & \lvert u\rvert \gt \tfrac{1}{2} \end{cases}$$
> i.e. a "brick-wall" gate that is 1 inside $[-\tfrac12, \tfrac12]$ and 0 everywhere else (the value exactly at the edges, $u=\pm\tfrac12$, is a convention and is often taken as $1$, $0$, or $\tfrac12$ depending on the textbook — it doesn't matter physically since it's a single point). Substituting $u = t/T$ rescales the gate to width $T$: $\text{rect}(t/T) = 1$ for $-T/2 \le t \le T/2$ and $0$ outside it. Its role here is purely to *window* the otherwise infinite-duration chirp $e^{j\pi\mu t^2}$ down to one finite pulse of duration $T$ — i.e., it is the mathematical way of saying "the radar transmits this chirp for $T$ seconds, then stops." Without the rect factor, the formula would describe a chirp that exists for all time; with it, $x(t)$ is a single, time-limited radar pulse, which is what is actually transmitted.

After sampling: $x(n) = e^{j\pi\mu(nT_s)^2}$. LFM enables pulse compression: a long pulse (high energy) is compressed to a short pulse (high resolution) via matched filtering.

**Frequency-diversity radar**: Successive pulses use different carrier frequencies to improve target discrimination and reduce scintillation. The transmitted waveform exhibits a stepped-frequency pattern.

> ![Figure 1.4](<./CourseADSP2026/Fig/fig_1_4.png>)
>
> *Figure 1.4: Radar signals — (left) LFM chirp with linearly increasing instantaneous frequency; (right) frequency-diverse radar with stepped carrier frequencies across pulses.*

---

## 1.2 Applications of Digital Signal Processing

DSP methods underpin virtually every modern technology domain:

| Domain | Key DSP Tasks |
|--------|--------------|
| **Multimedia processing** | Filtering, enhancement, compression (MP3, JPEG, H.265), speech/face recognition |
| **Wireless communications** | Modulation/demodulation, channel estimation, equalization, source/channel coding (OFDM, 5G NR) |
| **Radar and sonar** | Waveform design, pulse compression, detection, localization, target recognition, DOA estimation |
| **Biomedical engineering** | ECG/EEG monitoring, ultrasound imaging, diagnostic signal processing |
| **Geophysics and meteorology** | Seismic data processing, weather signal analysis, oil exploration |
| **Automatic control** | Digital PID controllers, state estimation (Kalman filter), adaptive systems |

**OFDM transceiver chain.** The transmitter and receiver form a symmetric pipeline, with the cyclic prefix marking the boundary between digital baseband processing and the analog/RF front end:

- **Transmitter (digital baseband → analog/RF):** Data Bits → Channel Coding → Interleaving → QAM Modulation → S/P (Serial-to-Parallel) → IFFT (Inverse Fast Fourier Transform) → P/S (Parallel-to-Serial) → CP Insertion (Cyclic Prefix) → DAC (Digital-to-Analog Converter) → RF TX (Radio Frequency Transmitter)
- **Receiver (analog/RF → digital baseband), the symmetric reverse chain:** RF RX (Radio Frequency Receiver) → ADC (Analog-to-Digital Converter) → CP Removal → S/P → FFT (Fast Fourier Transform) → P/S → QAM Demodulation → De-Interleaving → Channel Decoding → Data Bits

---

## 1.3 General Digital Signal Processing System

### 1.3.1 System Pipeline

A continuous-time bandlimited signal $x_a(t)$ (bandwidth $B$ Hz) is processed as follows:

$$x_a(t) \xrightarrow{\text{AAF}} \xrightarrow{\text{A/D},\; f_s} x(n) \xrightarrow{h(n)} y(n) \xrightarrow{\text{D/A}} y_a(t)$$

1. **Anti-aliasing filter (AAF)**: Lowpass filter with cutoff $f_s/2$ — removes frequency components above $f_s/2$ to prevent aliasing.

   > **Where do the periodic spectral copies come from?** Ideal sampling can be modeled as multiplying the continuous signal by a periodic train of unit impulses spaced $T_s$ apart (a "Dirac comb"), $p(t) = \sum_{n} \delta(t-nT_s)$, so that $x_s(t) = x_a(t)\,p(t) = \sum_n x_a(nT_s)\,\delta(t-nT_s)$ — this is exact, since "keep the value at $nT_s$, discard everything in between" is precisely what an impulse at each sampling instant encodes. Multiplication in time becomes *convolution* in frequency, so the spectrum of $x_s(t)$ is $X_a(f)$ convolved with $P(f)$, the Fourier transform of the impulse train. Impulse trains have a self-dual property: the Fourier transform of a period-$T_s$ impulse train is *itself* an impulse train, but in frequency, with spacing $f_s = 1/T_s$: $P(f) = f_s\sum_k \delta(f-kf_s)$ (teeth packed close together in time spread far apart in frequency, and vice versa — the same time/frequency reciprocity that makes short pulses wideband). Convolving $X_a(f)$ with a single impulse $\delta(f-kf_s)$ just slides a copy of $X_a(f)$ to be centered at $kf_s$ (by the sifting property of convolution with an impulse); convolving with the *entire* impulse train therefore stamps down a whole sequence of shifted copies, one at every integer multiple of $f_s$: $X_s(f) = f_s\sum_k X_a(f-kf_s)$ — exactly the periodic replication used below.
   >
   > *Equivalent intuition, without impulse trains:* a sequence of samples literally cannot tell a sinusoid at frequency $f$ apart from one at $f + kf_s$. At the sampling instants $t = nT_s$, $e^{j2\pi(f+kf_s)nT_s} = e^{j2\pi f nT_s}\cdot e^{j2\pi k(f_sT_s)n} = e^{j2\pi f nT_s}\cdot e^{j2\pi kn} = e^{j2\pi f nT_s}$, because $f_sT_s = 1$ and $kn$ is an integer (so $e^{j2\pi kn}=1$). Every analog frequency $f$ is therefore indistinguishable, once sampled, from "phantom" frequencies $f \pm f_s,\, f\pm 2f_s,\dots$ — they all produce identical sample values. The sampled signal's spectrum *must* be periodic with period $f_s$, simply because sampling itself cannot resolve frequencies that differ by a multiple of $f_s$.
   >
   > **Why cutoff exactly at $f_s/2$?** Given that sampling at rate $f_s$ periodically *repeats* the analog spectrum, the discrete-time spectrum consists of shifted copies of the analog spectrum, $X_a(f - kf_s)$ for every integer $k$, spaced $f_s$ apart along the frequency axis. The baseband copy ($k=0$) occupies $[-f_s/2,\, f_s/2]$, and its nearest neighbor ($k=1$) occupies $[f_s/2,\, 3f_s/2]$ — so two adjacent replicas meet exactly at the midpoint between them, $f_s/2$. If the analog input still carries energy above $f_s/2$ at the moment it is sampled, that energy lands inside the span of the neighboring replica, and the two overlap. Once overlapped, the two contributions cannot be told apart by any downstream filter — this is aliasing, and it is irreversible by construction, not just inconvenient. So $f_s/2$ is not an arbitrary design choice; it is the largest frequency that can be let through *without guaranteeing overlap* with the next replica, i.e. it is the **Nyquist frequency** (also called the folding frequency, since this is the point about which spectral content "folds back" on top of itself). The AAF's role is to physically enforce — before sampling ever happens — the bandlimiting condition $B \le f_s/2$ that the Nyquist–Shannon theorem (§1.3.2 below) requires for perfect reconstruction, regardless of whether the original analog signal already happened to satisfy it.

2. **A/D conversion**: Samples $x_a(t)$ at rate $f_s \geq 2B$ and quantizes to $b$ bits.

   > **Why $f_s \geq 2B$, and not $f_s \geq B$?** A common error is to assume that since the highest frequency component is $B$ Hz, sampling once per "half-cycle" — i.e. at rate $f_s = B$ — should suffice. This overlooks that any *real-valued* signal has a two-sided spectrum: writing $\cos(2\pi ft) = \tfrac12 e^{j2\pi ft} + \tfrac12 e^{-j2\pi ft}$ shows that a real sinusoid is built from a conjugate *pair* of complex exponentials at $+f$ and $-f$. Consequently $X_a(f) = 0$ for $\lvert f\rvert \gt B$ means the spectrum truly occupies $[-B,B]$ — a total width of $2B$, not $B$. The replication argument above already used this: the baseband copy spans $[-f_s/2, f_s/2]$, so avoiding overlap with the neighboring replica requires the *full* $2B$-wide occupied band to fit inside one period, i.e. $f_s \geq 2B$.
   >
   > A concrete failure mode at $f_s = B$: consider $x_a(t) = \cos(2\pi Bt)$, a signal sitting right at the band edge. Sampling at $t = n/B$ gives $x(n) = \cos(2\pi B \cdot n/B) = \cos(2\pi n) = 1$ for every $n$ — a constant sequence, regardless of the sinusoid's true amplitude or phase. One sample per cycle can never distinguish "the wave went up and came back down" from "the wave never moved"; at least two samples per cycle (two different phases within one period) are needed to pin down a sinusoid, which is exactly the origin of the factor of 2.
   >
   > **Why "$b$" bits — a free parameter, not a derived constant.** Unlike $f_s \geq 2B$, which is a hard mathematical necessity (violating it causes irreversible aliasing), the number of quantization bits $b$ is an engineering *choice*: it sets the number of amplitude levels to $2^b$, trading quantization noise against bit rate/storage. The one quantitative rule of thumb is that each additional bit improves the signal-to-quantization-noise ratio by about 6 dB:
   > $$\text{SQNR} \approx 6.02\,b + 1.76\ \text{(dB)}$$
   > Typical choices reflect this trade-off rather than a derivation: $b=8$ for telephony (G.711), $b=16$ for CD audio, $b=24$ for studio recording.

3. **Discrete-time processing**: Convolution with $h(n)$, or more generally any digital signal processing algorithm.
4. **D/A conversion and reconstruction filter**: Converts the discrete output back to a continuous signal.

> ![Figure 1.6](<./CourseADSP2026/Fig/fig_1_6.png>)
>
> *Figure 1.6: General DSP system — pipeline from continuous input to continuous output, with frequency-domain illustrations at each stage.*

### 1.3.2 Sampling Theorem and Bandlimited Condition

**Nyquist–Shannon Sampling Theorem**: A continuous-time signal $x_a(t)$ with bandwidth $B$ Hz (i.e., $X_a(f) = 0$ for $\lvert f\rvert \gt B$) can be perfectly reconstructed from its samples $x(n) = x_a(nT_s)$ if and only if:

$$f_s = \frac{1}{T_s} \geq 2B$$

The minimum sampling rate $f_s = 2B$ is the **Nyquist rate**.

> **Why can frequency be negative?** Negative frequency is a by-product of representing real signals on a complex-exponential basis, not an independent physical quantity. By Euler's formula,
>
> $$\cos(2\pi ft) = \tfrac12 e^{j2\pi ft} + \tfrac12 e^{-j2\pi ft}$$
>
> any real sinusoid is necessarily built from a *conjugate pair* of complex exponentials at $+f$ and $-f$, whose imaginary parts cancel. For a real signal $x_a(t)$, this forces Hermitian (conjugate) symmetry on its spectrum,
>
> $$X_a(-f) = X_a^*(f)$$
>
> so the negative-frequency content has the same magnitude as its positive-frequency mirror and carries no independent information, but it is not optional: it must be present for the sum to come out real, and it occupies its own share of the frequency axis. This is precisely why "bandwidth $B$" in the theorem above means
>
> $$X_a(f) = 0 \quad \text{for } \lvert f\rvert \gt B$$
>
> the occupied spectrum is the *two-sided* interval $[-B,B]$, total width $2B$, not just $[0,B]$.
>
> Geometrically, $e^{j2\pi ft}$ is a unit phasor in the complex (I/Q) plane rotating at angular rate $2\pi f$: positive $f$ means counter-clockwise rotation, negative $f$ means clockwise. A real signal is the projection of this rotating phasor onto the real axis, and that projection alone cannot distinguish a counter-clockwise rotation from a clockwise one — both project identically onto the real axis. Hence a real signal must carry both rotation directions ($\pm f$) to be represented faithfully. For a genuinely *complex*-valued signal (e.g. the I/Q baseband signal in a digital receiver, or an analytic signal), $+f$ and $-f$ are no longer forced to be conjugate partners — they become independent, physically meaningful quantities (e.g. the sign of a Doppler shift indicates approach vs. recession).

The relationship between analog frequency $f$ and digital frequency $\omega$ is:

$$\omega = 2\pi f T_s = \frac{2\pi f}{f_s}$$

Digital frequency $\omega \in [-\pi, \pi]$ corresponds to analog frequency $f \in [-f_s/2,\; f_s/2]$.

**If $f_s \lt 2B$ (undersampling)**: Spectral replicas overlap → **aliasing** — high-frequency components masquerade as low-frequency ones, irreversibly corrupting the signal.

> ![Figure 1.7](<./CourseADSP2026/Fig/fig_1_7.png>)
>
> *Figure 1.7: Sampling operation — (a) continuous-time spectrum $X_a(f)$ bandlimited to $B$; (b) discrete-time spectrum when $f_s \gt 2B$ (no aliasing); (c) discrete-time spectrum when $f_s \lt 2B$ (aliasing).*

### 1.3.3 Bandpass Signals and the Bandpass Sampling Theorem

§1.3.2 implicitly assumed a **baseband (lowpass) signal**: $X_a(f) = 0$ for $\lvert f\rvert \gt B$, spectrum centered at $f = 0$. Many real signals instead are **bandpass**: their energy sits in a band $[f_1, f_2]$ away from $f = 0$ (plus its conjugate mirror $[-f_2, -f_1]$) — narrowband sonar/radar returns, AM/FM radio, and IF-stage receiver signals are typical examples. The conventional bandwidth is the width of the *positive-frequency* interval, $B = f_2 - f_1$, *not* $f_2$ itself.

**Naively applying §1.3.2** would suggest $f_s \geq 2f_2$ — sampling at twice the highest frequency present — which is wasteful when $f_2 \gg B$. The **bandpass sampling theorem** (also called IF sampling, or deliberate undersampling) shows that the true minimum sampling rate depends only on the bandwidth $B$, not on how far the band sits from $f = 0$: there exists an integer $q$, $1 \leq q \leq \lfloor f_2/B \rfloor$, such that

$$\frac{2f_2}{q} \;\leq\; f_s \;\leq\; \frac{2f_1}{q-1}$$

(for $q=1$ this reduces to the baseband condition $f_s \geq 2f_2$). When $f_2$ is an integer multiple of $B$ (the band is "harmonically aligned"), the theoretical minimum $f_s = 2B$ is achievable exactly — the *same* floor as for a baseband signal of bandwidth $B$.

> **Why is the floor still $2B$, not $B$?** It is tempting to think that replicating a single band of width $B$ at spacing $f_s = B$ tiles it edge-to-edge with no gap, so $f_s = B$ should suffice. This overlooks the conjugate mirror band: a real bandpass signal occupies $[f_1, f_2]$ **and** $[-f_2, -f_1]$ — two separate intervals on the frequency axis, each of width $B$, total occupied measure $2B$ (exactly the two-sided argument from §1.3.2, now applied to a shifted band). Sampling at rate $f_s$ is equivalent to wrapping the frequency axis onto a circle of circumference $f_s$ (taking frequency modulo $f_s$); for alias-free recovery this wrap must be injective on the occupied support. If $f_s \lt 2B$, the occupied measure ($2B$) exceeds the circle's circumference ($f_s$), so by a simple pigeonhole argument the wrap **cannot** be injective — overlap is unavoidable no matter where the band sits.
>
> Concretely, at $f_s = B$: the positive band's own replicas $[f_1+kB,\, f_2+kB]$ already tile the *entire* frequency axis with zero gaps (each replica has width exactly $B$, spaced exactly $B$ apart). The mirror band's replicas, sitting on that same comb, then have nowhere to land except squarely on top of the positive band's replicas — total, unavoidable overlap. At $f_s = 2B$ (achievable when $f_2 = nB$), the positive-band and mirror-band replicas instead tile the axis in a perfect alternating pattern — one period of width $2B$ holding exactly one positive-band replica plus one mirror-band replica, edge-to-edge, zero overlap. This is the tightest possible packing.

The general lesson: the Nyquist floor is set by the **total occupied measure of the signal's full (two-sided) spectral support**, $f_s \geq (\text{total occupied measure})$, regardless of where on the frequency axis that support sits. For baseband signals this measure is $2B$ (§1.3.2); for bandpass signals it is likewise $2B$, where $B = f_2-f_1$ is the one-sided occupied width. Equivalently: a real signal carrying $2B$ Hz of total spectral occupancy needs $2B$ real samples/second no matter how the carrier is chosen — or, if down-converted to a complex (I/Q) baseband representation first, $B$ complex samples/second, which is the same $2B$ real numbers/second once I and Q are counted separately.

### 1.3.4 Spectral Translation: Up-Conversion and the Apparent Doubling of Bandwidth

A natural follow-up question: if a baseband signal $x(t)$ of bandwidth $B$ is **up-converted** to a carrier frequency $f_c$ — e.g. by mixing with a real carrier, $s(t) = x(t)\cos(2\pi f_ct)$ — does the resulting bandpass signal's bandwidth grow?

By the modulation (frequency-shift) theorem,

$$S(f) = \frac{1}{2}\big[X(f-f_c) + X(f+f_c)\big]$$

Since $\cos(2\pi f_ct)$ itself has spectrum supported at $\pm f_c$ (Euler's formula again), multiplication produces **two** shifted copies of the *entire* original two-sided spectrum $X(f)$ — not half of it. $X(f)$ already spans $[-B,B]$ (total width $2B$, §1.3.2), so each shifted copy retains that full $2B$ width:

- Positive-frequency lobe: $[f_c - B,\, f_c + B]$, width $2B$
- Mirror lobe at $-f_c$: $[-f_c - B,\, -f_c + B]$, width $2B$

In other words, each lobe is exactly as wide as the *entire* original baseband spectrum was — nothing has grown; the same $2B$-wide shape has simply been picked up and moved to sit at $\pm f_c$ instead of straddling $f=0$.

Using the bandpass-signal convention $B' = f_2-f_1$ from §1.3.3, the new one-sided bandwidth is $B' = 2B$ — numerically double the original baseband $B$. But note what changed and what did not: the physical spectral content is still exactly $2B$ Hz wide (it is literally the same shape, just translated); what changed is the *labeling convention*. For a baseband (lowpass) signal, "bandwidth $B$" by convention reports only the *radius* from the natural reference point $f=0$ — the highest frequency present — leaving the other half of the occupied span ($[-B,0]$) implicit. For a bandpass signal centered away from $0$, there is no natural "halving point" left to exploit, so the convention instead reports the *full* one-sided occupied width $f_2-f_1$. The same $2B$ Hz of physical spectrum is therefore reported as "$B$" under one convention and "$2B$" under the other — the doubling is a bookkeeping artifact of where the signal sits relative to $f=0$, not new information appearing out of nowhere.

This is the textbook fact that **double-sideband (DSB) modulation occupies transmission bandwidth $2B$** for a baseband message of bandwidth $B$, because both the upper and lower sidebands are transmitted, and they are conjugate mirror images of each other — carrying no independent information, exactly as in §1.3.2's negative-frequency discussion, only now the redundant pair sits at $f_c$ instead of at $0$. **Single-sideband (SSB)** modulation removes one sideband (by filtering, or by mixing with a complex carrier $e^{j2\pi f_ct}$ instead of a real $\cos$) and recovers the original $B$-Hz efficiency, at the cost of more complex transmit/receive hardware.

The practical consequence for sampling: if this DSB signal is bandpass-sampled directly (§1.3.3) without first down-converting, the relevant bandwidth parameter is the doubled $2B$, so the minimum rate becomes $f_s \geq 2(2B) = 4B$ — twice what would be needed for the baseband signal or for an SSB-modulated version of it.

### 1.3.5 The Two Fundamental Problems of DSP

The DSP pipeline gives rise to two core problems:

**① Signal Description and Analysis — Transforms**

How do we efficiently describe and analyze discrete-time signals in alternative domains?
- **z-transform**: General framework for discrete-time system analysis
- **DTFT**: Frequency-domain representation on the unit circle
- **DFT/FFT**: Practical numerical computation tool

**② System Description and Design — Filters**

How do we characterize discrete-time systems and design them to perform desired signal transformations?
- Difference equations, impulse responses, frequency responses
- FIR and IIR filter structures and design methods

---

## 1.4 Review of Undergraduate DSP Course Structure

Undergraduate DSP courses are organized around two pillars:

```
Undergraduate DSP
├── Transforms
│   ├── DTFT — spectrum of infinite-length sequences
│   ├── z-Transform — generalization to the complex plane
│   ├── DFT — finite-length, numerical computation
│   └── FFT — fast algorithm for the DFT
│
└── Filters
    ├── FIR filter structures and design
    ├── IIR filter structures and design
    └── Special filter classes (allpass, minimum-phase, linear-phase)
```

This chapter reviews these foundations as preparation for the **stochastic signal processing** topics in subsequent chapters. The emphasis shifts from deterministic signals to **random signals** — signals characterized by statistical properties (mean, autocorrelation, power spectrum) rather than exact waveforms.

---

## 1.5 Elementary Sequences (Fundamental Building Blocks)

Three sequences serve as the universal building blocks of discrete-time signal processing. Every other sequence and system property is ultimately expressed in terms of these.

### 1.5.1 Unit Impulse $\delta(n)$

$$\delta(n) = \begin{cases} 1, & n = 0 \\ 0, & n \neq 0 \end{cases}$$

**Sifting property** — the reason $\delta(n)$ is fundamental:

$$\sum_{n=-\infty}^{\infty} x(n)\,\delta(n-k) = x(k)$$

A shifted impulse $\delta(n-k)$ "picks out" the single value $x(k)$. Equivalently, *any* sequence decomposes into a superposition of weighted, shifted impulses:

$$\boxed{x(n) = \sum_{k=-\infty}^{\infty} x(k)\,\delta(n-k)}$$

This decomposition is the starting point for deriving the convolution sum: if a system is LTI, its response to $x(n)$ is fully determined by its response to a single $\delta(n)$ (the impulse response $h(n)$).

### 1.5.2 Unit Step $u(n)$

$$\boxed{u(n) = \begin{cases} 1, & n \geq 0 \\ 0, & n < 0 \end{cases}}$$

> **Convention**: $u(0) = 1$ (the step "turns on" at $n = 0$). This differs from the continuous-time Heaviside function, where $u(0)$ is sometimes left ambiguous or set to $\tfrac{1}{2}$; in discrete time there is no such ambiguity.

**Relationship between $u(n)$ and $\delta(n)$**:

$$u(n) = \sum_{k=0}^{\infty} \delta(n-k), \qquad \delta(n) = u(n) - u(n-1)$$

The step is the running sum of the impulse; the impulse is the first difference of the step — a discrete-time analogy of integration and differentiation.

> **Why $u(n)$ has no DTFT in the ordinary sense**: The absolute sum $\sum_{n=0}^{\infty}\lvert u(n)\rvert = \infty$ diverges, so the DTFT does not exist as a conventional integral. $u(n)$ is instead handled via the z-transform (ROC: $\lvert z\rvert>1$, §2.2.3) or via distribution theory (adding a Dirac delta at $\omega=0$ to the formal DTFT).

### 1.5.3 Real Exponential Sequence $\alpha^n u(n)$

$$\alpha^n u(n) = \begin{cases} \alpha^n, & n \geq 0 \\ 0, & n < 0 \end{cases}$$

| $\lvert\alpha\rvert$ | Behavior |
|------------|----------|
| $\lvert\alpha\rvert < 1$ | Decays to zero — stable causal system |
| $\lvert\alpha\rvert = 1$ | Sustained oscillation (or DC for $\alpha=1$) |
| $\lvert\alpha\rvert > 1$ | Grows without bound — unstable |

This is the prototype impulse response of a first-order causal IIR filter. Its z-transform $\dfrac{1}{1-\alpha z^{-1}}$ (ROC: $\lvert z\rvert>\lvert\alpha\rvert$) is the single most-used entry in the z-transform table (§2.2.3).

### 1.5.4 Complex Exponential Sequence $e^{j\omega_0 n}$

$$x(n) = e^{j\omega_0 n} = \cos(\omega_0 n) + j\sin(\omega_0 n)$$

where $\omega_0 \in [-\pi, \pi]$ is the digital angular frequency (rad/sample). The real and imaginary parts are the discrete cosine and sine sequences respectively.

**Correspondence to analog frequency:**

Sampling the continuous-time complex exponential $e^{j\Omega_0 t}$ at rate $f_s = 1/T_s$ gives:

$$e^{j\Omega_0 t}\big|_{t=nT_s} = e^{j\Omega_0 T_s n} = e^{j\omega_0 n}, \qquad \omega_0 = \Omega_0 T_s = \frac{2\pi f_0}{f_s}$$

Higher analog frequencies map to larger $\omega_0$; lower analog frequencies map to smaller $\omega_0$. The digital frequency range $\omega_0 \in [-\pi, \pi]$ corresponds exactly to the analog Nyquist interval $f_0 \in [-f_s/2,\, f_s/2]$.

| $\omega_0$ | Analog frequency | Sequence behavior |
|---|---|---|
| $0$ | $0$ Hz (DC) | $x(n) = 1$ (constant) |
| $\pi/2$ | $f_s/4$ | Period of 4 samples |
| $\pi$ | $f_s/2$ (Nyquist) | $x(n) = (-1)^n$ (fastest oscillation) |

**Role as basis functions:**

The DTFT inverse formula decomposes any sequence into a continuum of complex exponentials:

$$x(n) = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega})\, e^{j\omega n}\, d\omega$$

$X(e^{j\omega})$ is the spectral weight — the amplitude and phase of each frequency component. The role of $e^{j\omega_0 n}$ in discrete-time exactly parallels that of $e^{j\Omega t}$ in continuous-time Fourier analysis:

| Analog domain | Digital domain |
|---|---|
| Basis function $e^{j\Omega t}$, $\Omega \in (-\infty, +\infty)$ | Basis function $e^{j\omega n}$, $\omega \in [-\pi, \pi]$ |
| Spectral weight $X_a(\Omega)$ | Spectral weight $X(e^{j\omega})$ |
| Frequency axis unbounded | Frequency axis folded into $[-\pi, \pi]$ |

**The root cause of aliasing:**

$e^{j\omega_0 n}$ is $2\pi$-periodic: $e^{j(\omega_0 + 2\pi)n} = e^{j\omega_0 n}$. Consequently, analog frequencies $f_0$ and $f_0 + kf_s$ (differing by any integer multiple of $f_s$) map to the same digital frequency $\omega_0$ and become indistinguishable after sampling. This is the fundamental cause of **aliasing**, and explains why an anti-aliasing lowpass filter with cutoff $f_c = f_s/2$ (i.e., $\omega_c = \pi$) must be applied before sampling.

---

# Part II: Transforms for Discrete-Time Signals

> 📖 Textbook §2.2 (Transform-Domain Representation, §2.2.1–§2.2.4)

---

## 2.1 Discrete-Time Fourier Transform (DTFT)

### 2.1.1 Definition and Physical Meaning

The **DTFT** transforms a discrete-time sequence $x(n)$ into a continuous function of frequency $\omega$:

$$\boxed{X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x(n)\, e^{-j\omega n}} \qquad \text{(Forward Transform)}$$

$$\boxed{x(n) = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega})\, e^{j\omega n}\, d\omega} \qquad \text{(Inverse Transform)}$$

**Physical meaning — DTFT as z-transform on the unit circle**:

The z-transform is $X(z) = \sum_{n=-\infty}^{\infty} x(n) z^{-n}$. Setting $z = e^{j\omega}$ (i.e., evaluating on the unit circle $\lvert z\rvert = 1$):

$$X(e^{j\omega}) = X(z)\big\rvert_{z = e^{j\omega}}$$

The DTFT exists when the ROC (Region of Convergence) of $X(z)$ includes the unit circle.

*Note: The DTFT is obtained by evaluating the Z-transform on the unit circle ($z = e^{j\omega}$, $\lvert z\rvert = 1$). The ROC is the region in the complex $z$-plane where the Z-transform sum $\sum_n x(n)z^{-n}$ converges to a finite value. A **pole** is a value of $z$ at which $X(z) \to \infty$ (denominator of the rational $X(z)$ equals zero); the ROC therefore never contains any pole. For a causal stable system, all poles lie strictly inside the unit circle, so the ROC extends outward from the outermost pole and includes the unit circle — hence the DTFT exists. If any pole lies on or outside the unit circle, the ROC excludes the unit circle and the DTFT does not exist.*

**Key structural properties**:
- $X(e^{j\omega})$ is **$2\pi$-periodic**: $X(e^{j(\omega+2\pi)}) = X(e^{j\omega})$
- For real $x(n)$: $X(e^{-j\omega}) = X^*(e^{j\omega})$ — conjugate symmetry, so the magnitude spectrum is even and the phase spectrum is odd

### 2.1.2 Main Properties of the DTFT

| Property | Time Domain | $\leftrightarrow$ | Frequency Domain |
|----------|-------------|---|-----------------|
| Linearity | $ax(n) + by(n)$ | $\leftrightarrow$ | $aX(e^{j\omega}) + bY(e^{j\omega})$ |
| Time shift | $x(n - n_0)$ | $\leftrightarrow$ | $e^{-j\omega n_0} X(e^{j\omega})$ |
| Frequency shift | $e^{j\omega_0 n} x(n)$ | $\leftrightarrow$ | $X(e^{j(\omega - \omega_0)})$ |
| Modulation | $x(n)\cos\omega_0 n$ | $\leftrightarrow$ | $\frac{1}{2}[X(e^{j(\omega-\omega_0)}) + X(e^{j(\omega+\omega_0)})]$ |
| Conjugate | $x^{*}(n)$ | $\leftrightarrow$ | $X^{*}(e^{-j\omega})$ |
| Time reversal | $x(-n)$ | $\leftrightarrow$ | $X(e^{-j\omega})$ |
| Convolution | $x(n) \ast y(n)$ | $\leftrightarrow$ | $X(e^{j\omega})Y(e^{j\omega})$ |
| Correlation | $\sum_k x(k)y^{*}(k-n)$ | $\leftrightarrow$ | $X(e^{j\omega})Y^{*}(e^{j\omega})$ |
| Multiplication | $x(n)y(n)$ | $\leftrightarrow$ | $\frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\theta})Y(e^{j(\omega-\theta)})d\theta$ |
| Frequency differentiation | $n\cdot x(n)$ | $\leftrightarrow$ | $j\dfrac{d}{d\omega}X(e^{j\omega})$ |
| Parseval's theorem | $\sum_{n} x(n)y^*(n)$ | $=$ | $\frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega})Y^*(e^{j\omega})d\omega$ |

**Energy form of Parseval's theorem** (setting $y = x$):

$$\sum_{n=-\infty}^{\infty} \lvert x(n)\rvert^2 = \frac{1}{2\pi}\int_{-\pi}^{\pi} \lvert X(e^{j\omega})\rvert^2\, d\omega$$

Total energy is preserved between the time and frequency domains.

> ![Figure 2.1](<./CourseADSP2026/Fig/fig_2_1.png>)
>
> *Figure 2.1: LTI system operation in the frequency domain — multiplying the input spectrum by the frequency response yields the output spectrum, shown for an aperiodic input (top) and a periodic input (bottom).*

---

## 2.2 z-Transform

### 2.2.1 Definition and Region of Convergence

The **z-transform** generalizes the DTFT to the complex plane:

$$\boxed{X(z) = \sum_{n=-\infty}^{\infty} x(n)\, z^{-n}, \quad z \in \mathbb{C}}$$

The **Region of Convergence (ROC)** is the set of $z$ values for which the sum converges absolutely:

$$\text{ROC} = \lbrace z \in \mathbb{C} : \sum_{n=-\infty}^{\infty} \lvert x(n)\rvert\, \lvert z\rvert^{-n} \lt \infty \rbrace$$

The ROC takes the form of an annulus $r_1 \lt \lvert z\rvert \lt r_2$ (two-sided sequences), $\lvert z\rvert \gt r_1$ (right-sided/causal), or $\lvert z\rvert \lt r_2$ (left-sided).

**Inverse z-transform** via contour integration:

$$x(n) = \frac{1}{2\pi j} \oint_C X(z) z^{n-1}\, dz$$

where $C$ is a counterclockwise contour within the ROC. In practice, inverse z-transforms are computed via **partial fraction expansion**.

### 2.2.2 Main Properties of the z-Transform

| Property | Time Sequence | $\leftrightarrow$ | z-Transform | ROC |
|----------|--------------|---|-------------|-----|
| Linearity | $ax(n)+by(n)$ | $\leftrightarrow$ | $aX(z)+bY(z)$ | At least $\text{ROC}_x \cap \text{ROC}_y$ |
| Time shift | $x(n-K)$ | $\leftrightarrow$ | $z^{-K}X(z)$ | $\text{ROC}_x$ (modified at $z=0,\infty$) |
| z-domain scaling | $\alpha^n x(n)$ | $\leftrightarrow$ | $X(z/\alpha)$ | $\lvert\alpha\rvert\,r_1 \lt \lvert z\rvert \lt \lvert\alpha\rvert\,r_2$ |
| Conjugate | $x^{*}(n)$ | $\leftrightarrow$ | $X^{*}(z^{*})$ | $\text{ROC}_x$ |
| Time reversal | $x(-n)$ | $\leftrightarrow$ | $X(1/z)$ | $1/r_2 \lt \lvert z\rvert \lt 1/r_1$ |
| Convolution | $x(n)\ast y(n)$ | $\leftrightarrow$ | $X(z)Y(z)$ | At least $\text{ROC}_x \cap \text{ROC}_y$ |
| Correlation | $r_{xy}(n) = x(n)*y(-n)$ | $\leftrightarrow$ | $X(z)Y(z^{-1})$ | — |
| z-domain differentiation | $n\cdot x(n)$ | $\leftrightarrow$ | $-z\dfrac{d}{dz}X(z)$ | $\text{ROC}_x$ |
| Initial value | $x(0)$, causal $x$ | — | $\lim_{z\to\infty} X(z)$ | — |
| Parseval | $\sum_n x(n)y^\ast(n)$ | $=$ | $\frac{1}{2\pi j}\oint X(v)Y^\ast(1/v^\ast)v^{-1}\,dv$ | — |

### 2.2.3 Common z-Transform Pairs

| Sequence $x(n)$ | z-Transform $X(z)$ | ROC |
|-----------------|-------------------|-----|
| $\delta(n)$ | $1$ | All $z$ |
| $u(n)$ (unit step, see §1.5.2) | $\dfrac{1}{1-z^{-1}}$ | $\lvert z\rvert \gt 1$ |
| $\alpha^n u(n)$ | $\dfrac{1}{1-\alpha z^{-1}}$ | $\lvert z\rvert \gt \lvert\alpha\rvert$ |
| $-\alpha^n u(-n-1)$ | $\dfrac{1}{1-\alpha z^{-1}}$ | $\lvert z\rvert \lt \lvert\alpha\rvert$ |
| $n\alpha^n u(n)$ | $\dfrac{\alpha z^{-1}}{(1-\alpha z^{-1})^2}$ | $\lvert z\rvert \gt \lvert\alpha\rvert$ |
| $r^n \cos(\omega_0 n) u(n)$ | $\dfrac{1-r\cos\omega_0\, z^{-1}}{1-2r\cos\omega_0\, z^{-1}+r^2z^{-2}}$ | $\lvert z\rvert \gt r$ |
| $r^n \sin(\omega_0 n) u(n)$ | $\dfrac{r\sin\omega_0\, z^{-1}}{1-2r\cos\omega_0\, z^{-1}+r^2z^{-2}}$ | $\lvert z\rvert \gt r$ |
| $\alpha^n u(n) - \alpha^n u(n-N)$ | $\dfrac{1-\alpha^N z^{-N}}{1-\alpha z^{-1}}$ | $\lvert z\rvert \gt 0$ |
| $\alpha^{\lvert n\rvert}$ | $\dfrac{1-\alpha^2}{(1-\alpha z^{-1})(1-\alpha z)}$ | $\lvert\alpha\rvert \lt \lvert z\rvert \lt 1/\lvert\alpha\rvert$ |

### 2.2.4 Relationship Between the z-Transform and the Laplace Transform

The z-transform is the **discrete-time counterpart** of the Laplace transform. The correspondence is established through the sampling relationship $z = e^{sT_s}$:

$$s = \sigma + j\Omega \xrightarrow{z = e^{sT_s}} z = e^{\sigma T_s}\, e^{j\Omega T_s} = r\, e^{j\omega}$$

where each variable carries a distinct physical meaning:

- $s = \sigma + j\Omega$ is the **complex frequency variable** of the Laplace transform ($s \in \mathbb{C}$), whose real and imaginary parts are independent:
  - $\sigma = \mathrm{Re}(s)$: the **damping factor** (units: Np/s). $\sigma < 0$ → exponentially decaying signal; $\sigma > 0$ → exponentially growing signal; $\sigma = 0$ → pure sinusoid (sustained oscillation).
  - $\Omega = \mathrm{Im}(s)$: the **analog angular frequency** (units: rad/s), i.e., the rate of oscillation in continuous time.

- $T_s$: the **sampling period** (units: s/sample). Its reciprocal $f_s = 1/T_s$ is the **sampling rate** (units: Hz = samples/s).

- $z = r\,e^{j\omega}$ is the **complex frequency variable** of the z-transform ($z \in \mathbb{C}$), expressed in polar form:
  - $r = \lvert z\rvert = e^{\sigma T_s}$: the **magnitude** of $z$, encoding the growth/decay rate after sampling. $r < 1$ (inside the unit circle) $\leftrightarrow$ $\sigma < 0$ (stable); $r > 1$ (outside) $\leftrightarrow$ $\sigma > 0$ (unstable); $r = 1$ (on the unit circle) $\leftrightarrow$ $\sigma = 0$ (marginally stable, pure sinusoidal response).
  - $\omega = \angle z = \Omega T_s = \Omega / f_s$: the **digital angular frequency** (units: rad/sample), the discrete-time counterpart of $\Omega$. It is obtained by normalizing $\Omega$ by the sampling period; it is $2\pi$-periodic by construction, with $\omega \in [-\pi, \pi]$ corresponding to the analog band $\Omega \in [-\pi/T_s,\, \pi/T_s]$ (rad/s), or equivalently $f \in [-f_s/2,\, f_s/2]$ (Hz) (one full traversal of the Nyquist interval).

**Key correspondences**:

| Laplace / s-domain | z-Transform / z-domain |
|-------------------|----------------------|
| Left half-plane ($\sigma \lt 0$) | Interior of unit circle ($\lvert z\rvert \lt 1$) |
| Right half-plane ($\sigma \gt 0$) | Exterior of unit circle ($\lvert z\rvert \gt 1$) |
| Imaginary axis ($\sigma = 0$, $s = j\Omega$) | Unit circle ($\lvert z\rvert = 1$) |
| Stable causal: poles in left half-plane | Stable causal: poles inside unit circle |
| Analog frequency $\Omega$ (rad/s) | Digital frequency $\omega = \Omega T_s$ (rad/sample) |

**Frequency mapping**: As $\Omega$ traverses $[-\pi/T_s,\; \pi/T_s]$ (rad/s), equivalently as analog cyclic frequency $f$ traverses $[-f_s/2,\; f_s/2]$ (Hz), the digital frequency $\omega = \Omega T_s$ completes one full traversal of $[-\pi, \pi]$ — one loop around the unit circle. This mapping is the theoretical basis of the **bilinear z-transform** and **impulse invariance** methods for IIR filter design.

> ![Figure 2.2](<./CourseADSP2026/Fig/fig_2_2.jpg>)
>
> *Figure 2.2: s-plane to z-plane mapping via $z = e^{sT_s}$.*

---

## 2.3 Discrete Fourier Transform (DFT)

### 2.3.1 From DTFT to DFT: Uniform Frequency Sampling

The DTFT $X(e^{j\omega})$ is continuous in $\omega \in [-\pi, \pi]$. For numerical computation, we sample it at $N$ equally spaced frequencies:

$$\omega_k = \frac{2\pi k}{N}, \quad k = 0, 1, \ldots, N-1$$

This yields the **DFT**:

$$\boxed{X(k) = \sum_{n=0}^{N-1} x(n)\, W_N^{nk}} \qquad k = 0, 1, \ldots, N-1$$

$$\boxed{x(n) = \frac{1}{N}\sum_{k=0}^{N-1} X(k)\, W_N^{-nk}} \qquad n = 0, 1, \ldots, N-1$$

where $W_N = e^{-j2\pi/N}$ is the **twiddle factor**.

The DFT can be written in matrix form:

$$\mathbf{X} = \mathbf{W}_N \mathbf{x}$$

where the matrix entries are:

$$[\mathbf{W}_N]_{kn} = W_N^{kn}, \qquad k,n = 0, 1, \ldots, N-1$$

Thus $\mathbf{W}_N$ is the $N\times N$ DFT matrix. The inverse uses:

$$\mathbf{W}_N^{-1} = \frac{1}{N}\mathbf{W}_N^*$$

### 2.3.2 Relationship Between DFT and DFS (Discrete Fourier Series)

The DFT and the DFS are closely related but apply to different objects: the DFT operates on a *finite-length* sequence; the DFS operates on an *infinite periodic* sequence. Understanding the bridge between them is essential for correctly interpreting circular convolution, time-domain aliasing, and the DFT convolution theorem.

#### Step 1 — The DFS: transform for periodic sequences

The **Discrete Fourier Series (DFS)** is the frequency-domain representation of a sequence $\tilde{x}(n)$ that is periodic with period $N$, i.e., $\tilde{x}(n + N) = \tilde{x}(n)$ for all $n \in \mathbb{Z}$.

Because $\tilde{x}(n)$ repeats every $N$ samples, it can be decomposed exactly into $N$ complex exponentials with frequencies $\omega_k = 2\pi k/N$:

$$\boxed{\tilde{X}(k) = \sum_{n=0}^{N-1} \tilde{x}(n)\, W_N^{nk}} \qquad \text{(DFS analysis, } k \in \mathbb{Z}\text{)}$$

$$\boxed{\tilde{x}(n) = \frac{1}{N}\sum_{k=0}^{N-1} \tilde{X}(k)\, W_N^{-nk}} \qquad \text{(DFS synthesis, } n \in \mathbb{Z}\text{)}$$

Two important structural facts:

1. **Both sides are periodic with period $N$.** The time-domain sequence $\tilde{x}(n)$ is periodic by assumption. The frequency-domain coefficients $\tilde{X}(k)$ are also periodic: $\tilde{X}(k+N) = \tilde{X}(k)$. This can be verified directly — replacing $k$ by $k+N$ in the analysis formula gives $W_N^{n(k+N)} = W_N^{nk} \cdot W_N^{nN} = W_N^{nk} \cdot 1 = W_N^{nk}$, so the sum is unchanged. The DFS thus describes a *doubly periodic* object: infinite in both time and frequency, with period $N$ in both domains.

2. **The DFS sum only needs one period.** Even though $\tilde{x}(n)$ extends over all integers, the analysis sum only runs over one period $n = 0, \ldots, N-1$ (any consecutive $N$ samples give the same result). This is the key fact exploited in Step 3 below.

#### Step 2 — Periodic extension of a finite-length sequence

Given a finite-length sequence $x(n)$ supported on $0 \leq n \leq N-1$, define its **periodic extension** with period $N$:

$$\tilde{x}(n) \triangleq x(\langle n \rangle_N), \qquad \langle n \rangle_N \triangleq n \bmod N$$

The operator $\langle \cdot \rangle_N$ wraps any integer $n$ into the range $[0, N-1]$ by taking the remainder after dividing by $N$. In other words, $\tilde{x}(n)$ tiles the finite block $x(0), x(1), \ldots, x(N-1)$ infinitely in both directions:

$$\ldots,\; \underbrace{x(0),\, x(1),\, \ldots,\, x(N-1)}_{\text{original block}},\; \underbrace{x(0),\, x(1),\, \ldots,\, x(N-1)}_{\text{copy}},\; \ldots$$

On the canonical interval $0 \leq n \leq N-1$, the two sequences agree exactly: $\tilde{x}(n) = x(n)$.

#### Step 3 — DFT equals one period of the DFS

Now compute the DFS coefficients of $\tilde{x}(n)$:

$$\tilde{X}(k) = \sum_{n=0}^{N-1} \tilde{x}(n)\, W_N^{nk}$$

On the summation interval $0 \leq n \leq N-1$, we have $\tilde{x}(n) = x(n)$ (from Step 2). Substituting:

$$\tilde{X}(k) = \sum_{n=0}^{N-1} x(n)\, W_N^{nk} = X(k)$$

The right-hand side is exactly the DFT definition from §2.3.1. Therefore:

$$\boxed{\tilde{X}(k) = X(k), \quad k = 0, 1, \ldots, N-1}$$

The DFT values $X(k)$ are simply *one period of the (doubly-periodic) DFS coefficients* $\tilde{X}(k)$. The DFT isolates and works with just this one representative period, while the DFS describes the full periodic structure in both domains.

#### Summary: DFT vs. DFS at a glance

| | **DFS** | **DFT** |
|---|---|---|
| Input | Infinite periodic sequence $\tilde{x}(n)$, period $N$ | Finite-length sequence $x(n)$, $0 \leq n \leq N-1$ |
| Output | Infinite periodic coefficients $\tilde{X}(k)$, period $N$ | $N$ complex numbers $X(k)$, $k = 0, \ldots, N-1$ |
| Relationship | Both domains periodic; extends over all $n, k \in \mathbb{Z}$ | One period of the DFS: $X(k) = \tilde{X}(k)$ for $k = 0,\ldots, N-1$ |
| Role | Theoretical framework; explains circular convolution | Practical numerical tool; computed by the FFT |

**Key implication**: Since the DFT is just one period of the DFS, and the DFS diagonalizes *periodic* (circular) convolution, the DFT likewise diagonalizes circular convolution — giving the convolution theorem $x(n) \circledast y(n) \overset{\text{DFT}}{\longleftrightarrow} X(k) \cdot Y(k)$. To compute *linear* convolution of sequences of lengths $L_1$ and $L_2$ via the DFT, the transform length must satisfy $N \geq L_1 + L_2 - 1$ (with zero-padding); otherwise the periodic wrap-around causes the tails of the linear convolution to fold back onto the result — a phenomenon called **time-domain aliasing**.

### 2.3.3 Main Properties and Uses of the DFT

| Property | Condition | Result |
|----------|-----------|--------|
| Linearity | — | $\mathrm{DFT}\lbrace a x(n)+b y(n)\rbrace = aX(k)+bY(k)$ |
| Circular shift | $x(\langle n-m\rangle_N)$ | $W_N^{mk} X(k)$ |
| Circular convolution | $x(n) \circledast y(n)$ | $X(k)\cdot Y(k)$ |
| Multiplication | $x(n)\cdot y(n)$ | $\frac{1}{N} X(k) \circledast Y(k)$ |
| Conjugate symmetry | $x(n)$ real | $X(N-k) = X^*(k)$ |
| Parseval's theorem | — | $\sum_{n=0}^{N-1}\lvert x(n)\rvert^2 = \frac{1}{N}\sum_{k=0}^{N-1}\lvert X(k)\rvert^2$ |

**Primary uses**:
1. **Spectral analysis**: Estimates frequency content of a finite-duration signal
2. **Fast linear convolution**: FIR filtering via FFT (overlap-add, overlap-save methods)
3. **Correlation computation**: Cross-correlation and autocorrelation via FFT
4. **Filter design**: Frequency-sampling method

---

## 2.4 Fast Fourier Transform (FFT)

### 2.4.1 Divide-and-Conquer Strategy and Twiddle Factor Properties

Direct DFT computation requires $O(N^2)$ complex multiplications and additions. The FFT exploits two fundamental properties of the twiddle factor $W_N = e^{-j2\pi/N}$:

**Periodicity**:

$$W_N^{r+N} = W_N^r$$

**Symmetry**:

$$W_N^{r+N/2} = -W_N^r, \qquad W_N^{N/2} = -1, \qquad W_{2N}^{2r} = W_N^r$$

Using a divide-and-conquer strategy, an $N$-point DFT ($N = 2^m$) is recursively split into two $N/2$-point DFTs, reducing complexity from $O(N^2)$ to $O(N\log_2 N)$.

### 2.4.2 Radix-2 DIT and DIF FFT

#### Decimation-in-Time (DIT) FFT

Split $x(n)$ into even-indexed and odd-indexed subsequences:

$$x_1(n) = x(2n), \quad x_2(n) = x(2n+1), \quad n = 0, 1, \ldots, N/2-1$$

Then:

$$X(k) = \underbrace{\sum_{n=0}^{N/2-1} x(2n)\, W_{N/2}^{nk}}_{U(k)} + W_N^k \underbrace{\sum_{n=0}^{N/2-1} x(2n+1)\, W_{N/2}^{nk}}_{V(k)}$$

Using the symmetry $W_N^{k+N/2} = -W_N^k$, the **butterfly equations** are:

$$\boxed{X(k) = U(k) + W_N^k V(k)} \qquad k = 0, 1, \ldots, N/2-1$$

$$\boxed{X(k+N/2) = U(k) - W_N^k V(k)}$$

Apply recursively for $m = \log_2 N$ stages.

> ![Figure 2.3](<./CourseADSP2026/Fig/fig_2_3.jpg>)
>
> *Figure 2.3: 8-point DIT-FFT signal flow graph.*

#### Decimation-in-Frequency (DIF) FFT

Split $x(n)$ into the first half and second half:

$$X(2k) = \sum_{n=0}^{N/2-1}[x(n) + x(n+N/2)]\, W_{N/2}^{nk}$$

$$X(2k+1) = \sum_{n=0}^{N/2-1}[x(n) - x(n+N/2)]\, W_N^n\, W_{N/2}^{nk}$$

**Comparison**:

| | DIT-FFT | DIF-FFT |
|---|---------|---------|
| Input order | Bit-reversed | Natural |
| Output order | Natural | Bit-reversed |
| Structure | Butterfly inputs combined | Butterfly outputs split |
| Complexity | $O(N\log_2 N)$ | $O(N\log_2 N)$ |

### 2.4.3 Computational Complexity

For $N = 2^m$:

| Algorithm | Complex Multiplications | Complex Additions |
|-----------|------------------------|------------------|
| Direct DFT | $N^2$ | $N(N-1)$ |
| Radix-2 FFT | $\dfrac{N}{2}\log_2 N$ | $N\log_2 N$ |
| Speedup | $\dfrac{2N}{\log_2 N}$ | $\approx \dfrac{N}{\log_2 N}$ |

**Example** ($N = 1024 = 2^{10}$): Direct DFT requires $1{,}048{,}576$ multiplications; the FFT requires only $5{,}120$ — a speedup of $\approx 200\times$.

### 2.4.4 Bit-Reversal Permutation

> **Why DIT requires bit-reversed input.** Recall that DIT (Decimation-in-Time, §2.4.2) splits the input **recursively by even/odd indices** at each stage:
> - Stage 1: separate by bit 0 (even/odd) → two groups of $N/2$
> - Stage 2: within each group, separate by bit 1 → four groups of $N/4$
> - $\vdots$
> - Stage $m$: separate by bit $m-1$ → $N$ individual samples
>
> Because each stage sorts by one bit, starting from the **rightmost bit** (bit 0, which determines even/odd) and moving leftward to bit $m-1$, after all $m$ stages the element originally at index $n$ ends up at the position whose binary representation is $n$'s bits **written in reverse order**. This bit-reversed ordering is therefore not an arbitrary pre-processing step — it is the **natural outcome of DIT's recursive even/odd decomposition**, baked into the algorithm itself.
>
> By contrast, **DIF** (Decimation-in-Frequency, §2.4.2) splits the *output* spectrum first (even/odd frequency bins), so the *output* ends up in bit-reversed order while the input remains in natural order — a perfect input/output duality between DIT and DIF.

In **DIT-FFT** (Decimation-in-Time FFT), the input sequence must therefore be reordered into **bit-reversed** order before the butterfly stages can proceed. The bit-reversed index is obtained by reversing the $m$-bit binary representation of $n$:

| Index $n$ | Binary ($m=3$) | Bit-reversed | Reordered index |
|-----------|---------------|-------------|-----------------|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

Here $\operatorname{br}(n)$ denotes the **bit-reversed index** of $n$. If the $m$-bit binary representation of $n$ is

$$
n = (b_{m-1}\cdots b_1 b_0)_2,
$$

then

$$
\operatorname{br}(n) = (b_0 b_1 \cdots b_{m-1})_2,
$$

which is obtained by mirroring those bits.

For example, when $m=3$:

$$
\operatorname{br}(1) = \operatorname{br}(001_2) = 100_2 = 4,
$$

and

$$
\operatorname{br}(3) = \operatorname{br}(011_2) = 110_2 = 6,
$$

consistent with the table above.

The permutation can be performed in-place with an $O(N)$ algorithm: compare each index with its bit-reverse and swap if $n \lt \text{br}(n)$. The condition $n < \text{br}(n)$ ensures each pair is swapped exactly once; indices satisfying $\text{br}(n) = n$ (such as $0$ and $7$ in the $N=8$ example) are already in their correct positions and require no action.

**FFT variants**:
- **Radix-4 FFT**: Groups of 4; further reduces multiplicative count
- **Split-radix FFT**: Best known operation count for power-of-2 $N$
- **Mixed-radix FFT**: Handles arbitrary $N$ by factoring $N = N_1 \times N_2 \times \cdots$ (used in FFTW)

---

# Part III: Digital Filter Structures and Design

> 📖 Textbook §2.3 (Discrete-Time Systems); §2.4 (Minimum-Phase and System Invertibility); §2.5 (Lattice Filter Realizations)

---

## 3.0 Characterizing Discrete-Time Systems

> **Guiding question**: How do we characterize a discrete-time system and design it to perform a desired signal transformation?

A **discrete-time LTI (Linear Time-Invariant) system** is any processor that maps an input sequence $x(n)$ to an output sequence $y(n)$ while satisfying two properties:
- **Linearity**: $\mathcal{H}\lbrace ax_1(n) + bx_2(n)\rbrace = a\mathcal{H}\lbrace x_1(n)\rbrace + b\mathcal{H}\lbrace x_2(n)\rbrace$
- **Time-invariance**: if $y(n) = \mathcal{H}\lbrace x(n)\rbrace$, then $\mathcal{H}\lbrace x(n-k)\rbrace = y(n-k)$ for any integer $k$

These two properties together guarantee that an LTI system is *completely characterized* by a single function — its **impulse response** $h(n)$ — and that system behavior in frequency is described by the **frequency response** $H(e^{j\omega})$. The difference equation is the computational recipe for implementing the system.

---

### 3.0.1 The Difference Equation — Computational Recipe for a Filter

The most general form of a causal LTI digital filter is the **$N$-th order Linear Constant-Coefficient Difference Equation (LCCDE)**:

$$\boxed{y(n) = \underbrace{-\sum_{k=1}^{N} b_k\, y(n-k)}_{\text{feedback (IIR part)}} + \underbrace{\sum_{k=0}^{M} a_k\, x(n-k)}_{\text{feedforward (FIR part)}}}$$

or in the equivalent symmetric form (dividing through and setting $b_0 = 1$):

$$\sum_{k=0}^{N} b_k\, y(n-k) = \sum_{k=0}^{M} a_k\, x(n-k)$$

**What each term means physically**: At each time step $n$, the new output $y(n)$ is formed by two contributions: (i) a weighted sum of the current and past $M$ *inputs* $x(n), x(n-1), \ldots, x(n-M)$ — the **feedforward path**, which "looks backward" through the input — and (ii) a weighted sum of the past $N$ *outputs* $y(n-1), \ldots, y(n-N)$ — the **feedback path**, which feeds the system's own past decisions back into the computation.

> **Why does feedback matter so much?** Feedback is what makes a filter's impulse response potentially infinite. A single tap of feedback — even just $y(n) = a\,y(n-1) + x(n)$ — means the output at time $n$ depends on $y(n-1)$, which depended on $y(n-2)$, and so on: the influence of a single input impulse propagates indefinitely into the future. Remove all feedback (set all $b_k = 0$, $k \geq 1$), and the equation collapses to a pure weighted sum of inputs — the impulse response is finite by construction.

**The FIR/IIR split directly from the difference equation:**

| Condition | Filter type | Equation reduces to |
|---|---|---|
| All $b_k = 0$ for $k \geq 1$ | **FIR** | $y(n) = \sum_{k=0}^{M} a_k\, x(n-k)$ — pure feedforward |
| At least one $b_k \neq 0$ | **IIR** | Full **Linear Constant-Coefficient Difference Equation (LCCDE)** with feedback: $y(n) = -\sum_{k=1}^{N} b_k\,y(n-k) + \sum_{k=0}^{M} a_k\,x(n-k)$ |

The z-transform of the LCCDE (assuming zero initial conditions) gives the **transfer function** directly. Taking $\mathcal{Z}\lbrace\cdot\rbrace$ of both sides and using the shift property $\mathcal{Z}\lbrace x(n-k)\rbrace = z^{-k} X(z)$:

$$\left(\sum_{k=0}^{N} b_k\, z^{-k}\right) Y(z) = \left(\sum_{k=0}^{M} a_k\, z^{-k}\right) X(z)$$

$$\boxed{H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} a_k\, z^{-k}}{\sum_{k=0}^{N} b_k\, z^{-k}} = \frac{A(z)}{B(z)}}$$

**FIR**: denominator $B(z) = 1$ (constant); $H(z)$ is a polynomial — all poles are at the origin $z = 0$.
**IIR**: $B(z)$ is a nontrivial polynomial; $H(z)$ is rational — poles can be anywhere in the $z$-plane.

---

### 3.0.2 The Impulse Response — Complete Characterization of an LTI System

**Definition**: The **impulse response** $h(n)$ is the output of the system when the input is a unit impulse:

$$h(n) = \mathcal{H}\lbrace\delta(n)\rbrace \quad \text{(with zero initial conditions)}$$

**Why $h(n)$ completely determines the system**: Any input signal can be decomposed into a weighted sum of shifted impulses (the sifting property):

$$x(n) = \sum_{k=-\infty}^{\infty} x(k)\, \delta(n-k)$$

By linearity, the system responds to each $x(k)\,\delta(n-k)$ with $x(k)\,h(n-k)$ (scaled + shifted impulse response). By time-invariance, those shifted responses are exact copies. Summing over all $k$:

$$\boxed{y(n) = x(n) \ast h(n) = \sum_{k=-\infty}^{\infty} x(k)\, h(n-k)}$$

This is the **convolution sum** — the universal input-output relationship for any LTI system, regardless of structure. Knowing $h(n)$ means knowing the system's response to *every* possible input.

> **Intuitive picture**: $h(n)$ is the system's "memory trace" — the record of how long and how strongly the system rings after being struck by a single unit impulse. A short, decaying $h(n)$ means the system forgets its input quickly. A long $h(n)$ means the system "remembers" the input for many samples — either because the designer deliberately chose many taps (FIR), or because feedback causes the response to decay slowly (IIR).

**Computing $h(n)$ from the difference equation**: Set $x(n) = \delta(n)$ and solve the LCCDE with zero initial conditions. For an FIR filter:

$$h(n) = \sum_{k=0}^{M} a_k\, \delta(n-k) = \begin{cases} a_n & 0 \leq n \leq M \\ 0 & \text{otherwise} \end{cases}$$

The impulse response *is* the coefficient sequence. For an IIR filter, the feedback generates an infinite tail. **Canonical example** — first-order recursive filter $y(n) = a\,y(n-1) + x(n)$, $\lvert a\rvert < 1$:

$$h(n) = a^n\, u(n) = \begin{cases} a^n & n \geq 0 \\ 0 & n < 0 \end{cases}$$

Even though the difference equation has only two terms, the impulse response is a geometrically decaying infinite sequence. This single pole at $z = a$ can approximate a sharp lowpass or highpass filter far more efficiently than an FIR would.

> **Deep dive: why one pole, and why can it be both lowpass and highpass?**
>
> **① Why just one pole**
>
> The first-order recursive filter $y(n) = ay(n-1) + x(n)$ has transfer function:
>
> $$H(z) = \frac{1}{1 - az^{-1}} = \frac{z}{z - a}$$
>
> The denominator is linear in $z$, so it has exactly **one root — the pole at $z = a$**. There is also a zero at $z = 0$ (the origin), but a zero at the origin contributes no frequency-selective behavior: its distance from any point $e^{j\omega}$ on the unit circle is always $\lvert e^{j\omega} - 0\rvert = 1$ — a constant. So all the filter's frequency shaping comes from the single pole alone.
>
> **② Geometric interpretation of the frequency response**
>
> Evaluating $H(z)$ on the unit circle ($z = e^{j\omega}$):
>
> $$\lvert H(e^{j\omega})\rvert = \frac{1}{\lvert e^{j\omega} - a\rvert}$$
>
> The denominator $\lvert e^{j\omega} - a\rvert$ is the **Euclidean distance** in the complex plane from the unit-circle point $e^{j\omega}$ to the pole $z = a$. As $\omega$ sweeps from $0$ to $\pi$, this distance changes, and the magnitude response is **large wherever the unit circle passes closest to the pole** and small where it is farthest away.
>
> ```
>       Im
>        |         pole at z=a (positive real → near z=+1 → near ω=0)
>        |      ↑ large gain at ω=0 (DC)
>  ------+------●------------ Re
>        |   (a,0)    z=+1 is close to the pole → small distance → large magnitude
>        |
>        |       z=−1 is far from the pole → large distance → small magnitude
> ```
>
> **③ Lowpass filter: $0 < a < 1$ (pole on the positive real axis)**
>
> The pole sits close to the point $z = +1$ (which corresponds to DC, $\omega = 0$):
>
> | Frequency | Unit-circle point | Distance to pole $a$ | $\lvert H\rvert$ |
> |-----------|-------------------|-----------------------|-------|
> | DC ($\omega = 0$) | $e^{j0} = +1$ | $1 - a$ (small, since $a \approx 1$) | **large** |
> | Nyquist ($\omega = \pi$) | $e^{j\pi} = -1$ | $1 + a$ (large) | **small** |
>
> Low frequencies are amplified, high frequencies are suppressed → **lowpass behavior**. The closer $a$ is to $1$, the smaller the DC-pole distance, the higher and sharper the low-frequency peak.
>
> **④ Highpass filter: $-1 < a < 0$ (pole on the negative real axis)**
>
> The pole now sits close to $z = -1$ (which corresponds to the Nyquist frequency, $\omega = \pi$):
>
> | Frequency | Unit-circle point | Distance to pole $a < 0$ | $\lvert H\rvert$ |
> |-----------|-------------------|---------------------------|-------|
> | DC ($\omega = 0$) | $+1$ | $1 - a = 1 + \lvert a\rvert$ (large) | **small** |
> | Nyquist ($\omega = \pi$) | $-1$ | $\lvert{-1 - a}\rvert = 1 - \lvert a\rvert$ (small, since $a \approx -1$) | **large** |
>
> High frequencies are amplified, low frequencies are suppressed → **highpass behavior**. Symmetrically, as $a \to -1$, the filter sharpens at the Nyquist end.
>
> **⑤ Why more efficient than FIR**
>
> An FIR filter with $M$ taps achieves its frequency selectivity by choosing $M+1$ coefficients — more taps means sharper cutoff, but always at the cost of $M+1$ multiplications per sample. A single-pole IIR requires **exactly 1 multiplication and 1 addition per sample**, regardless of how sharp the desired rolloff is: sharpening is achieved by pushing $\lvert a\rvert$ closer to $1$ — a change to a single number, not an increase in computation. This is the fundamental computational advantage of poles over taps: a pole close to the unit circle creates a tall, narrow peak in $\lvert H(e^{j\omega})\rvert$ essentially for free.

> ![Figure 3.0a](<./CourseADSP2026/Fig/fig_3_0a.png>)
>
> *Figure 3.0a: Impulse responses contrasted. (Left) FIR: $h(n)$ has finite support — the 8-tap example has exactly 8 nonzero values and decays to zero in finite time. (Right) IIR: first-order recursive filter with $a = 0.8$; $h(n) = 0.8^n u(n)$ decays exponentially but is theoretically nonzero for all $n \geq 0$. The practical consequence: the FIR requires 8 multiplications per sample; the IIR achieves a similarly shaped frequency response with 2.*

**BIBO stability from $h(n)$**: The system is bounded-input bounded-output (BIBO) stable if and only if $h(n)$ is absolutely summable:

$$\sum_{n=-\infty}^{\infty} \lvert h(n)\rvert < \infty$$

For rational $H(z)$, this is equivalent to all poles lying strictly inside the unit circle. FIR filters have all poles at the origin and are therefore **unconditionally stable** regardless of coefficient values — no stability check is ever required.

> **Why poles inside the unit circle ↔ absolute summability: the full argument**
>
> **Step 1 — What is a pole?** A *rational* transfer function is one expressible as a ratio of polynomials in $z^{-1}$ (or equivalently $z$):
> $$H(z) = \frac{B(z)}{A(z)} = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}$$
> The **poles** $\lbrace p_1, p_2, \ldots, p_N\rbrace$ are the roots of the denominator polynomial $A(z)$ — the values of $z$ at which $H(z) \to \infty$. They fully determine the system's natural (unforced) behavior: how the system "rings" after an impulse.
>
> **Step 2 — Partial fraction expansion turns $H(z)$ into geometric sequences.** For a causal system with $N$ distinct poles, the partial fraction expansion gives:
> $$H(z) = \sum_{k=1}^{N} \frac{A_k}{1 - p_k z^{-1}} + (\text{FIR terms if } M \geq N)$$
> Taking the inverse z-transform of each term (using the pair $\dfrac{1}{1 - pz^{-1}} \xrightarrow{\mathcal{Z}^{-1}} p^n u(n)$ for a causal system):
> $$h(n) = \left(\sum_{k=1}^{N} A_k\, p_k^n\right) u(n) + (\text{finite-length FIR part})$$
> Each pole $p_k$ contributes a **geometric sequence** $p_k^n$ to the impulse response. The stability question reduces to: when is $\sum_{n=0}^{\infty} \lvert p_k^n\rvert = \sum_{n=0}^{\infty} \lvert p_k\rvert^n$ finite?
>
> **Step 3 — The geometric series test decides stability.** This is a standard geometric series:
> $$\sum_{n=0}^{\infty} \lvert p_k\rvert^n =
> \begin{cases}
> \dfrac{1}{1 - \lvert p_k\rvert} < \infty, & \text{if } \lvert p_k\rvert < 1 \quad \text{(stable)} \\
> \text{diverges}, & \text{if } \lvert p_k\rvert \geq 1 \quad \text{(unstable)}
> \end{cases}$$
> The three cases have a direct physical picture:
>
> | Pole location | $\lvert p_k\rvert$ | Impulse response contribution | Stability |
> |---|---|---|---|
> | Strictly inside unit circle | $< 1$ | Exponential decay: $\lvert p_k\rvert^n \to 0$ | **BIBO stable** |
> | On the unit circle | $= 1$ | Sustained oscillation (no decay): e.g., $e^{j\omega_0 n}$ | **Marginally stable** (BIBO unstable — a bounded input at resonance produces unbounded output) |
> | Outside unit circle | $> 1$ | Exponential growth: $\lvert p_k\rvert^n \to \infty$ | **Unstable** |
>
> Since $h(n)$ is a sum of such geometric terms, the whole system is BIBO stable if and only if **every** pole satisfies $\lvert p_k\rvert < 1$ — i.e., all poles lie strictly inside the unit circle $\lvert z\rvert = 1$.
>
> **Step 4 — Geometric picture in the z-plane.** The unit circle $\lvert z\rvert = 1$ plays the same role for discrete-time systems that the imaginary axis plays for continuous-time systems in the s-plane (left half-plane = stable, right half-plane = unstable). The mapping $z = e^{sT_s}$ transforms: left half-plane $\operatorname{Re}(s) < 0 \;\Longleftrightarrow\; \lvert e^{sT_s}\rvert = e^{\operatorname{Re}(s)T_s} < 1 \;\Longleftrightarrow\; \lvert z\rvert < 1$. So stability boundaries are directly analogous; the unit circle is simply the discrete-time version of the imaginary axis.
>
> **Step 5 — ROC perspective (equivalent characterization).** For a causal system, the ROC of $H(z)$ is the exterior of a disk: $\operatorname{ROC} = \left\lbrace z : \lvert z\rvert > r_{\max}\right\rbrace$ where $r_{\max} = \max_k \lvert p_k\rvert$ is the magnitude of the outermost pole. The DTFT $H(e^{j\omega})$ exists (and $h(n)$ is absolutely summable) if and only if the unit circle $\lvert z\rvert = 1$ is inside the ROC — i.e., $r_{\max} < 1$ — which is exactly the condition that all poles lie strictly inside the unit circle.
>
> **FIR special case.** An FIR filter $H(z) = \sum_{n=0}^{M} h(n) z^{-n}$ has denominator $A(z) = z^M$ (after multiplying through) — a single pole of multiplicity $M$ at the **origin** $z = 0$. Since $\lvert 0\rvert = 0 < 1$, all FIR poles are trivially inside the unit circle, so FIR filters are unconditionally stable regardless of coefficient values.



---

### 3.0.3 The Frequency Response — What the Filter Does in the Frequency Domain

The **frequency response** $H(e^{j\omega})$ is the DTFT of the impulse response:

$$\boxed{H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h(n)\, e^{-j\omega n}}$$

Equivalently, it is the transfer function $H(z)$ evaluated on the unit circle: $H(e^{j\omega}) = H(z)\big\rvert_{z = e^{j\omega}}$.

**Physical meaning**: $H(e^{j\omega})$ is the complex gain the filter applies to a pure complex sinusoid $e^{j\omega n}$. If $x(n) = e^{j\omega n}$, then:

$$y(n) = H(e^{j\omega})\, e^{j\omega n}$$

The sinusoid passes through *unchanged in shape*, multiplied only by the complex constant $H(e^{j\omega})$ — this is the defining property of an eigenfunction of an LTI system. Because real sinusoids $\cos(\omega n)$ and $\sin(\omega n)$ are superpositions of $e^{\pm j\omega n}$, the filter's effect on any sinusoidal input is completely described by $H(e^{j\omega})$.

**Decomposing the frequency response** into magnitude and phase:

$$H(e^{j\omega}) = \lvert H(e^{j\omega})\rvert\, e^{j\angle H(e^{j\omega})}$$

| Component | Definition | Physical meaning |
|---|---|---|
| **Magnitude response** | $\lvert H(e^{j\omega})\rvert$ | Gain applied to frequency $\omega$; squaring gives power gain |
| **Phase response** | $\angle H(e^{j\omega})$ | Phase shift (in radians) applied to frequency $\omega$ |
| **Group delay** | $\tau(\omega) = -\dfrac{d}{d\omega}\angle H(e^{j\omega})$ | Delay (in samples) experienced by the envelope of a narrowband signal near frequency $\omega$ |

> **Why group delay matters more than phase**: In most signal processing contexts, a constant phase shift $e^{-j\omega n_0}$ merely time-delays all frequencies equally by $n_0$ samples — perfectly acceptable. What causes waveform distortion is *frequency-dependent* delay: different frequency components arriving at different times, smearing the signal in time. Group delay $\tau(\omega)$ directly quantifies this frequency-dependent delay. A filter with **constant group delay** (linear phase response) delays all frequencies equally and therefore transmits waveforms without shape distortion — the key motivation for linear-phase FIR design.

**Reading the frequency response plot**: The frequency axis runs over one period $\omega \in [0, \pi]$ for a real-coefficient filter (by conjugate symmetry, $H(e^{-j\omega}) = H^*(e^{j\omega})$, so the range $[\pi, 2\pi]$ is redundant). $\omega = 0$ is DC; $\omega = \pi$ is the Nyquist frequency $f_s/2$.

**FIR frequency response — a polynomial in $e^{-j\omega}$**:

$$H(e^{j\omega}) = \sum_{n=0}^{M} h(n)\, e^{-j\omega n} = h(0) + h(1)e^{-j\omega} + \cdots + h(M)e^{-j\omega M}$$

This is a trigonometric polynomial in $e^{-j\omega}$, capable of approximating any desired magnitude shape over $[0,\pi]$ given sufficiently large $M$.

**IIR frequency response — a rational function of $e^{-j\omega}$**:

$$H(e^{j\omega}) = \frac{a_0 + a_1 e^{-j\omega} + \cdots + a_N e^{-j\omega N}}{1 + b_1 e^{-j\omega} + \cdots + b_N e^{-j\omega N}}$$

Poles near the unit circle create **resonance peaks** — the magnitude rises sharply near the pole angle. Zeros on the unit circle create **exact nulls** — the magnitude is identically zero at the zero angle. The interaction of poles and zeros shapes the frequency response with great efficiency: an IIR filter of order $N = 6$ can achieve stopband attenuation that an FIR filter would need $M \approx 100$ taps to match.

> ![Figure 3.0b](<./CourseADSP2026/Fig/fig_3_0b.png>)
>
> *Figure 3.0b: Frequency response anatomy. (Top) Magnitude $\lvert H(e^{j\omega})\rvert$ in dB vs. normalized frequency $\omega/\pi \in [0,1]$; passband, transition band, and stopband regions labeled. (Bottom) Phase $\angle H(e^{j\omega})$: linear (constant-slope) for a linear-phase FIR; nonlinear for a typical IIR. Group delay (negative slope of phase) is flat for the FIR and frequency-varying for the IIR.*

**The connection between poles/zeros and frequency response**: The magnitude at any frequency $\omega$ can be read geometrically from the pole-zero plot:

$$\lvert H(e^{j\omega})\rvert = \lvert a_0\rvert \cdot \frac{\prod_k \lvert e^{j\omega} - z_k\rvert}{\prod_k \lvert e^{j\omega} - p_k\rvert}$$

where $z_k$ are zeros and $p_k$ are poles. As the evaluation point $e^{j\omega}$ sweeps around the unit circle, the magnitude is the product of distances to all zeros divided by the product of distances to all poles. A pole *close to* the unit circle at angle $\omega_0$ makes the denominator small near $\omega_0$, creating a magnitude peak. A zero *on* the unit circle at angle $\omega_0$ makes the numerator zero at $\omega_0$, creating a notch.

---

### 3.0.4 FIR vs. IIR — The Fundamental Design Choice

Every filter design begins with choosing between FIR and IIR. The difference equation, impulse response, and frequency response all reflect this architectural choice:

| Property | FIR | IIR |
|---|---|---|
| Difference equation | Pure feedforward: $y(n) = \sum_{k} a_k x(n-k)$ | Feedback + feedforward: $y(n) = -\sum b_k y(n-k) + \sum a_k x(n-k)$ |
| Transfer function $H(z)$ | Polynomial (all poles at $z=0$) | Rational (poles can be anywhere) |
| Impulse response | Finite: $h(n) = 0$ for $n > M$ | Infinite: $h(n) \neq 0$ for all $n \geq 0$ |
| Stability | Unconditional (no poles on/outside unit circle possible) | Must verify all poles inside unit circle |
| Linear phase | Achievable exactly (symmetric $h(n)$) | Not achievable exactly |
| Filter order for sharp cutoff | High (hundreds of taps) | Low (4–12 for Butterworth/Chebyshev) |
| Arithmetic cost per sample | $O(M+1)$ — high for sharp specs | $O(N)$ — low |
| Suitable analog prototype | No natural counterpart | Yes (Butterworth, Chebyshev, Elliptic) |
| Primary design methods | Window, Parks-McClellan, frequency-sampling | Bilinear transform, impulse invariance |
| Typical application | Audio EQ, communications pulse shaping, linear-phase requirements | Anti-aliasing, data acquisition, where order efficiency matters |

**Practical decision guide**:
- **Choose FIR** when the application requires exactly linear phase (distortion-free signal transmission, matched filtering in radar/sonar, image processing) or when guaranteed stability is essential and the computational cost of many taps is acceptable.
- **Choose IIR** when filter order and computational efficiency are the primary constraints, phase distortion is tolerable (or can be corrected separately with an allpass equalizer), and the filter resembles a classical analog prototype (Butterworth, Chebyshev, Elliptic).

> **Worked example**: Suppose you need a lowpass filter with passband $[0, 0.2\pi]$, stopband $[\geq 0.25\pi]$, stopband attenuation $\geq 60$ dB. Using the Kaiser formula (§3.3.2), the FIR order is approximately $M \approx (60 - 7.95)/(2.285 \times 0.05\pi) \approx 145$ taps. An equivalent Elliptic IIR filter achieves the same specifications with order $N = 5$ — a reduction of $28\times$ in coefficient count. But the IIR introduces frequency-dependent group delay; if you need to cascade it with a linear-phase processor, you must also design an allpass phase equalizer or accept the distortion.

---

## 3.1 FIR Filter Implementations

An **FIR (Finite Impulse Response)** filter of order $M$ has transfer function:

$$H(z) = \sum_{n=0}^{M} h(n)\, z^{-n} = h(0) + h(1)z^{-1} + \cdots + h(M)z^{-M}$$

All $M+1$ poles are at the origin $z=0$. FIR filters are **unconditionally stable** and can achieve exactly linear phase.

### 3.1.1 Direct Form — Transversal Filter

The output is a weighted sum of the current and past $M$ inputs:

$$y(n) = \sum_{m=0}^{M} h(m)\, x(n - m)$$

Implemented as a **tapped delay line**: a shift register of $M$ unit delays, each tap multiplied by $h(m)$, all products summed.

- Multiplications per sample: $M+1$
- Additions per sample: $M$
- Delay elements: $M$

> ![Figure 3.1](<./CourseADSP2026/Fig/fig_3_1.png>)
>
> *Figure 3.1: Direct-form (transversal) FIR filter.*

**Symmetric exploitation**: For linear-phase FIR with $h(n) = h(M-n)$, paired coefficients are equal, nearly halving the number of distinct multiplications.

### 3.1.2 Cascade Form

Factor $H(z)$ into real-coefficient second-order sections (biquads):

$$H(z) = h(0)\prod_{k=1}^{\lfloor M/2 \rfloor} H_k(z), \qquad H_k(z) = 1 + b_{1k}z^{-1} + b_{2k}z^{-2}$$

Complex conjugate zero pairs are combined into each real biquad. Each biquad is a 3-tap transversal FIR.

**Advantages**: Numerically more robust than the direct form for high-order filters (coefficient sensitivity is localized to each section); individual zeros are easily modified.

> ![Figure 3.2](<./CourseADSP2026/Fig/fig_3_2.png>)
>
> *Figure 3.2: Cascade FIR implementation.*

### 3.1.3 Recursive Realization and Comb Filter

An FIR filter can sometimes be implemented recursively via pole-zero cancellation. The classic example is the **uniform moving-average filter**:

$$h(n) = \begin{cases} \dfrac{1}{N} & 0 \leq n \leq N-1 \\ 0 & \text{otherwise} \end{cases}$$

Its transfer function factors as:

$$H(z) = \underbrace{\frac{1-z^{-N}}{N}}_{\displaystyle H_1(z):\ \text{comb filter}} \cdot \underbrace{\frac{1}{1-z^{-1}}}_{\displaystyle H_2(z):\ \text{accumulator}}$$

- $H_1(z)$: **comb filter** — $N$ zeros equally spaced at the $N$-th roots of unity
- $H_2(z)$: **first-order IIR accumulator** — pole at $z=1$ cancels the corresponding comb zero

The pole-zero cancellation preserves the FIR character, while the recursive structure reduces computation from $N$ operations to **only 2 per sample**, regardless of $N$.

### 3.1.4 Frequency-Sampling Form

When the desired DFT values $H(k)$ are specified, the transfer function is:

$$H(z) = \frac{1 - z^{-N}}{N} \sum_{k=0}^{N-1} \frac{H(k)}{1 - W_N^{-k} z^{-1}}, \qquad W_N = e^{-j2\pi/N}$$

This is a **comb filter** cascaded with a **bank of $N$ first-order resonators**, each tuned to a DFT frequency $\omega_k = 2\pi k/N$. Transition-band samples can be optimized to minimize stopband ripple.

---

## 3.2 IIR Filter Implementations

An **IIR (Infinite Impulse Response)** filter has transfer function:

$$H(z) = \frac{\sum_{k=0}^{N} a_k z^{-k}}{1 + \sum_{k=1}^{N} b_k z^{-k}}$$

IIR filters achieve sharp frequency selectivity at low order, but cannot have exactly linear phase, and stability must be explicitly ensured (all poles inside the unit circle).

### 3.2.1 Direct Form I

Implements the difference equation with two separate delay chains — one for input terms $x(n-k)$ (FIR part) and one for output feedback terms $y(n-k)$.

- Total delay elements: $2N$

> ![Figure 3.3](<./CourseADSP2026/Fig/fig_3_3.png>)
>
> *Figure 3.3: Direct Form I IIR filter.*

### 3.2.2 Direct Form II — Canonical Form

Merge the two delay chains by sharing a single intermediate state variable $v(n)$ (distinct from the unit step $u(n)$):

$$v(n) = x(n) - \sum_{k=1}^{N} b_k\, v(n-k)$$

$$y(n) = \sum_{k=0}^{N} a_k\, v(n-k)$$

- Delay elements: $N$ — the minimum possible (**canonical form**)

> ![Figure 3.4](<./CourseADSP2026/Fig/fig_3_4.png>)
>
> *Figure 3.4: All-pole system realization (direct form).*

**Transposed Direct Form II**: Reversing the signal flow graph (transpose) yields different accumulation order and superior finite-precision performance.

### 3.2.3 Cascade Form

Factor $H(z)$ into real-coefficient second-order sections:

$$H(z) = \prod_{k=1}^{\lfloor N/2 \rfloor} H_k(z), \qquad H_k(z) = \frac{a_{0k} + a_{1k}z^{-1} + a_{2k}z^{-2}}{1 + b_{1k}z^{-1} + b_{2k}z^{-2}}$$

Each biquad is implemented as Direct Form II. Coefficient sensitivity is localized; the format used by virtually all practical implementations (e.g., MATLAB `sos` format).

### 3.2.4 Parallel Form

Expand $H(z)$ via partial-fraction decomposition:

$$H(z) = C + \sum_{k=1}^{\lfloor N/2 \rfloor} \frac{a_{0k} + a_{1k}z^{-1}}{1 + b_{1k}z^{-1} + b_{2k}z^{-2}}$$

Each branch is an independent second-order section computed in parallel.

**Advantages**: Best finite-precision performance — round-off errors in each section do not accumulate across branches. Suitable for highly parallel hardware.

---

## 3.3 Digital Filter Design

### 3.3.1 Ideal and Practical Frequency Responses

**Ideal filter shapes** (brickwall responses):
- **Lowpass**: passband $[0, \omega_c]$, stopband $[\omega_c, \pi]$
- **Highpass**: stopband $[0, \omega_c]$, passband $[\omega_c, \pi]$
- **Bandpass**: passband $[\omega_1, \omega_2]$
- **Bandstop**: stopband $[\omega_1, \omega_2]$

**Practical filter specification**:

$$1 - \delta_1 \leq \lvert H(e^{j\omega})\rvert \leq 1 + \delta_1 \quad (\text{passband},\ \omega \leq \omega_p)$$

$$\lvert H(e^{j\omega})\rvert \leq \delta_2 \quad (\text{stopband},\ \omega \geq \omega_s)$$

In dB: passband ripple $R_p = -20\log_{10}(1-\delta_1)$ dB; stopband attenuation $A_s = -20\log_{10}(\delta_2)$ dB.

### 3.3.2 FIR Filter Design: Window Method, Frequency-Sampling Method, Kaiser Formula

FIR filters can achieve **exactly linear phase** — a critical property for distortion-free applications.

**① Window Method**

Truncate the ideal (infinite-length) impulse response $h_d(n)$ using a window $w(n)$ of length $M+1$:

$$h(n) = w(n) \cdot h_d(n), \qquad n = 0, 1, \ldots, M$$

| Window | Transition width | Peak sidelobe | Min. stopband attenuation |
|--------|-----------------|--------------|--------------------------|
| Rectangular | $0.9 \times 2\pi/M$ | $-13$ dB | 21 dB |
| Bartlett | $1.8 \times 2\pi/M$ | $-25$ dB | 25 dB |
| Hanning | $3.1 \times 2\pi/M$ | $-31$ dB | 44 dB |
| Hamming | $3.3 \times 2\pi/M$ | $-41$ dB | 53 dB |
| Blackman | $5.5 \times 2\pi/M$ | $-57$ dB | 74 dB |
| Kaiser | Adjustable via $\beta$ | Adjustable | Adjustable |

**Kaiser window**: $w(n) = I_0\!\left(\beta\sqrt{1-(1-2n/M)^2}\right) / I_0(\beta)$, where $I_0$ is the zeroth-order modified Bessel function of the first kind. The parameter $\beta$ controls the trade-off between main-lobe width and sidelobe level.

**Kaiser formula** — given stopband attenuation $A_s = -20\log_{10}\delta_2$ (dB) and normalized transition bandwidth $\Delta\omega = \omega_s - \omega_p$ (rad/sample):

$$\boxed{M \approx \frac{A_s - 7.95}{2.285\,\Delta\omega}} \qquad \text{(filter order)}$$

$$\boxed{\beta = \begin{cases}
0.1102\,(A_s - 8.7) & A_s \gt 50 \\
0.5842\,(A_s - 21)^{0.4} + 0.07886\,(A_s - 21) & 21 \leq A_s \leq 50 \\
0 & A_s \lt 21
\end{cases}}$$

Given $A_s$ and $\Delta\omega$, both $M$ and $\beta$ are fully determined.

**② Frequency-Sampling Method**

Specify the desired response at $N$ DFT frequencies and compute the IDFT to obtain $h(n)$. Transition-band samples can be optimized iteratively to minimize maximum ripple.

**③ Chebyshev (Equiripple) Approximation**

Minimizes the maximum frequency-domain error $\lVert H(e^{j\omega}) - H_d(e^{j\omega})\rVert_\infty$ over specified bands. The **Parks-McClellan algorithm** (Remez exchange) finds the optimal equiripple solution. For equal specifications, equiripple designs require lower filter order than window-based designs.

### 3.3.3 IIR Filter Design: Impulse Invariance, Bilinear Transform, Frequency Transformation

IIR filter design typically proceeds from a classical analog (continuous-time) filter prototype.

**① Impulse Invariance**

Map an analog filter $H_a(s)$ to a digital filter $H(z)$ by matching the impulse response at sample instants: $h(n) = T_s\, h_a(nT_s)$. Poles at $s_k$ in $H_a(s)$ map to poles at $z_k = e^{s_k T_s}$ in $H(z)$.

**Limitation — spectral aliasing**: The frequency response satisfies:

$$H(e^{j\omega}) = \sum_{k=-\infty}^{\infty} H_a\!\left(j\frac{\omega + 2\pi k}{T_s}\right)$$

Replicas from adjacent periods overlap. Suitable only for **lowpass and bandpass** designs where the analog prototype is sufficiently bandlimited.

**② Bilinear z-Transform**

Map $s$-domain to $z$-domain via:

$$s = \frac{2}{T_s}\cdot\frac{1-z^{-1}}{1+z^{-1}} \qquad \Longleftrightarrow \qquad z = \frac{1 + (T_s/2)\,s}{1 - (T_s/2)\,s}$$

Frequency mapping (analog $\Omega$ $\leftrightarrow$ digital $\omega$):

$$\Omega = \frac{2}{T_s}\tan\!\left(\frac{\omega}{2}\right) \qquad \Longleftrightarrow \qquad \omega = 2\arctan\!\left(\frac{\Omega T_s}{2}\right)$$

**Advantage**: No aliasing — the entire analog axis $(-\infty, +\infty)$ is compressed onto $(-\pi, \pi)$.

**Limitation — frequency warping**: The nonlinear $\Omega$-$\omega$ relationship distorts the frequency axis. Critical frequencies must be **pre-warped** before designing the analog prototype:

$$\Omega_c = \frac{2}{T_s}\tan\!\left(\frac{\omega_c}{2}\right) \qquad \text{(pre-warping)}$$

**③ Frequency Transformation Method**

Design a digital lowpass prototype $H_{lp}(z')$, then apply a digital-domain frequency transformation to obtain the target filter type:

| Transformation | Substitution | Purpose |
|---|---|---|
| Lowpass → Lowpass | $z'^{-1} \to \dfrac{z^{-1} - \alpha}{1 - \alpha z^{-1}}$ | Shift cutoff frequency |
| Lowpass → Highpass | $z'^{-1} \to -\dfrac{z^{-1} + \alpha}{1 + \alpha z^{-1}}$ | Spectral inversion |
| Lowpass → Bandpass | $z'^{-1} \to -\dfrac{z^{-2} - a_1 z^{-1} + a_2}{a_2 z^{-2} - a_1 z^{-1} + 1}$ | Frequency-axis expansion |
| Lowpass → Bandstop | $z'^{-1} \to \dfrac{z^{-2} - a_1 z^{-1} + a_2}{a_2 z^{-2} - a_1 z^{-1} + 1}$ | Notch transformation |

The parameter $\alpha$ (and $a_1, a_2$ for bandpass/bandstop) is determined by the desired cutoff frequencies.

---

# Part IV: Special Sequences and Corresponding Filters

> 📖 Textbook §2.4 (Minimum-Phase and System Invertibility, §2.4.1–§2.4.4); §2.5 (Lattice Filter Realizations)

---

## 4.1 Allpass Sequences and Allpass Filters

### 4.1.1 Definition and Properties

An **allpass filter** has unit magnitude response at all frequencies:

$$\lvert H_{ap}(e^{j\omega})\rvert = 1 \quad \forall\, \omega$$

This implies:

$$H_{ap}(e^{j\omega})\, H_{ap}^{*}(e^{j\omega}) = 1, \qquad h_{ap}(n) \ast h_{ap}^{*}(-n) = \delta(n)$$

$$\boxed{H_{ap}(z)\, H_{ap}^{*}(1/z^{*}) = 1}$$

**Pole-zero structure**: For a stable rational allpass filter, every pole at $z = c_k$ (inside the unit circle) is paired with a zero at $z = 1/c_k^*$ (outside the unit circle) — the **conjugate reciprocal** location.

**First-order allpass** ($\lvert\alpha\rvert \lt 1$, $\alpha \in \mathbb{C}$):

$$H_{ap1}(z) = \frac{z^{-1} - \alpha^*}{1 - \alpha\, z^{-1}}$$

**General $N$-th order allpass**:

$$H_{ap}(z) = \prod_{k=1}^{N} \frac{z^{-1} - \alpha_k^*}{1 - \alpha_k\, z^{-1}} = \frac{z^{-N} + a_1^* z^{-N+1} + \cdots + a_N^*}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}$$

The numerator polynomial is the **conjugate-reversed** polynomial of the denominator.

> ![Figure 4.1](<./CourseADSP2026/Fig/fig_4_1.png>)
>
> *Figure 4.1: Typical pole-zero patterns of a PZ (pole-zero) system, all-pass system: (a) complex-valued coefficients and (b) real-valued coefficients.*

**Use cases**: Phase equalization (correcting phase distortion without altering magnitude), building blocks in lattice filters and filter banks, Schur-Cohn stability testing.

### 4.1.2 Group Delay of Allpass Filters — Proof That Group Delay is Always Positive

The **group delay** of a filter is:

$$\tau(\omega) = -\frac{d}{d\omega}\angle H(e^{j\omega})$$

For the first-order real allpass filter

$$H_{ap}(z) = \frac{z^{-1} - \alpha}{1 - \alpha z^{-1}}$$

where $\lvert\alpha\rvert \lt 1$ and $\alpha \in \mathbb{R}$, the phase response is:

$$\angle H_{ap}(e^{j\omega}) = \omega - 2\arctan\!\left(\frac{\alpha \sin\omega}{1 - \alpha\cos\omega}\right) - \pi$$

Differentiating and negating:

$$\tau(\omega) = -\frac{d}{d\omega}\angle H_{ap}(e^{j\omega}) = \frac{1 - \alpha^2}{1 - 2\alpha\cos\omega + \alpha^2}$$

$$\boxed{\tau(\omega) = \frac{1 - \lvert\alpha\rvert^2}{\lvert 1 - \alpha e^{-j\omega}\rvert^2} \gt 0 \quad \forall\,\omega}$$

**Proof that $\tau(\omega) \gt 0$**:
- Numerator: $1 - \alpha^2 \gt 0$ since $\lvert\alpha\rvert \lt 1$
- Denominator: $\lvert 1 - \alpha e^{-j\omega}\rvert^2 \gt 0$ since the pole $\alpha$ is not on the unit circle

**General $N$-th order result**: The group delay of any stable allpass filter is always positive:

$$\tau(\omega) = \sum_{k=1}^{N} \frac{1 - \lvert\alpha_k\rvert^2}{\lvert 1 - \alpha_k e^{-j\omega}\rvert^2} \gt 0 \quad \forall\,\omega$$

This is a sum of positive terms. An allpass filter is a pure **phase-lag device**: it introduces causal group delay at every frequency without altering the magnitude response. This makes allpass filters ideal for **phase equalization**.

---

## 4.2 Minimum-Phase Sequences and Minimum-Phase Filters

### 4.2.1 Definition

A causal, stable filter $H_m(z)$ is **minimum-phase** if and only if:
1. **Stable**: ROC includes the unit circle (all poles inside)
2. **Causal**: $h_m(n) = 0$ for $n \lt 0$
3. **All zeros inside or on the unit circle**

Among all causal, stable filters with the same magnitude response, the minimum-phase filter has:
- The smallest phase lag at every frequency
- The fastest energy buildup: $\sum_{n=0}^{k}\lvert h_m(n)\rvert^2 \geq \sum_{n=0}^{k}\lvert h(n)\rvert^2$ for all $k$
- A **causal, stable inverse** $1/H_m(z)$ (all zeros inside the unit circle → all inverse poles inside)

> ![Figure 4.2](<./CourseADSP2026/Fig/fig_4_2.png>)
>
> *Figure 4.2: Pole-zero, magnitude, phase, and group-delay plots for minimum-phase, maximum-phase, and two mixed-phase systems sharing the same magnitude response — the minimum-phase system has the smallest phase lag (group delay) at every frequency.*

> ![Figure 4.3](<./CourseADSP2026/Fig/fig_4_3.png>)
>
> *Figure 4.3: Impulse responses (top) and energy-delay curves $\sum_{n=0}^{k}\lvert h(n)\rvert^2$ (bottom) for the same four systems — the minimum-phase system achieves the fastest energy buildup at every $k$.*

### 4.2.2 Theorem 1.1: Minimum-Phase / Allpass Decomposition

> **Theorem 1.1**: Any causal, stable system $H(z)$ can be **uniquely** decomposed as:
>
> $$\boxed{H(z) = H_{ap}(z) \cdot H_m(z)}$$
>
> where $H_m(z)$ is minimum-phase and $H_{ap}(z)$ is allpass.

**Proof by construction**:

Given $H(z)$ with poles $\lbrace p_k\rbrace$ (all inside the unit circle) and zeros $\lbrace z_k\rbrace$ (some possibly outside):

**Step 1** — Partition zeros:
- $\mathcal{Z}_{in} = \lbrace z_k : \lvert z_k\rvert \leq 1\rbrace$: zeros inside or on the unit circle
- $\mathcal{Z}_{out} = \lbrace z_k : \lvert z_k\rvert \gt 1\rbrace$: zeros outside the unit circle

**Step 2** — Construct $H_m(z)$:
- Assign all poles $\lbrace p_k\rbrace$ to $H_m(z)$
- Assign all zeros in $\mathcal{Z}_{in}$ to $H_m(z)$
- For each zero $c \in \mathcal{Z}_{out}$: add a zero at $1/c^*$ (its conjugate reciprocal, inside the unit circle) to $H_m(z)$

The resulting $H_m(z)$ is causal, stable, and all-zeros-inside — minimum-phase.

**Step 3** — Construct $H_{ap}(z) = H(z)/H_m(z)$:
- For each $c \in \mathcal{Z}_{out}$: $H_{ap}(z)$ inherits a zero at $c$ (outside) and a pole at $1/c^*$ (inside) — exactly one allpass factor $\dfrac{z^{-1} - c^*}{1 - c\, z^{-1}}$

**Verification**: $\lvert H(e^{j\omega})\rvert = \lvert H_{ap}(e^{j\omega})\rvert \cdot \lvert H_m(e^{j\omega})\rvert = 1 \cdot \lvert H_m(e^{j\omega})\rvert$. The phase of $H$ exceeds that of $H_m$ by the allpass phase lag (always positive group delay). ✓

> ![Figure 4.4](<./CourseADSP2026/Fig/fig_4_4.png>)
>
> *Figure 4.4: Minimum phase and all-pass decomposition.*

---

## 4.3 Linear-Phase Sequences and Linear-Phase Filters

### 4.3.1 Strict and Generalized Linear Phase

A filter has **generalized linear phase** if:

$$H(e^{j\omega}) = e^{j\beta}\, e^{-j\alpha\omega}\, A(e^{j\omega})$$

where $A(e^{j\omega})$ is **real-valued** (amplitude function), $\alpha$ is the **constant group delay** (samples), and $\beta \in \lbrace 0, \pm\pi/2\rbrace$.

The z-transform satisfies (for real $h(n)$):

$$H(z) = \pm z^{-(N-1)} H(1/z)$$

If $z_0$ is a zero of $H(z)$, then $1/z_0$ is also a zero — zeros off the unit circle come in reciprocal pairs.

**Strict linear phase** ($\beta = 0$, real $h(n)$): requires **even symmetry**:

$$h(n) = h(N-1-n), \quad n = 0, 1, \ldots, N-1$$

Group delay is constant: $\tau(\omega) = \alpha = (N-1)/2$ samples.

### 4.3.2 Four Types of Linear-Phase FIR Filters

Classified by symmetry type (even or odd) and filter length (odd or even):

| Type | Symmetry of $h(n)$ | Length $N$ | Group delay $\alpha$ | Phase $\beta$ |
|------|-------------------|-----------|---------------------|--------------|
| **I** | Even: $h(n)=h(N{-}1{-}n)$ | Odd | $(N-1)/2$ (integer) | $0$ |
| **II** | Even: $h(n)=h(N{-}1{-}n)$ | Even | $(N-1)/2$ (half-integer) | $0$ |
| **III** | Odd: $h(n)=-h(N{-}1{-}n)$ | Odd | $(N-1)/2$ (integer) | $\pi/2$ |
| **IV** | Odd: $h(n)=-h(N{-}1{-}n)$ | Even | $(N-1)/2$ (half-integer) | $\pi/2$ |

**Amplitude functions** $A(e^{j\omega})$ (all real-valued):

| Type | Amplitude function $A(e^{j\omega})$ |
|------|-----------------------------------|
| I | $\displaystyle a(0) + 2\sum_{n=1}^{(N-1)/2} a(n)\cos(n\omega)$, where $a(n) = h\!\left(\tfrac{N-1}{2} - n\right)$ |
| II | $\displaystyle 2\sum_{n=1}^{N/2} b(n)\cos\!\left[\!\left(n-\tfrac{1}{2}\right)\omega\right]$, where $b(n) = h\!\left(\tfrac{N}{2} - n\right)$ |
| III | $\displaystyle 2\sum_{n=1}^{(N-1)/2} c(n)\sin(n\omega)$, where $c(n) = h\!\left(\tfrac{N-1}{2} - n\right)$ |
| IV | $\displaystyle 2\sum_{n=1}^{N/2} d(n)\sin\!\left[\!\left(n-\tfrac{1}{2}\right)\omega\right]$, where $d(n) = h\!\left(\tfrac{N}{2} - n\right)$ |

### 4.3.3 Applicable Filter Types per Class

| Type | Lowpass | Highpass | Bandpass | Bandstop | Hilbert / Differentiator |
|------|:-------:|:--------:|:--------:|:--------:|:------------------------:|
| **I** | ✓ | ✓ | ✓ | ✓ | — |
| **II** | ✓ | **✗** | ✓ | **✗** | — |
| **III** | **✗** | **✗** | ✓ | **✗** | ✓ |
| **IV** | **✗** | ✓ | ✓ | **✗** | ✓ |

**Constraints and their causes**:

- **Type II** — forced zero at $\omega = \pi$ ($z = -1$):

  Even symmetry + even $N$ implies odd $N-1$. From $H(z) = z^{-(N-1)}H(1/z)$, evaluating at $z = -1$:
  $H(-1) = (-1)^{-(N-1)}H(-1) = -H(-1) \Rightarrow H(-1) = 0$.
  Cannot implement **highpass or bandstop** filters.

- **Type III** — forced zeros at both $\omega = 0$ and $\omega = \pi$:

  Odd symmetry forces $h\!\left(\frac{N-1}{2}\right) = 0$ (center tap is zero). Evaluating $H(z)$ at $z = \pm 1$ gives $H(\pm 1) = 0$. Useful only for **bandpass**, differentiators, and Hilbert transformers.

- **Type IV** — forced zero at $\omega = 0$:

  Odd symmetry + even $N$: evaluating at $z = 1$ gives $H(1) = -H(1) \Rightarrow H(1) = 0$.
  Cannot implement **lowpass or bandstop** filters; suitable for highpass and Hilbert transformers.

- **Type I** — most general: no forced zeros at $\omega = 0$ or $\omega = \pi$; suitable for all filter types.

---

## 4.4 Positive Semi-Definite Sequences

### 4.4.1 Autocorrelation Sequences and Power Spectral Non-Negativity

A sequence $r(n)$ is **Hermitian (conjugate-symmetric)** if:

$$r(n) = r^*(-n)$$

Its DTFT is **real-valued**: $R(e^{j\omega}) \in \mathbb{R}$, and its z-transform satisfies $R(z) = R^*(1/z^*)$.

A Hermitian sequence is **positive semi-definite** if additionally:

$$R(e^{j\omega}) \geq 0 \quad \forall\, \omega$$

**Connection to autocorrelation**: The autocorrelation of a signal $x(n)$ is:

$$r_x(n) = \sum_{k=-\infty}^{\infty} x(k)\, x^{*}(k-n) = x(n) \ast x^{*}(-n)$$

Its DTFT is the **power spectral density**:

$$P_x(e^{j\omega}) = \lvert X(e^{j\omega})\rvert^2 \geq 0$$

Autocorrelation sequences are therefore **always positive semi-definite**. The autocorrelation matrix is:

$$\mathbf{R} = [r_x(i-j)]_{i,j}$$

It is a **Hermitian Toeplitz positive semi-definite matrix** — the central object in Wiener filtering (Chapter 6) and linear prediction (Chapter 3).

### 4.4.2 Theorem 1.2: Zero Pairing in Rational Positive Semi-Definite Sequences

> **Theorem 1.2**: Let $R(z)$ be the rational z-transform of a positive semi-definite sequence with real-valued coefficients. Then:
> 1. **Zeros on the unit circle** ($\lvert z_0\rvert = 1$): occur in **conjugate pairs** $(z_0,\, z_0^*)$, each with **even multiplicity**
> 2. **Zeros off the unit circle**: occur in **quadruples** $\lbrace z_0,\; z_0^*,\; 1/z_0,\; 1/z_0^*\rbrace$

**Explanation**:
- Hermitian symmetry ($R(z) = R^*(1/z^*)$, real coefficients): if $z_0$ is a zero, then so are $z_0^*$, $1/z_0$, and $1/z_0^*$
- Non-negativity $R(e^{j\omega}) \geq 0$: zeros on the unit circle must have even multiplicity, otherwise $R(e^{j\omega})$ would change sign

**Spectral Factorization**: Any rational positive semi-definite sequence factors uniquely as:

$$\boxed{R(z) = \sigma^2\, H_m(z)\, H_m^*(1/z^*)}$$

On the unit circle: $R(e^{j\omega}) = \sigma^2\lvert H_m(e^{j\omega})\rvert^2$, where $H_m(z)$ is **minimum-phase** (all zeros inside the unit circle). The quadruple structure $\lbrace z_0, z_0^*, 1/z_0, 1/z_0^*\rbrace$ splits as $\lbrace 1/z_0, 1/z_0^*\rbrace \to H_m(z)$ and $\lbrace z_0, z_0^*\rbrace \to H_m^*(1/z^*)$.

This spectral factorization theorem is **fundamental** to:
- **Wiener filter theory** (Chapter 6): whitening the input requires the minimum-phase spectral factor
- **Linear prediction** (Chapter 3): the prediction error filter is the inverse of the spectral factor
- **Power spectral estimation** (Chapter 5): parametric spectral models exploit this structure

---

## Chapter 1 Summary

| Concept | Key Property | Application |
|---------|-------------|-------------|
| DTFT | Spectrum on unit circle; $2\pi$-periodic | Frequency analysis |
| z-Transform | Generalizes DTFT to complex plane | System analysis, filter design |
| z–s mapping: $z=e^{sT_s}$ | Left half-plane $\leftrightarrow$ interior of unit circle | Analog-to-digital filter conversion |
| DFT | Sampled DTFT; finite-length | Numerical computation, FFT algorithms |
| FFT (Radix-2) | $\frac{N}{2}\log_2 N$ multiplications; $\approx 200\times$ speedup at $N=1024$ | Fast convolution, spectral analysis |
| FIR direct form | Always stable; $M+1$ taps | Baseline implementation |
| Comb + accumulator | Recursive FIR: 2 ops/sample | Efficient moving-average |
| IIR Direct Form II | $N$ delays (canonical) | Efficient IIR realization |
| IIR Parallel Form | Independent poles; best numerical stability | High-order stable filters |
| Bilinear transform | Alias-free; pre-warp critical frequencies | IIR design from analog prototype |
| Allpass filter | $\lVert H_{ap}\rVert=1$; group delay $\tau(\omega) \gt 0$ always | Phase equalization |
| Min-phase decomposition | $H(z) = H_{ap}(z)\,H_m(z)$, unique | Causal inversion, equalization |
| Linear-phase FIR | $h(n) = \pm h(N-1-n)$; four types | Distortion-free filtering |
| Type I | Even symmetry, odd $N$; no forced zeros | All filter types |
| Type II | Even symmetry, even $N$; zero at $\omega=\pi$ | Lowpass and bandpass only |
| Positive semi-definite | $R(e^{j\omega}) \geq 0$; zeros in quadruples off unit circle | Autocorrelation structure |
| Spectral factorization | $R(z) = \sigma^2 H_m(z)H_m^*(1/z^*)$ | Wiener filter, linear prediction |

### Looking Ahead

This chapter reviewed **deterministic** signal and system theory. Starting in Chapter 2, the focus shifts to **random (stochastic) signals** — characterized by statistical properties (mean, autocorrelation, power spectrum) rather than exact waveforms. The spectral factorization of $P_x(e^{j\omega})$ (Theorem 1.2) and the minimum-phase/allpass decomposition (Theorem 1.1) will recur as key tools throughout the course.

---

*End of Chapter 1*
