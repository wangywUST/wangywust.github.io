# Modern Digital Signal Processing
## Chapter 1: Discrete-Time Signal Processing — Undergraduate DSP Review

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005
> Chapters covered: Ch. 1 (Introduction) + Ch. 2 (Fundamentals of Discrete-Time Signal Processing)

## Table of Contents

1. [Part I: Digital Signals and DSP Overview](#part-i-digital-signals-and-dsp-overview)
2. [Part II: Transforms for Discrete-Time Signals](#part-ii-transforms-for-discrete-time-signals)
3. [Part III: Digital Filter Structures and Design](#part-iii-digital-filter-structures-and-design)
4. [Part IV: Special Sequences and Corresponding Filters](#part-iv-special-sequences-and-corresponding-filters)

---

# Part I: Digital Signals and DSP Overview

> 📖 Textbook §1.1 (Random Signals overview); §2.1 (Discrete-Time Signals)

---

## 1.1 Basic Types and Examples of Digital Signals

A **digital signal** is a discrete-time, discrete-amplitude representation of a physical quantity. It arises from sampling and quantizing a continuous-time analog signal.

### 1.1.1 Speech Signals

A speech signal is a **one-dimensional time-domain discrete sequence**. A continuous pressure wave $x_a(t)$ is sampled at rate $f_s$ (typically 8 kHz for telephony, 44.1 kHz for audio) to produce $x(n) = x_a(nT_s)$ where $T_s = 1/f_s$.

> ![Figure 1.1 speech waveform](<./CourseADSP2026/Fig/fig_1_1.pdf>)
> *Figure 1.1: A speech signal sampled at a fixed rate, shown as a discrete sequence in the time domain. Voiced segments (vowels) exhibit quasi-periodic structure; unvoiced segments (fricatives) appear noise-like.*

### 1.1.2 Image Signals

An image signal is a **two-dimensional discrete sequence** $x(m, n)$, where $(m, n)$ indexes the row and column of a pixel. Grayscale images have scalar values; color images have vector values (e.g., RGB). Standard digital image resolutions range from $320 \times 240$ (VGA) to $3840 \times 2160$ (4K).

$$x(m, n), \quad m = 0, 1, \ldots, M-1;\; n = 0, 1, \ldots, N-1$$

> **[PLACEHOLDER: 2D discrete image grid — grayscale pixel array with coordinate axes $(m,n)$]**
> *Figure 1.2: An image signal — 2D pixel array. Each element $x(m,n)$ stores intensity (grayscale) or a color vector.*

### 1.1.3 Video Signals

A video signal is a **dynamic image sequence**: a series of image frames $x_k(m,n)$ indexed by frame number $k$, captured at a frame rate (typically 25, 30, or 60 fps). Video introduces temporal correlation in addition to spatial correlation, requiring 3D signal processing.

$$\text{Video:} \quad \{x_k(m,n)\}_{k=0}^{K-1}, \quad k = \text{frame index}$$

The large data volume of uncompressed video (e.g., 1080p at 30 fps $\approx$ 1.5 Gbps) motivates video compression standards such as H.264, H.265/HEVC, and AV1.

### 1.1.4 Communication Signals

Modern digital communication uses discrete baseband sequences. Key examples:

**OFDM baseband sequence**: An Orthogonal Frequency Division Multiplexing system maps bits onto $N$ subcarriers via modulation (e.g., QAM), applies an $N$-point IFFT to produce the time-domain discrete sequence $x(n)$, and appends a cyclic prefix (CP) to combat multipath. The received signal after FFT demodulation is:

$$Y(k) = H(k) X(k) + W(k), \quad k = 0, 1, \ldots, N-1$$

**Constellation diagrams**: The modulation alphabet is visualized as a constellation. Common examples:
- **QPSK** ($M=4$): 4 points on a circle, 2 bits/symbol
- **8-PSK** ($M=8$): 8 points on a circle, 3 bits/symbol
- **16-QAM** ($M=16$): 16 points on a square grid, 4 bits/symbol
- **64-QAM** ($M=64$): 64 points, 6 bits/symbol

> **[PLACEHOLDER: Constellation diagrams for QPSK, 8-PSK, 16-QAM, 64-QAM]**
> *Figure 1.3: Baseband communication signals — constellation maps. Each point represents one symbol carrying $\log_2 M$ bits. Higher-order constellations achieve greater spectral efficiency but require higher SNR.*

### 1.1.5 Radar Signals

Radar signals are specially designed waveforms transmitted and received after reflection from a target. Two common types:

**Linear frequency modulation (LFM / chirp)**: The instantaneous frequency increases linearly with time:

$$x(t) = \text{rect}\!\left(\frac{t}{T}\right) e^{j\pi \mu t^2}, \quad \mu = B/T \text{ (chirp rate)}$$

After sampling: $x(n) = e^{j\pi\mu(nT_s)^2}$. LFM enables pulse compression: a long pulse (high energy) is compressed to a short pulse (high resolution) via matched filtering.

**Frequency-diversity radar**: Successive pulses use different carrier frequencies to improve target discrimination and reduce scintillation. The transmitted waveform exhibits a stepped-frequency pattern.

> **[PLACEHOLDER: LFM radar waveform — time-domain showing increasing frequency, frequency-domain showing constant amplitude spectrum]**
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

> **[PLACEHOLDER: OFDM transceiver block diagram]**
> *Figure 1.5: OFDM transceiver — Transmitter: data bits → channel coding → interleaving → QAM modulation → S/P → IFFT → P/S → CP insertion → DAC → RF TX. Receiver is the symmetric reverse chain.*

---

## 1.3 General Digital Signal Processing System

### 1.3.1 System Pipeline

A continuous-time bandlimited signal $x_a(t)$ (bandwidth $B$ Hz) is processed as follows:

$$x_a(t) \xrightarrow{\text{AAF}} \xrightarrow{\text{A/D},\; f_s} x(n) \xrightarrow{h(n)} y(n) \xrightarrow{\text{D/A}} y_a(t)$$

1. **Anti-aliasing filter (AAF)**: Lowpass filter with cutoff $f_s/2$ — removes frequency components above $f_s/2$ to prevent aliasing.
2. **A/D conversion**: Samples $x_a(t)$ at rate $f_s \geq 2B$ and quantizes to $b$ bits.
3. **Discrete-time processing**: Convolution with $h(n)$, or more generally any digital signal processing algorithm.
4. **D/A conversion and reconstruction filter**: Converts the discrete output back to a continuous signal.

> **[PLACEHOLDER: DSP pipeline block diagram with time-domain and frequency-domain waveforms at each stage]**
> *Figure 1.6: General DSP system — pipeline from continuous input to continuous output, with frequency-domain illustrations at each stage.*

### 1.3.2 Sampling Theorem and Bandlimited Condition

**Nyquist–Shannon Sampling Theorem**: A continuous-time signal $x_a(t)$ with bandwidth $B$ Hz (i.e., $X_a(f) = 0$ for $\lvert f\rvert > B$) can be perfectly reconstructed from its samples $x(n) = x_a(nT_s)$ if and only if:

$$f_s = \frac{1}{T_s} \geq 2B$$

The minimum sampling rate $f_s = 2B$ is the **Nyquist rate**. The relationship between analog frequency $f$ and digital frequency $\omega$ is:

$$\omega = 2\pi f T_s = \frac{2\pi f}{f_s}$$

Digital frequency $\omega \in [-\pi, \pi]$ corresponds to analog frequency $f \in [-f_s/2,\; f_s/2]$.

**If $f_s < 2B$ (undersampling)**: Spectral replicas overlap → **aliasing** — high-frequency components masquerade as low-frequency ones, irreversibly corrupting the signal.

### 1.3.3 The Two Fundamental Problems of DSP

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

The DTFT exists when the ROC of $X(z)$ includes the unit circle.

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
| Conjugate | $x^*(n)$ | $\leftrightarrow$ | $X^*(e^{-j\omega})$ |
| Time reversal | $x(-n)$ | $\leftrightarrow$ | $X(e^{-j\omega})$ |
| Convolution | $x(n) * y(n)$ | $\leftrightarrow$ | $X(e^{j\omega})Y(e^{j\omega})$ |
| Correlation | $\sum_k x(k)y^*(k-n)$ | $\leftrightarrow$ | $X(e^{j\omega})Y^*(e^{j\omega})$ |
| Multiplication | $x(n)y(n)$ | $\leftrightarrow$ | $\frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\theta})Y(e^{j(\omega-\theta)})d\theta$ |
| Frequency differentiation | $n\cdot x(n)$ | $\leftrightarrow$ | $j\dfrac{d}{d\omega}X(e^{j\omega})$ |
| Parseval's theorem | $\sum_{n} x(n)y^*(n)$ | $=$ | $\frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega})Y^*(e^{j\omega})d\omega$ |

**Energy form of Parseval's theorem** (setting $y = x$):

$$\sum_{n=-\infty}^{\infty} \lvert x(n)\rvert^2 = \frac{1}{2\pi}\int_{-\pi}^{\pi} \lvert X(e^{j\omega})\rvert^2\, d\omega$$

Total energy is preserved between the time and frequency domains.

---

## 2.2 z-Transform

### 2.2.1 Definition and Region of Convergence

The **z-transform** generalizes the DTFT to the complex plane:

$$\boxed{X(z) = \sum_{n=-\infty}^{\infty} x(n)\, z^{-n}, \quad z \in \mathbb{C}}$$

The **Region of Convergence (ROC)** is the set of $z$ values for which the sum converges absolutely:

$$\text{ROC} = \left\{ z \in \mathbb{C} : \sum_{n=-\infty}^{\infty} \lvert x(n)\rvert\, \lvert z\rvert^{-n} < \infty \right\}$$

The ROC takes the form of an annulus $r_1 < \lvert z\rvert < r_2$ (two-sided sequences), $\lvert z\rvert > r_1$ (right-sided/causal), or $\lvert z\rvert < r_2$ (left-sided).

**Inverse z-transform** via contour integration:

$$x(n) = \frac{1}{2\pi j} \oint_C X(z) z^{n-1}\, dz$$

where $C$ is a counterclockwise contour within the ROC. In practice, inverse z-transforms are computed via **partial fraction expansion**.

### 2.2.2 Main Properties of the z-Transform

| Property | Time Sequence | $\leftrightarrow$ | z-Transform | ROC |
|----------|--------------|---|-------------|-----|
| Linearity | $ax(n)+by(n)$ | $\leftrightarrow$ | $aX(z)+bY(z)$ | At least $\text{ROC}_x \cap \text{ROC}_y$ |
| Time shift | $x(n-K)$ | $\leftrightarrow$ | $z^{-K}X(z)$ | $\text{ROC}_x$ (modified at $z=0,\infty$) |
| z-domain scaling | $\alpha^n x(n)$ | $\leftrightarrow$ | $X(z/\alpha)$ | $\lvert\alpha\rvert\,r_1 < \lvert z\rvert < \lvert\alpha\rvert\,r_2$ |
| Conjugate | $x^*(n)$ | $\leftrightarrow$ | $X^*(z^*)$ | $\text{ROC}_x$ |
| Time reversal | $x(-n)$ | $\leftrightarrow$ | $X(1/z)$ | $1/r_2 < \lvert z\rvert < 1/r_1$ |
| Convolution | $x(n)*y(n)$ | $\leftrightarrow$ | $X(z)Y(z)$ | At least $\text{ROC}_x \cap \text{ROC}_y$ |
| Correlation | $r_{xy}(n) = x(n)*y(-n)$ | $\leftrightarrow$ | $X(z)Y(z^{-1})$ | — |
| z-domain differentiation | $n\cdot x(n)$ | $\leftrightarrow$ | $-z\dfrac{d}{dz}X(z)$ | $\text{ROC}_x$ |
| Initial value | $x(0)$, causal $x$ | — | $\lim_{z\to\infty} X(z)$ | — |
| Parseval | $\sum_n x(n)y^*(n)$ | $=$ | $\frac{1}{2\pi j}\oint X(v)Y^*(1/v^*)v^{-1}dv$ | — |

### 2.2.3 Common z-Transform Pairs

| Sequence $x(n)$ | z-Transform $X(z)$ | ROC |
|-----------------|-------------------|-----|
| $\delta(n)$ | $1$ | All $z$ |
| $u(n)$ (unit step) | $\dfrac{1}{1-z^{-1}}$ | $\lvert z\rvert > 1$ |
| $\alpha^n u(n)$ | $\dfrac{1}{1-\alpha z^{-1}}$ | $\lvert z\rvert > \lvert\alpha\rvert$ |
| $-\alpha^n u(-n-1)$ | $\dfrac{1}{1-\alpha z^{-1}}$ | $\lvert z\rvert < \lvert\alpha\rvert$ |
| $n\alpha^n u(n)$ | $\dfrac{\alpha z^{-1}}{(1-\alpha z^{-1})^2}$ | $\lvert z\rvert > \lvert\alpha\rvert$ |
| $r^n \cos(\omega_0 n) u(n)$ | $\dfrac{1-r\cos\omega_0\, z^{-1}}{1-2r\cos\omega_0\, z^{-1}+r^2z^{-2}}$ | $\lvert z\rvert > r$ |
| $r^n \sin(\omega_0 n) u(n)$ | $\dfrac{r\sin\omega_0\, z^{-1}}{1-2r\cos\omega_0\, z^{-1}+r^2z^{-2}}$ | $\lvert z\rvert > r$ |
| $\alpha^n u(n) - \alpha^n u(n-N)$ | $\dfrac{1-\alpha^N z^{-N}}{1-\alpha z^{-1}}$ | $\lvert z\rvert > 0$ |
| $\alpha^{\lvert n\rvert}$ | $\dfrac{1-\alpha^2}{(1-\alpha z^{-1})(1-\alpha z)}$ | $\lvert\alpha\rvert < \lvert z\rvert < 1/\lvert\alpha\rvert$ |

### 2.2.4 Relationship Between the z-Transform and the Laplace Transform

The z-transform is the **discrete-time counterpart** of the Laplace transform. The correspondence is established through the sampling relationship $z = e^{sT_s}$:

$$s = \sigma + j\Omega \xrightarrow{z = e^{sT_s}} z = e^{\sigma T_s}\, e^{j\Omega T_s} = r\, e^{j\omega}$$

where $r = e^{\sigma T_s}$ and $\omega = \Omega T_s = \Omega / f_s$.

**Key correspondences**:

| Laplace / s-domain | z-Transform / z-domain |
|-------------------|----------------------|
| Left half-plane ($\sigma < 0$) | Interior of unit circle ($\lvert z\rvert < 1$) |
| Right half-plane ($\sigma > 0$) | Exterior of unit circle ($\lvert z\rvert > 1$) |
| Imaginary axis ($\sigma = 0$, $s = j\Omega$) | Unit circle ($\lvert z\rvert = 1$) |
| Stable causal: poles in left half-plane | Stable causal: poles inside unit circle |
| Analog frequency $\Omega$ (rad/s) | Digital frequency $\omega = \Omega T_s$ (rad/sample) |

**Frequency mapping**: As $\Omega$ traverses $[-\pi/T_s,\; \pi/T_s]$ (i.e., $[-f_s/2,\; f_s/2]$), the digital frequency $\omega = \Omega T_s$ completes one full traversal of $[-\pi, \pi]$ — one loop around the unit circle. This mapping is the theoretical basis of the **bilinear z-transform** and **impulse invariance** methods for IIR filter design.

> **[PLACEHOLDER: Side-by-side s-plane and z-plane showing the mapping $z = e^{sT_s}$: left half-plane maps to interior of unit circle; imaginary axis maps to unit circle; horizontal strips of width $2\pi/T_s$ map periodically onto the full z-plane]**
> *Figure 2.1: s-plane to z-plane mapping via $z = e^{sT_s}$.*

---

## 2.3 Discrete Fourier Transform (DFT)

### 2.3.1 From DTFT to DFT: Uniform Frequency Sampling

The DTFT $X(e^{j\omega})$ is continuous in $\omega \in [-\pi, \pi]$. For numerical computation, we sample it at $N$ equally spaced frequencies:

$$\omega_k = \frac{2\pi k}{N}, \quad k = 0, 1, \ldots, N-1$$

This yields the **DFT**:

$$\boxed{X(k) = \sum_{n=0}^{N-1} x(n)\, W_N^{nk}} \qquad k = 0, 1, \ldots, N-1$$

$$\boxed{x(n) = \frac{1}{N}\sum_{k=0}^{N-1} X(k)\, W_N^{-nk}} \qquad n = 0, 1, \ldots, N-1$$

where $W_N = e^{-j2\pi/N}$ is the **twiddle factor**.

The DFT can be written in matrix form $\mathbf{X} = \mathbf{W}_N \mathbf{x}$, where $[\mathbf{W}_N]_{kn} = W_N^{kn}$ is the $N\times N$ DFT matrix. The inverse uses $\mathbf{W}_N^{-1} = \frac{1}{N}\mathbf{W}_N^*$.

### 2.3.2 Relationship Between DFT and DFS

The DFT of a finite-length sequence $x(n)$, $0 \leq n \leq N-1$, equals one period of the **Discrete Fourier Series (DFS)** of its periodic extension $\tilde{x}(n) = x(\langle n \rangle_N)$:

$$\tilde{X}(k) = \sum_{n=0}^{N-1} \tilde{x}(n)\, W_N^{nk} = X(k)$$

**Key implication**: circular convolution in time $\leftrightarrow$ pointwise multiplication in the DFT domain. To compute linear convolution of sequences of lengths $L_1$ and $L_2$ using the DFT, the transform length must satisfy $N \geq L_1 + L_2 - 1$ (with zero-padding) to avoid time-domain aliasing.

### 2.3.3 Main Properties and Uses of the DFT

| Property | Condition | Result |
|----------|-----------|--------|
| Linearity | — | $\text{DFT}\{ax+by\} = aX+bY$ |
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

> **[PLACEHOLDER: 8-point DIT-FFT butterfly flow graph — 3 stages with twiddle factors $W_8^0, W_8^1, W_8^2, W_8^3$; bit-reversed input, natural output]**
> *Figure 2.2: 8-point DIT-FFT signal flow graph.*

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

In DIT-FFT, the input sequence must be reordered in **bit-reversed** order before the butterfly stages. The bit-reversed index is obtained by reversing the $m$-bit binary representation of $n$:

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

The permutation can be performed in-place with an $O(N)$ algorithm: compare each index with its bit-reverse and swap if $n < \text{br}(n)$.

**FFT variants**:
- **Radix-4 FFT**: Groups of 4; further reduces multiplicative count
- **Split-radix FFT**: Best known operation count for power-of-2 $N$
- **Mixed-radix FFT**: Handles arbitrary $N$ by factoring $N = N_1 \times N_2 \times \cdots$ (used in FFTW)

---

# Part III: Digital Filter Structures and Design

> 📖 Textbook §2.3 (Discrete-Time Systems); §2.4 (Minimum-Phase and System Invertibility); §2.5 (Lattice Filter Realizations)

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

> **[PLACEHOLDER: Transversal FIR signal flow graph — delay chain $z^{-1}$, multipliers $h(0),\ldots,h(M)$, adder tree]**
> *Figure 3.1: Direct-form (transversal) FIR filter.*

**Symmetric exploitation**: For linear-phase FIR with $h(n) = h(M-n)$, paired coefficients are equal, nearly halving the number of distinct multiplications.

### 3.1.2 Cascade Form

Factor $H(z)$ into real-coefficient second-order sections (biquads):

$$H(z) = h(0)\prod_{k=1}^{\lfloor M/2 \rfloor} H_k(z), \qquad H_k(z) = 1 + b_{1k}z^{-1} + b_{2k}z^{-2}$$

Complex conjugate zero pairs are combined into each real biquad. Each biquad is a 3-tap transversal FIR.

**Advantages**: Numerically more robust than the direct form for high-order filters (coefficient sensitivity is localized to each section); individual zeros are easily modified.

> **[PLACEHOLDER: Cascade FIR structure — series of second-order biquad sections]**
> *Figure 3.2: Cascade FIR implementation.*

### 3.1.3 Recursive Realization and Comb Filter

An FIR filter can sometimes be implemented recursively via pole-zero cancellation. The classic example is the **uniform moving-average filter**:

$$h(n) = \begin{cases} \dfrac{1}{N} & 0 \leq n \leq N-1 \\ 0 & \text{otherwise} \end{cases}$$

Its transfer function factors as:

$$H(z) = \underbrace{\frac{1-z^{-N}}{N}}_{\displaystyle H_1(z):\ \text{comb filter}} \cdot \underbrace{\frac{1}{1-z^{-1}}}_{\displaystyle H_2(z):\ \text{accumulator}}$$

- $H_1(z)$: **comb filter** — $N$ zeros equally spaced at the $N$-th roots of unity
- $H_2(z)$: **first-order IIR accumulator** — pole at $z=1$ cancels the corresponding comb zero

The pole-zero cancellation preserves the FIR character, while the recursive structure reduces computation from $N$ operations to **only 2 per sample**, regardless of $N$.

> **[PLACEHOLDER: Signal flow graph of recursive FIR: input → comb $H_1(z)$ → accumulator $H_2(z)$ → output, with feedback loop]**
> *Figure 3.3: Recursive FIR via comb filter and accumulator — pole-zero cancellation preserves the FIR response.*

### 3.1.4 Frequency-Sampling Form

When the desired DFT values $H(k)$ are specified, the transfer function is:

$$H(z) = \frac{1 - z^{-N}}{N} \sum_{k=0}^{N-1} \frac{H(k)}{1 - W_N^{-k} z^{-1}}, \qquad W_N = e^{-j2\pi/N}$$

This is a **comb filter** cascaded with a **bank of $N$ first-order resonators**, each tuned to a DFT frequency $\omega_k = 2\pi k/N$. Transition-band samples can be optimized to minimize stopband ripple.

> **[PLACEHOLDER: Frequency-sampling FIR: comb filter $(1-z^{-N})/N$ feeding parallel resonator bank]**
> *Figure 3.4: Frequency-sampling FIR implementation.*

---

## 3.2 IIR Filter Implementations

An **IIR (Infinite Impulse Response)** filter has transfer function:

$$H(z) = \frac{\sum_{k=0}^{N} a_k z^{-k}}{1 + \sum_{k=1}^{N} b_k z^{-k}}$$

IIR filters achieve sharp frequency selectivity at low order, but cannot have exactly linear phase, and stability must be explicitly ensured (all poles inside the unit circle).

### 3.2.1 Direct Form I

Implements the difference equation with two separate delay chains — one for input terms $x(n-k)$ (FIR part) and one for output feedback terms $y(n-k)$.

- Total delay elements: $2N$

> **[PLACEHOLDER: Direct Form I signal flow graph — input delay chain with multipliers $a_k$, output delay chain with multipliers $b_k$]**
> *Figure 3.5: Direct Form I IIR filter.*

### 3.2.2 Direct Form II — Canonical Form

Merge the two delay chains by sharing a single intermediate state variable $u(n)$:

$$u(n) = x(n) - \sum_{k=1}^{N} b_k\, u(n-k)$$

$$y(n) = \sum_{k=0}^{N} a_k\, u(n-k)$$

- Delay elements: $N$ — the minimum possible (**canonical form**)

> **[PLACEHOLDER: Direct Form II signal flow graph — single shared delay chain; feedback on left, feedforward on right]**
> *Figure 3.6: Direct Form II IIR filter — canonical structure.*

**Transposed Direct Form II**: Reversing the signal flow graph (transpose) yields different accumulation order and superior finite-precision performance.

### 3.2.3 Cascade Form

Factor $H(z)$ into real-coefficient second-order sections:

$$H(z) = \prod_{k=1}^{\lfloor N/2 \rfloor} H_k(z), \qquad H_k(z) = \frac{a_{0k} + a_{1k}z^{-1} + a_{2k}z^{-2}}{1 + b_{1k}z^{-1} + b_{2k}z^{-2}}$$

Each biquad is implemented as Direct Form II. Coefficient sensitivity is localized; the format used by virtually all practical implementations (e.g., MATLAB `sos` format).

> **[PLACEHOLDER: IIR cascade form — series of Direct-Form-II biquad sections]**
> *Figure 3.7: IIR cascade implementation.*

### 3.2.4 Parallel Form

Expand $H(z)$ via partial-fraction decomposition:

$$H(z) = C + \sum_{k=1}^{\lfloor N/2 \rfloor} \frac{a_{0k} + a_{1k}z^{-1}}{1 + b_{1k}z^{-1} + b_{2k}z^{-2}}$$

Each branch is an independent second-order section computed in parallel.

**Advantages**: Best finite-precision performance — round-off errors in each section do not accumulate across branches. Suitable for highly parallel hardware.

> **[PLACEHOLDER: IIR parallel form — constant $C$ plus bank of parallel second-order sections]**
> *Figure 3.8: IIR parallel implementation.*

---

## 3.3 Digital Filter Design

### 3.3.1 Ideal and Practical Frequency Responses

**Ideal filter shapes** (brickwall responses):
- **Lowpass**: passband $[0, \omega_c]$, stopband $[\omega_c, \pi]$
- **Highpass**: stopband $[0, \omega_c]$, passband $[\omega_c, \pi]$
- **Bandpass**: passband $[\omega_1, \omega_2]$
- **Bandstop**: stopband $[\omega_1, \omega_2]$

> **[PLACEHOLDER: Four ideal magnitude responses (lowpass, highpass, bandpass, bandstop) vs $\omega \in [0,\pi]$]**
> *Figure 3.9: Ideal (brickwall) filter shapes.*

**Practical filter specification**:

$$1 - \delta_1 \leq \lvert H(e^{j\omega})\rvert \leq 1 + \delta_1 \quad (\text{passband},\ \omega \leq \omega_p)$$

$$\lvert H(e^{j\omega})\rvert \leq \delta_2 \quad (\text{stopband},\ \omega \geq \omega_s)$$

In dB: passband ripple $R_p = -20\log_{10}(1-\delta_1)$ dB; stopband attenuation $A_s = -20\log_{10}(\delta_2)$ dB.

> **[PLACEHOLDER: Practical filter frequency response with passband, transition band, stopband, and tolerances $\delta_1$, $\delta_2$, $\omega_p$, $\omega_s$ labeled]**
> *Figure 3.10: Practical filter specification.*

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
0.1102\,(A_s - 8.7) & A_s > 50 \\
0.5842\,(A_s - 21)^{0.4} + 0.07886\,(A_s - 21) & 21 \leq A_s \leq 50 \\
0 & A_s < 21
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

> **[PLACEHOLDER: Comparison of IIR design methods: (a) impulse invariance — aliased frequency mapping; (b) bilinear transform — warped but alias-free frequency mapping]**
> *Figure 3.11: IIR design method comparison.*

---

# Part IV: Special Sequences and Corresponding Filters

> 📖 Textbook §2.4 (Minimum-Phase and System Invertibility, §2.4.1–§2.4.4); §2.5 (Lattice Filter Realizations)

---

## 4.1 Allpass Sequences and Allpass Filters

### 4.1.1 Definition and Properties

An **allpass filter** has unit magnitude response at all frequencies:

$$\lvert H_{ap}(e^{j\omega})\rvert = 1 \quad \forall\, \omega$$

This implies:

$$H_{ap}(e^{j\omega})\, H_{ap}^*(e^{j\omega}) = 1, \qquad h_{ap}(n) * h_{ap}^*(-n) = \delta(n)$$

$$\boxed{H_{ap}(z)\, H_{ap}^*(1/z^*) = 1}$$

**Pole-zero structure**: For a stable rational allpass filter, every pole at $z = c_k$ (inside the unit circle) is paired with a zero at $z = 1/c_k^*$ (outside the unit circle) — the **conjugate reciprocal** location.

**First-order allpass** ($\lvert\alpha\rvert < 1$, $\alpha \in \mathbb{C}$):

$$H_{ap1}(z) = \frac{z^{-1} - \alpha^*}{1 - \alpha\, z^{-1}}$$

**General $N$-th order allpass**:

$$H_{ap}(z) = \prod_{k=1}^{N} \frac{z^{-1} - \alpha_k^*}{1 - \alpha_k\, z^{-1}} = \frac{z^{-N} + a_1^* z^{-N+1} + \cdots + a_N^*}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}$$

The numerator polynomial is the **conjugate-reversed** polynomial of the denominator.

> **[PLACEHOLDER: Pole-zero diagram for a first-order allpass filter: pole at $z=\alpha$ inside unit circle, zero at $z=1/\alpha^*$ outside]**
> *Figure 4.1: Allpass filter pole-zero structure.*

**Use cases**: Phase equalization (correcting phase distortion without altering magnitude), building blocks in lattice filters and filter banks, Schur-Cohn stability testing.

### 4.1.2 Group Delay of Allpass Filters — Proof That Group Delay is Always Positive

The **group delay** of a filter is:

$$\tau(\omega) = -\frac{d}{d\omega}\angle H(e^{j\omega})$$

For the first-order real allpass filter

$$H_{ap}(z) = \frac{z^{-1} - \alpha}{1 - \alpha z^{-1}}$$

where $\lvert\alpha\rvert < 1$ and $\alpha \in \mathbb{R}$, the phase response is:

$$\angle H_{ap}(e^{j\omega}) = \omega - 2\arctan\!\left(\frac{\alpha \sin\omega}{1 - \alpha\cos\omega}\right) - \pi$$

Differentiating and negating:

$$\tau(\omega) = -\frac{d}{d\omega}\angle H_{ap}(e^{j\omega}) = \frac{1 - \alpha^2}{1 - 2\alpha\cos\omega + \alpha^2}$$

$$\boxed{\tau(\omega) = \frac{1 - \lvert\alpha\rvert^2}{\lvert 1 - \alpha e^{-j\omega}\rvert^2} > 0 \quad \forall\,\omega}$$

**Proof that $\tau(\omega) > 0$**:
- Numerator: $1 - \alpha^2 > 0$ since $\lvert\alpha\rvert < 1$
- Denominator: $\lvert 1 - \alpha e^{-j\omega}\rvert^2 > 0$ since the pole $\alpha$ is not on the unit circle

**General $N$-th order result**: The group delay of any stable allpass filter is always positive:

$$\tau(\omega) = \sum_{k=1}^{N} \frac{1 - \lvert\alpha_k\rvert^2}{\lvert 1 - \alpha_k e^{-j\omega}\rvert^2} > 0 \quad \forall\,\omega$$

This is a sum of positive terms. An allpass filter is a pure **phase-lag device**: it introduces causal group delay at every frequency without altering the magnitude response. This makes allpass filters ideal for **phase equalization**.

---

## 4.2 Minimum-Phase Sequences and Minimum-Phase Filters

### 4.2.1 Definition

A causal, stable filter $H_m(z)$ is **minimum-phase** if and only if:
1. **Stable**: ROC includes the unit circle (all poles inside)
2. **Causal**: $h_m(n) = 0$ for $n < 0$
3. **All zeros inside or on the unit circle**

Among all causal, stable filters with the same magnitude response, the minimum-phase filter has:
- The smallest phase lag at every frequency
- The fastest energy buildup: $\sum_{n=0}^{k}\lvert h_m(n)\rvert^2 \geq \sum_{n=0}^{k}\lvert h(n)\rvert^2$ for all $k$
- A **causal, stable inverse** $1/H_m(z)$ (all zeros inside the unit circle → all inverse poles inside)

### 4.2.2 Theorem 1.1: Minimum-Phase / Allpass Decomposition

> **Theorem 1.1**: Any causal, stable system $H(z)$ can be **uniquely** decomposed as:
>
> $$\boxed{H(z) = H_{ap}(z) \cdot H_m(z)}$$
>
> where $H_m(z)$ is minimum-phase and $H_{ap}(z)$ is allpass.

**Proof by construction**:

Given $H(z)$ with poles $\{p_k\}$ (all inside the unit circle) and zeros $\{z_k\}$ (some possibly outside):

**Step 1** — Partition zeros:
- $\mathcal{Z}_{in} = \{z_k : \lvert z_k\rvert \leq 1\}$: zeros inside or on the unit circle
- $\mathcal{Z}_{out} = \{z_k : \lvert z_k\rvert > 1\}$: zeros outside the unit circle

**Step 2** — Construct $H_m(z)$:
- Assign all poles $\{p_k\}$ to $H_m(z)$
- Assign all zeros in $\mathcal{Z}_{in}$ to $H_m(z)$
- For each zero $c \in \mathcal{Z}_{out}$: add a zero at $1/c^*$ (its conjugate reciprocal, inside the unit circle) to $H_m(z)$

The resulting $H_m(z)$ is causal, stable, and all-zeros-inside — minimum-phase.

**Step 3** — Construct $H_{ap}(z) = H(z)/H_m(z)$:
- For each $c \in \mathcal{Z}_{out}$: $H_{ap}(z)$ inherits a zero at $c$ (outside) and a pole at $1/c^*$ (inside) — exactly one allpass factor $\dfrac{z^{-1} - c^*}{1 - c\, z^{-1}}$

**Verification**: $\lvert H(e^{j\omega})\rvert = \lvert H_{ap}(e^{j\omega})\rvert \cdot \lvert H_m(e^{j\omega})\rvert = 1 \cdot \lvert H_m(e^{j\omega})\rvert$. The phase of $H$ exceeds that of $H_m$ by the allpass phase lag (always positive group delay). ✓

> **[PLACEHOLDER: Pole-zero diagram: H(z) with zeros inside and outside unit circle → H_ap(z) (zeros outside, poles inside) × H_m(z) (all zeros inside)]**
> *Figure 4.2: Minimum-phase/allpass decomposition — zeros outside the unit circle are reflected inside (with allpass compensation) to form the minimum-phase factor.*

---

## 4.3 Linear-Phase Sequences and Linear-Phase Filters

### 4.3.1 Strict and Generalized Linear Phase

A filter has **generalized linear phase** if:

$$H(e^{j\omega}) = e^{j\beta}\, e^{-j\alpha\omega}\, A(e^{j\omega})$$

where $A(e^{j\omega})$ is **real-valued** (amplitude function), $\alpha$ is the **constant group delay** (samples), and $\beta \in \{0, \pm\pi/2\}$.

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

> **[PLACEHOLDER: Four-panel figure — (top) $h(n)$ impulse responses for Types I–IV with symmetry; (bottom) typical amplitude responses $\lvert A(e^{j\omega})\rvert$]**
> *Figure 4.3: Four types of linear-phase FIR filters.*

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

$$r_x(n) = \sum_{k=-\infty}^{\infty} x(k)\, x^*(k-n) = x(n) * x^*(-n)$$

Its DTFT is the **power spectral density**:

$$P_x(e^{j\omega}) = \lvert X(e^{j\omega})\rvert^2 \geq 0$$

Autocorrelation sequences are therefore **always positive semi-definite**. The autocorrelation matrix $\mathbf{R} = [r_x(i-j)]_{i,j}$ is a **Hermitian Toeplitz positive semi-definite matrix** — the central object in Wiener filtering (Chapter 6) and linear prediction (Chapter 3).

### 4.4.2 Theorem 1.2: Zero Pairing in Rational Positive Semi-Definite Sequences

> **Theorem 1.2**: Let $R(z)$ be the rational z-transform of a positive semi-definite sequence with real-valued coefficients. Then:
> 1. **Zeros on the unit circle** ($\lvert z_0\rvert = 1$): occur in **conjugate pairs** $(z_0,\, z_0^*)$, each with **even multiplicity**
> 2. **Zeros off the unit circle**: occur in **quadruples** $\{z_0,\; z_0^*,\; 1/z_0,\; 1/z_0^*\}$

**Explanation**:
- Hermitian symmetry ($R(z) = R^*(1/z^*)$, real coefficients): if $z_0$ is a zero, then so are $z_0^*$, $1/z_0$, and $1/z_0^*$
- Non-negativity $R(e^{j\omega}) \geq 0$: zeros on the unit circle must have even multiplicity, otherwise $R(e^{j\omega})$ would change sign

**Spectral Factorization**: Any rational positive semi-definite sequence factors uniquely as:

$$\boxed{R(z) = \sigma^2\, H_m(z)\, H_m^*(1/z^*)}$$

On the unit circle: $R(e^{j\omega}) = \sigma^2\lvert H_m(e^{j\omega})\rvert^2$, where $H_m(z)$ is **minimum-phase** (all zeros inside the unit circle). The quadruple structure $\{z_0, z_0^*, 1/z_0, 1/z_0^*\}$ splits as $\{1/z_0, 1/z_0^*\} \to H_m(z)$ and $\{z_0, z_0^*\} \to H_m^*(1/z^*)$.

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
| Allpass filter | $\lVert H_{ap}\rVert=1$; group delay $\tau(\omega) > 0$ always | Phase equalization |
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
