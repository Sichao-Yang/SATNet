#include <assert.h>
#include <float.h>
#include <math.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <omp.h>

#include "satnet.h"

#define saxpy mysaxpy
#define scopy myscopy
#define sscal mysscal
#define sdot mysdot
#define snrm2 mysnrm2
#define szero myszero
#define saturate mysaturate

const double MEPS = 1e-24;

// The saxpy function performs the operation y =a*x+y for vectors x and y of
// length l. The implementation provided uses SIMD (Single Instruction, Multiple
// Data) instructions to speed up the computation, specifically using the SSE
// (Streaming SIMD Extensions) instructions for floating-point operations.
void saxpy(float *__restrict__ y, float a, const float *__restrict__ x, int l) {
  y = (float *)__builtin_assume_aligned(y, 4 * sizeof(float));
  x = (float *)__builtin_assume_aligned(x, 4 * sizeof(float));
  __m128 const a_ = _mm_set1_ps(a);
  for (int i = 0; i < l; i += 4, x += 4, y += 4) {
    __m128 y_ = _mm_load_ps(y);
    __m128 x_ = _mm_load_ps(x);
    y_ = _mm_add_ps(_mm_mul_ps(a_, x_), y_);
    _mm_store_ps(y, y_);
  }
}

void scopy(float *x, float *y, int l) { memcpy(y, x, sizeof(*x) * (size_t)l); }

float sdot(const float *__restrict__ x, const float *__restrict__ y, int l) {
  // The sdot function computes the dot product of two aligned float arrays
  // using SIMD (Single Instruction, Multiple Data) instructions for efficiency.
  // This function takes advantage of the SIMD capabilities provided by the SSE
  // (Streaming SIMD Extensions) instructions in the x86 architecture.
  x = (float *)__builtin_assume_aligned(x, 4 * sizeof(float));
  y = (float *)__builtin_assume_aligned(y, 4 * sizeof(float));
  __m128 s = _mm_set1_ps(0);
  for (int i = 0; i < l; i += 4, x += 4, y += 4) {
    __m128 x_ = _mm_load_ps(x);
    __m128 y_ = _mm_load_ps(y);
    __m128 t = _mm_dp_ps(x_, y_, 0xf1);
    s = _mm_add_ss(s, t);
  }
  float s_;
  _mm_store_ss(&s_, s);

  return s_;
}

// The sscal function scales a float array x with length l by a scalar value
// a. The implementation aims to optimize the scaling operation by processing
// multiple elements per loop iteration, reducing the number of iterations and
// potentially increasing cache efficiency.
void sscal(float *x, float a, int l) {
  int m = l - 4;
  int i;
  for (i = 0; i < m; i += 5) {
    x[i] *= a;
    x[i + 1] *= a;
    x[i + 2] *= a;
    x[i + 3] *= a;
    x[i + 4] *= a;
  }

  for (; i < l; i++) /* clean-up loop */
    x[i] *= a;
}

float snrm2(const float *x, int l) {
  float xx = sdot(x, x, l);
  return sqrt(xx);
}
void szero(float *v, int l) { memset(v, 0, l * sizeof(*v)); }

void mix_init(int32_t *perm, int n, int k, const int32_t *is_input,
              int32_t *index, const float *z, float *V) {
  // The mix_init function initializes and transforms the V matrix based on the
  // is_input flag and the z array, normalizing non-input vectors. It then
  // populates the index array based on the permutation and is_input flags.

  // compare to satnet paper equation 4,5, here if we initial implicitly v_true
  // as [1,0,0...] and v_rand as [0, 1 or -1, 0,0,...], then the following
  // initilization process complies with eq. 4&5

  // rand_unit(V+0, k);
  for (int i = 0; i < n; i++) {
    if (is_input[i]) {
      float Vi1 = V[i * k + 1];
      szero(V + i * k, k); // sets the i-th vector of V to zero.
      V[i * k] = -cos(z[i] * M_PI);
      V[i * k + 1] = copysign(sin(z[i] * M_PI), Vi1);
    } else {
      float s = sdot(V + i * k, V + i * k, k);
      s = 1 / sqrtf(s);
      sscal(V + i * k, s, k);
    }
  }
  // added some randomness in the coordinate decent, only permute the output
  int i_ = 0, j = 0;
  for (; i_ < n - 1; i_++) {
    int i = perm[i_] + 1;
    if (!is_input[i])
      index[j++] = i;
  }
  for (; j < n; j++)
    index[j] = 0;
}

float mix_kernel(int is_forward, float prox_lam, int m, int k,
                 const int32_t *__restrict__ index, const float *__restrict__ S,
                 const float *__restrict__ dz, float *__restrict__ V,
                 const float *__restrict__ Vproj, float *__restrict__ W,
                 float *__restrict__ gnrm, const float *__restrict__ Snrms,
                 float *__restrict__ g) {
  // this function is for both Algo2 (forward pass) and 3 (backward pass),
  // in backward pass, U is mapped to V, V is mapped to Vproj (to
  // calculate P), phi is mapped to W, dg is mapped to g
  float delta = 0;
  for (int i, i_ = 0; (i = index[i_]); i_++) {
    const float Sii = Snrms[i];
    const float *__restrict__ Si = S + i * m;

    // Algo2 line6: go = W'So - Snorm^2*vo,
    // dim: g: kx1, Si: mx1, W: kxm, Sii: scalar, V: nxk
    // first part of line6: p1=omega*so
    // Algo3 line6: dgo = phi'So - Snorm^2*uo
    for (int kk = 0; kk < k; kk++)
      g[kk] = sdot(Si, W + kk * m, m);
    // second part: p1 -s_norm^2*vo, y=p1, a=-s_norm^2, x=vo
    saxpy(g, -Sii, V + i * k, k);

    float gnrmi;
    if (is_forward) {
      // algo2 line7: vo=-go/norm(go)
      gnrmi = snrm2(g, k);
      sscal(g, -1, k);
    } else {
      // algo3 line7: g = -(I-v_i v_i') (g+v_0 dz[i])
      // this part is very tricky, for detailed alignment, please refer to .md
      // file
      gnrmi = gnrm[i] + prox_lam;
      float c = sdot(Vproj + i * k, g, k) + dz[i] * Vproj[i * k];
      sscal(g, -1, k);
      saxpy(g, c, Vproj + i * k, k);
      g[0] -= dz[i];
    }
    sscal(g, 1 / gnrmi, k);

    float t;
    // algo2: cooresponds to line 7's assignment, and line8's calculation
    // define a temprory var, set vo_new = g, and store vo_new-vo_old into g
    // algo3 line 7 & 8
    for (int kk = 0; kk < k; kk++)
      t = g[kk], g[kk] -= V[i * k + kk], V[i * k + kk] = t;
    // W += (vi^new-vi^old) Si'
    for (int kk = 0; kk < k; kk++)
      saxpy(W + kk * m, g[kk], Si, m);

    if (is_forward) {
      // Calc function decrease: gnrmi represents gradient size, sdot(g, g, k)
      // represents vo difference size (vo difference is stored in g)
      delta += gnrmi * sdot(g, g, k);
      gnrm[i] = gnrmi;
    }
  }
  return delta;
}

// clamp a floating-point value x within the range [0, 1].
inline float saturate(float x) { return x - (x < 0) * x + (x > 1) * (1 - x); }

// consider the \min unsat problem,
void mix_forward(int max_iter, float eps, int n, int m, int k,
                 const int32_t *index, int32_t *niter, const float *S, float *z,
                 float *V, float *W, float *gnrm, float *Snrms, float *cache) {
  float delta;
  int iter = 0;
  // this is the outer loop in algo2 line4
  for (; iter < max_iter; iter++) {
    delta =
        mix_kernel(1, 0, m, k, index, S, NULL, V, NULL, W, gnrm, Snrms, cache);
    if (iter && delta < eps)
      break;
    if (iter == 0)
      eps = delta * eps;
  }

  *niter = iter;

  // because v_true is set to [1,0,0], dot(vo,v_true)=vo[0] therefore eq.7 can
  // be computed as: 1 - acosf(vo[0]) / M_PI, notice acos(-z)/pi == 1-acos(z)/pi
  // saturate is just to map anything into [0,1] for probabilistic output
  for (int i, i_ = 0; (i = index[i_]); i_++) {
    float zi = V[i * k];
    zi = saturate((zi + 1) / 2) * 2 - 1;
    zi = saturate(1 - acosf(zi) / M_PI);
    z[i] = zi;
  }
}

void mix_backward(float prox_lam, int n, int m, int k, int32_t *is_input,
                  int32_t *index, int32_t *niter, const float *S, float *dS,
                  float *z, float *dz, const float *V, float *U, float *W,
                  float *Phi, float *gnrm, float *Snrms, float *cache) {

  // eq.8 to get dvo
  int invalid_flag = 0;
  for (int i, i_ = 0; (i = index[i_]); i_++) {
    float zi = z[i];
    float dzi = dz[i] / M_PI / sin(zi * M_PI);
    if (isnan(dzi) || isinf(dzi) || gnrm[i] < MEPS)
      invalid_flag = 1;
    dz[i] = dzi;
  }
  if (invalid_flag) {
    szero(dz, n);
    return;
  }

  // eq.9, solve P (S'S+D_z-D_sii)xI_k P U = -dz P v0 approximately
  for (int iter = 0; iter < *niter; iter++) {
    mix_kernel(0, prox_lam, m, k, index, S, dz, U, V, Phi, gnrm, Snrms, cache);
  }

  // sanity check
  for (int ik = 0; ik < n * k; ik++) {
    if (isnan(U[ik]) || isinf(U[ik]))
      invalid_flag = 1;
  }
  if (invalid_flag) {
    szero(dz, n);
    return;
  }

  // eq.11, dS = U W + V Phi
  // dS: nxm, U: nxk, W: kxm
  for (int i = 0; i < n; i++) {
    for (int kk = 0; kk < k; kk++) {
      saxpy(dS + i * m, U[i * k + kk], W + kk * m, m);
      saxpy(dS + i * m, V[i * k + kk], Phi + kk * m, m);
    }
  }
  // eq.10,12, 13, dzi = v0'Phi si
  for (int i = 1; i < n; i++) {
    if (!is_input[i]) {
      dz[i] = 0;
      continue;
    }
    float val1 = sdot(S + i * m, Phi + 0 * m, m),
          val2 = sdot(S + i * m, Phi + 1 * m, m);
    dz[i] = (dz[i] + val1) * sin(z[i] * M_PI) * M_PI +
            val2 * copysign(cos(z[i] * M_PI) * M_PI, V[i * k + 1]) * M_PI;
  }
}

void mix_init_launcher_cpu(mix_t mix, int32_t *perm) {
  int n = mix.n, k = mix.k;
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < mix.b; i++) {
    mix_init(perm, mix.n, mix.k, mix.is_input + i * n, mix.index + i * n,
             mix.z + i * n, mix.V + i * n * k);
  }
}

void mix_forward_launcher_cpu(mix_t mix, int max_iter, float eps) {
  int n = mix.n, m = mix.m, k = mix.k;
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < mix.b; i++) {
    mix_forward(max_iter, eps, mix.n, mix.m, mix.k, mix.index + i * n,
                mix.niter + i, mix.S, mix.z + i * n, mix.V + i * n * k,
                mix.W + i * m * k, mix.gnrm + i * n, mix.Snrms,
                mix.cache + i * k);
  }
}

void mix_backward_launcher_cpu(mix_t mix, float prox_lam) {
  int n = mix.n, m = mix.m, k = mix.k;
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < mix.b; i++) {
    mix_backward(prox_lam, mix.n, mix.m, mix.k, mix.is_input + i * n,
                 mix.index + i * n, mix.niter + i, mix.S, mix.dS + i * n * m,
                 mix.z + i * n, mix.dz + i * n, mix.V + i * n * k,
                 mix.U + i * n * k, mix.W + i * m * k, mix.Phi + i * m * k,
                 mix.gnrm + i * n, mix.Snrms, mix.cache + i * k);
  }
}
