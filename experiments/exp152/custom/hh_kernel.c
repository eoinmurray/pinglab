/* HH RK4 CPU kernel. Units: ms, mV, nF, uS and nA.
 * Same sodium/potassium/leak equations as the SNNLab HH example;
 * synaptic reversal currents are recomputed at every RK4 stage.
 */
#include <math.h>
#include <stddef.h>

static double vtrap(double x) {
    return fabs(x) < 1e-4 ? 1 + x / 2 + x * x / 12 : x / -expm1(-x);
}

static void derivative(const double *y, double ge, double gi,
                       const double *p, double *d) {
    const double v = y[0], m = y[1], h = y[2], n = y[3];
    const double ionic = p[1]*m*m*m*h*(v-p[4])
                       + p[2]*n*n*n*n*(v-p[5]) + p[3]*(v-p[6]);
    d[0] = (ge*(p[7]-v) + gi*(p[8]-v) - ionic) / p[0];
    d[1] = vtrap((v+40)/10)*(1-m) - 4*exp(-(v+65)/18)*m;
    d[2] = 0.07*exp(-(v+65)/20)*(1-h) - h/(1+exp(-(v+35)/10));
    d[3] = 0.1*vtrap((v+55)/10)*(1-n) - 0.125*exp(-(v+65)/80)*n;
}

void hh_advance(const double *voltage, const double *m, const double *h,
                const double *n, const double *ge, const double *gi,
                double *out, size_t count, double elapsed_ms,
                double max_step_ms, const double *parameters) {
    const int steps = (int)ceil(elapsed_ms/max_step_ms);
    const double dt = elapsed_ms / steps;
    for (size_t i=0; i<count; ++i) {
        double y[4] = {voltage[i], m[i], h[i], n[i]};
        double k1[4], k2[4], k3[4], k4[4], z[4], next[4];
        double crossed = 0;
        for (int s=0; s<steps; ++s) {
            derivative(y, ge[i], gi[i], parameters, k1);
            for (int j=0; j<4; ++j) z[j] = y[j] + dt/2*k1[j];
            derivative(z, ge[i], gi[i], parameters, k2);
            for (int j=0; j<4; ++j) z[j] = y[j] + dt/2*k2[j];
            derivative(z, ge[i], gi[i], parameters, k3);
            for (int j=0; j<4; ++j) z[j] = y[j] + dt*k3[j];
            derivative(z, ge[i], gi[i], parameters, k4);
            for (int j=0; j<4; ++j)
                next[j] = y[j] + dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j]);
            if (y[0] < 0 && next[0] >= 0) crossed = 1;
            for (int j=0; j<4; ++j) y[j] = next[j];
        }
        for (int j=0; j<4; ++j) out[j*count+i] = y[j];
        out[4*count+i] = crossed;
    }
}
