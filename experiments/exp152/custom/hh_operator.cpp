// Torch dispatcher bridge; the C kernel owns only the HH RK4 integration.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

extern "C" void hh_advance(const double*, const double*, const double*, const double*,
                           const double*, const double*, double*, size_t, double,
                           double, const double*);

static at::Tensor advance(const at::Tensor& voltage, const at::Tensor& m,
                          const at::Tensor& h, const at::Tensor& n,
                          const at::Tensor& excitation, const at::Tensor& inhibition,
                          double dt_ms, std::vector<double> parameters) {
    TORCH_CHECK(dt_ms > 0 && std::isfinite(dt_ms), "Invalid HH timestep");
    TORCH_CHECK(parameters.size() == 9 && parameters[0] > 0, "Invalid HH parameters");
    for (double value : parameters) TORCH_CHECK(std::isfinite(value), "Nonfinite HH parameter");
    std::vector<at::Tensor> inputs;
    for (const auto& input : {voltage, m, h, n, excitation, inhibition}) {
        TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kDouble
                    && input.sizes() == voltage.sizes(), "HH needs matching CPU float64 tensors");
        inputs.push_back(input.contiguous());
    }
    auto shape = voltage.sizes().vec();
    shape.insert(shape.begin(), 5);
    auto output = at::empty(shape, voltage.options());
    hh_advance(inputs[0].data_ptr<double>(), inputs[1].data_ptr<double>(),
               inputs[2].data_ptr<double>(), inputs[3].data_ptr<double>(),
               inputs[4].data_ptr<double>(), inputs[5].data_ptr<double>(),
               output.data_ptr<double>(), voltage.numel(), dt_ms, 0.025, parameters.data());
    return output;
}

TORCH_LIBRARY(pinglab_exp152, m) {
    m.def("hh_advance(Tensor voltage, Tensor m, Tensor h, Tensor n, Tensor excitation, "
          "Tensor inhibition, float dt_ms, float[] parameters) -> Tensor");
}
TORCH_LIBRARY_IMPL(pinglab_exp152, CPU, m) {
    m.impl("hh_advance", TORCH_FN(advance));
}
