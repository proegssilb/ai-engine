// The only C++ in the project (decision D3 exception): a C entry point for
// llama.cpp's automatic fit, which lives in its C++ `common` library.
//
// common_fit_params changes global logger state and isn't thread-safe; the
// worker calls it once, before loading a model.

#include "fit.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

extern "C" int aie_fit(const char * model_path, llama_model_params * mparams, llama_context_params * cparams,
                       float * tensor_split, llama_model_tensor_buft_override * overrides, size_t * margins,
                       uint32_t n_ctx_min) {
    try {
        std::vector<std::string> paths{ model_path };
        return (int) common_fit_params(paths, mparams, cparams, tensor_split, overrides, margins, n_ctx_min, nullptr,
                                       GGML_LOG_LEVEL_ERROR);
    } catch (...) {
        return (int) COMMON_PARAMS_FIT_STATUS_ERROR;
    }
}
