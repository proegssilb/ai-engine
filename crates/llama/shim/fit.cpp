// The only C++ in the project (decision D3 exception): a C entry point for
// llama.cpp's automatic fit, which lives in its C++ `common` library.
//
// common_fit_params changes global logger state and isn't thread-safe; the
// worker calls it once, before loading a model.

#include "fit.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
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

// llama.cpp's dry run (no_alloc): what each device would hold for this model
// and context, as llama.cpp computes it. Rows: devices, then host, then other
// buffer types. Returns the row count, or -1 on failure.
struct aie_mem_row {
    char     name[128];
    int32_t  kind; // 0 device, 1 host, 2 buffer type
    uint64_t total;
    uint64_t free;
    uint64_t model;
    uint64_t context;
    uint64_t compute;
};
// Rust's `MemRow` (src/fit.rs) mirrors this layout and asserts the same size.
static_assert(sizeof(aie_mem_row) == 176, "aie_mem_row layout changed: update MemRow in src/fit.rs");

extern "C" int aie_measure(const char * model_path, const llama_model_params * mparams,
                           const llama_context_params * cparams, aie_mem_row * rows, int max_rows) {
    try {
        std::vector<std::string> paths{ model_path };
        auto r = common_fit_measure(paths, mparams, cparams, nullptr, nullptr, GGML_LOG_LEVEL_ERROR);
        int n = 0;
        for (const auto & x : r) {
            if (n >= max_rows) {
                break;
            }
            aie_mem_row & o = rows[n++];
            snprintf(o.name, sizeof(o.name), "%s", x.name.c_str());
            o.kind    = (int32_t) x.kind;
            o.total   = (uint64_t) x.total;
            o.free    = (uint64_t) x.free;
            o.model   = (uint64_t) x.model;
            o.context = (uint64_t) x.context;
            o.compute = (uint64_t) x.compute;
        }
        return n;
    } catch (...) {
        return -1;
    }
}

// The same breakdown for a live context: what this context really allocated,
// per device and on the host. Returns the row count, or -1 on failure.
extern "C" int aie_ctx_breakdown(const llama_context * ctx, aie_mem_row * rows, int max_rows) {
    try {
        auto r = common_memory_breakdown_rows(ctx);
        int n = 0;
        for (const auto & x : r) {
            if (n >= max_rows) {
                break;
            }
            aie_mem_row & o = rows[n++];
            snprintf(o.name, sizeof(o.name), "%s", x.name.c_str());
            o.kind    = (int32_t) x.kind;
            o.total   = (uint64_t) x.total;
            o.free    = (uint64_t) x.free;
            o.model   = (uint64_t) x.model;
            o.context = (uint64_t) x.context;
            o.compute = (uint64_t) x.compute;
        }
        return n;
    } catch (...) {
        return -1;
    }
}
