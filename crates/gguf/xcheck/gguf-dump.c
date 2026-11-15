#include <stdio.h>
#include "ggml.h"
#include "gguf.h"
int main(int argc, char **argv) {
  struct ggml_context *meta = NULL;
  struct gguf_init_params p = { .no_alloc = true, .ctx = &meta };
  struct gguf_context *g = gguf_init_from_file(argv[1], p);
  if (!g) { fprintf(stderr, "llama.cpp gguf failed\n"); return 1; }
  printf("version %u alignment %zu data_offset %zu n_kv %lld\n", gguf_get_version(g), gguf_get_alignment(g), gguf_get_data_offset(g), (long long)gguf_get_n_kv(g));
  for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
    struct ggml_tensor *t = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
    printf("%s %s [%lld,%lld,%lld,%lld] off %zu size %zu\n", gguf_get_tensor_name(g, i), ggml_type_name(gguf_get_tensor_type(g, i)),
      (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], gguf_get_tensor_offset(g, i), gguf_get_tensor_size(g, i));
  }
  return 0;
}
