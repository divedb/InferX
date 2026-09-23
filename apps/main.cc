#include "cli/app.h"

int main(int argc, char** argv) {
  inferx::cli::InferxCli cli{"InferX — High-performance LLM inference engine",
                             "inferx"};
  return cli.Run(argc, argv);
}
