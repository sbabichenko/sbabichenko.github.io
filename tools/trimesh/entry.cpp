// Browser entry: the engine's own main(), renamed at compile time, called with a seed.
#include <emscripten/emscripten.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
int dm_engine_main(int argc, char** argv);
// env: newline-separated KEY=VALUE pairs set with setenv before the run (an empty VALUE unsets KEY),
// because Emscripten's ENV object is frozen into environ at startup.
extern "C" EMSCRIPTEN_KEEPALIVE int dm_run(int seed, const char* env) {
    std::istringstream lines(env ? env : "");
    for (std::string line; std::getline(lines, line);) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        if (value.empty()) unsetenv(key.c_str()); else setenv(key.c_str(), value.c_str(), 1);
    }
    std::string s = std::to_string(seed);
    char prog[] = "trimesh";
    char* argv[] = {prog, s.data(), nullptr};
    const int rc = dm_engine_main(2, argv);
    std::fflush(stdout); std::fflush(stderr);
    return rc;
}
