/* Symbols exported by libcoralnpu_simulator_mpact.so: the API in
 * coralnpu_simulator.h, whose only entry point is CoralNPUSimulator::Create().
 * Everything else (MPACT, the statically linked protobuf and abseil) stays
 * local, so it cannot collide with another copy in the same process, such as
 * TensorFlow's protobuf. Keep in sync with exported_symbols_macos.txt. */
{
  global:
    extern "C++" {
      "CoralNPUSimulator::Create()";
    };
  local:
    *;
};
