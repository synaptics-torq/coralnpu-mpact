// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <signal.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "sim/coralnpu_v2_simulator.h"
#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

using ::coralnpu::sim::CoralNPUV2Simulator;
using ::coralnpu::sim::CoralNPUV2SimulatorOptions;

// Flags for specifying interactive mode.
ABSL_FLAG(bool, i, false, "Interactive mode");
ABSL_FLAG(bool, interactive, false, "Interactive mode");

ABSL_FLAG(std::optional<uint32_t>, entry_point, std::nullopt,
          "Optionally set the entry point of the program.");

ABSL_FLAG(uint32_t, itcm_start_address, 0x0,
          "Set the start address of the ITCM range.");
ABSL_FLAG(uint32_t, itcm_length, 0x2000, "Set the length of the ITCM range.");
ABSL_FLAG(uint32_t, initial_misa_value, 0x40201120,
          "Set the initial value of the misa register.");
ABSL_FLAG(bool, exit_on_ebreak, false, "Exit on ebreak instruction.");

ABSL_FLAG(std::vector<std::string>, allow_lsu_range, {"0x10000:0x8000"},
          "Allowed LSU range. Format is start_address:length. "
          "Repeat this option to specify multiple ranges.");

// Static pointer to the simulator instance. Used by the control-C handler.
static CoralNPUV2Simulator* g_simulator = nullptr;

// Control-c handler to interrupt any running simulation.
static void sim_sigint_handler(int arg) {
  if (g_simulator != nullptr) {
    absl::Status status = g_simulator->Halt();
    if (!status.ok()) {
      LOG(ERROR) << "Error halting simulation: " << status;
    }
    return;
  } else {
    exit(-1);
  }
}

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetProgramUsageMessage("CoralNPUV2 MPACT-Sim based CLI tool");
  auto out_args = absl::ParseCommandLine(argc, argv);
  argc = out_args.size();
  argv = &out_args[0];
  if (argc != 2) {
    LOG(ERROR) << "Only a single input file allowed";
    return -1;
  }
  std::string file_name = argv[1];

  CoralNPUV2SimulatorOptions options;
  options.itcm_start_address = absl::GetFlag(FLAGS_itcm_start_address);
  options.itcm_length = absl::GetFlag(FLAGS_itcm_length);
  options.initial_misa_value = absl::GetFlag(FLAGS_initial_misa_value);
  options.exit_on_ebreak = absl::GetFlag(FLAGS_exit_on_ebreak);
  options.lsu_access_ranges.clear();

  for (const std::string& range_str : absl::GetFlag(FLAGS_allow_lsu_range)) {
    std::vector<std::string> range = absl::StrSplit(range_str, ':');
    if (range.size() != 2) {
      LOG(ERROR) << "Invalid LSU range: " << range_str << ". The expected "
                 << "format is start_address:length. Use hex or decimal "
                 << "values.";
      return -1;
    }
    uint32_t start, length;
    std::string error;
    if (!absl::ParseFlag(range[0], &start, &error)) {
      LOG(ERROR) << "Invalid LSU range: " << range_str << " " << error;
      return -1;
    }
    if (!absl::ParseFlag(range[1], &length, &error)) {
      LOG(ERROR) << "Invalid LSU range: " << range_str << " " << error;
      return -1;
    }
    options.lsu_access_ranges.push_back({start, length});
  }

  CoralNPUV2Simulator simulator(options);

  // Set up control-c handling.
  g_simulator = &simulator;
  struct sigaction sa;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sigaddset(&sa.sa_mask, SIGINT);
  sa.sa_handler = &sim_sigint_handler;
  sigaction(SIGINT, &sa, nullptr);

  bool interactive = absl::GetFlag(FLAGS_i) || absl::GetFlag(FLAGS_interactive);
  std::optional<uint32_t> entry_point = absl::GetFlag(FLAGS_entry_point);

  absl::Status load_status = simulator.LoadProgram(file_name, entry_point);
  if (!load_status.ok()) {
    LOG(ERROR) << "Error while loading '" << file_name
               << "': " << load_status.message();
    return -1;
  }

  if (interactive) {
    simulator.RunInteractive();
  } else {
    std::cout << "Starting simulation\n";
    auto t0 = absl::Now();
    absl::Status run_status = simulator.Run();
    if (!run_status.ok()) {
      LOG(ERROR) << run_status.message();
    }
    absl::Status wait_status = simulator.Wait();
    if (!wait_status.ok()) {
      LOG(ERROR) << wait_status.message();
    }
    auto sec = absl::ToDoubleSeconds(absl::Now() - t0);
    std::cout << "Total cycles: " << simulator.GetCycleCount() << '\n';
    std::cout << absl::StrFormat("Simulation done: %0.3f sec\n", sec);
  }

  g_simulator = nullptr;
  return 0;
}
