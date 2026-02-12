// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIM_CORALNPU_V2_STATE_H_
#define SIM_CORALNPU_V2_STATE_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

namespace internal {
// StretchMisa32 stretches the 32-bit value into a 64-bit value by moving the
// upper 2 bits of the 32-bit input to the upper 2 bits of the 64-bit output.
static inline uint64_t StretchMisa32(uint32_t value) {
  uint64_t value64 = static_cast<uint64_t>(value);
  value64 = ((value64 & 0xc000'0000) << 32) | (value64 & 0x03ff'ffff);
  return value64;
}
}  // namespace internal

struct CoralNPUV2LsuAccessRange {
  uint32_t start_address;
  uint32_t length;
};

constexpr int kCoralnpuV2VectorByteLength = 16;
constexpr uint32_t kCoralNPUV2DefaultItcmStartAddress = 0;
constexpr uint32_t kCoralNPUV2DefaultItcmLength = 0x2000;
constexpr uint32_t kCoralNPUV2DefaultDtcmStartAddress = 0x10000;
constexpr uint32_t kCoralNPUV2DefaultDtcmLength = 0x8000;
constexpr uint32_t kCoralNPUV2DefaultExtmemStartAddress = 0x20000000;
constexpr uint32_t kCoralNPUV2DefaultExtmemLength = 0x400000;
constexpr uint32_t kCoralNPUV2DefaultInitialMisaValue =
    (*::mpact::sim::riscv::RiscVXlen::RV32 << 30) |
    *::mpact::sim::riscv::IsaExtension::kIntegerMulDiv |
    *::mpact::sim::riscv::IsaExtension::kRVIBaseIsa |
    *::mpact::sim::riscv::IsaExtension::kSinglePrecisionFp |
    *::mpact::sim::riscv::IsaExtension::kVectorExtension;

class CoralNPUV2State : public ::mpact::sim::riscv::RiscVState {
 public:
  using AtomicMemoryOpInterface = ::mpact::sim::util::AtomicMemoryOpInterface;
  using MemoryInterface = ::mpact::sim::util::MemoryInterface;
  using RiscVState = ::mpact::sim::riscv::RiscVState;
  using RiscVXlen = ::mpact::sim::riscv::RiscVXlen;
  using Instruction = ::mpact::sim::generic::Instruction;

  CoralNPUV2State(absl::string_view id, RiscVXlen xlen,
                  MemoryInterface* /*absl_nonnull*/ memory,
                  AtomicMemoryOpInterface* /*absl_nullable*/ atomic_memory);
  CoralNPUV2State(absl::string_view id, RiscVXlen xlen,
                  MemoryInterface* /*absl_nonnull*/ memory)
      : CoralNPUV2State(id, xlen, memory, nullptr) {}
  ~CoralNPUV2State() override;

  // Deleted Constructors and operators.
  CoralNPUV2State(const CoralNPUV2State&) = delete;
  CoralNPUV2State(CoralNPUV2State&&) = delete;
  CoralNPUV2State& operator=(const CoralNPUV2State&) = delete;
  CoralNPUV2State& operator=(CoralNPUV2State&&) = delete;

  // Dispatches the mpause instruction to the registered handlers. If no
  // handlers return true, the instruction is trapped.
  void MPause(const Instruction* instruction);

  // Registers a handler for the `mpause` instruction.
  // When an `mpause` instruction is executed, handlers are invoked in
  // registration order. If a handler returns `true`, it is assumed to have
  // handled the instruction, and no further handlers are invoked.
  void AddMpauseHandler(absl::AnyInvocable<bool(const Instruction*)> handler) {
    on_mpause_.emplace_back(std::move(handler));
  }

  uint32_t itcm_start_address() const { return itcm_start_address_; }
  uint32_t itcm_length() const { return itcm_length_; }
  void set_itcm_start_address(uint32_t start_address) {
    itcm_start_address_ = start_address;
  }
  void set_itcm_length(uint32_t length) { itcm_length_ = length; }

  bool IsAddressInItcmRange(uint32_t address) const {
    // CoralNPUV2 only uses 32 bit instructions so this simple range check is
    // sufficient.
    return address >= itcm_start_address_ &&
           address <= itcm_start_address_ + itcm_length_ - sizeof(address);
  }

  void AddLsuAccessRange(uint32_t start_address, uint32_t length) {
    lsu_access_ranges_.push_back(
        {.start_address = start_address, .length = length});
  }

  bool IsLsuAccessValid(uint32_t address, uint32_t size);
  bool IsJumpValid(uint32_t address);

 private:
  std::vector<absl::AnyInvocable<bool(const Instruction*)>> on_mpause_;
  std::vector<CoralNPUV2LsuAccessRange> lsu_access_ranges_;
  uint32_t itcm_start_address_ = 0;
  uint32_t itcm_length_ = 0;
};

struct CoralNPUV2StateConfig {
  // The start address of the ITCM range.
  uint32_t itcm_start_address = kCoralNPUV2DefaultItcmStartAddress;
  // The length of the ITCM range.
  uint32_t itcm_length = kCoralNPUV2DefaultItcmLength;
  // The initial value of the misa register.
  uint32_t initial_misa_value = kCoralNPUV2DefaultInitialMisaValue;
  // The ranges that are allowed for LSU access.
  std::vector<CoralNPUV2LsuAccessRange> lsu_access_ranges = {
      {.start_address = kCoralNPUV2DefaultDtcmStartAddress,
       .length = kCoralNPUV2DefaultDtcmLength},
      {.start_address = kCoralNPUV2DefaultExtmemStartAddress,
       .length = kCoralNPUV2DefaultExtmemLength}};
};

std::unique_ptr<CoralNPUV2State> CreateCoralNPUV2State(
    absl::string_view id, ::mpact::sim::riscv::RiscVXlen xlen,
    ::mpact::sim::util::MemoryInterface* /*absl_nonnull*/ memory,
    ::mpact::sim::util::AtomicMemoryOpInterface* /*absl_nullable*/ atomic_memory,
    const CoralNPUV2StateConfig* /*absl_nullable*/ config);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_V2_STATE_H_
