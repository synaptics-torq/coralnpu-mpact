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

#include "sim/coralnpu_v2_instructions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sim/coralnpu_v2_state.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "riscv/riscv_f_instructions.h"
#include "riscv/riscv_i_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_memory_instructions.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

/* start - new test macros */
#define ABSL_EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::absl_testing::IsOk())
#define ABSL_ASSERT_OK(expression) \
  ASSERT_THAT(expression, ::absl_testing::IsOk())
/* end - new test macros */

/* start - helper test macro */
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  auto status_or = (rexpr); \
  ABSL_ASSERT_OK(status_or); \
  lhs = std::move(status_or).value();
/* end - helper test macro */

namespace {
using ::coralnpu::sim::CoralNPUV2Fsw;
using ::coralnpu::sim::CoralNPUV2Jal;
using ::coralnpu::sim::CoralNPUV2Jalr;
using ::coralnpu::sim::CoralNPUV2Lb;
using ::coralnpu::sim::CoralNPUV2Lbu;
using ::coralnpu::sim::CoralNPUV2Lh;
using ::coralnpu::sim::CoralNPUV2Lhu;
using ::coralnpu::sim::CoralNPUV2Lw;
using ::coralnpu::sim::CoralNPUV2Mpause;
using ::coralnpu::sim::CoralNPUV2Sb;
using ::coralnpu::sim::CoralNPUV2Sh;
using ::coralnpu::sim::CoralNPUV2State;
using ::coralnpu::sim::CoralNPUV2Sw;
using ::coralnpu::sim::MemoryPermission;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::RegisterBase;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::riscv::kFRegisterAliases;
using ::mpact::sim::riscv::kXRegisterAliases;
using ::mpact::sim::riscv::RiscVIFlwChild;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32VectorDestinationOperand;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::VlSegmentChild;
using ::mpact::sim::riscv::RV32::RiscVILbChild;
using ::mpact::sim::riscv::RV32::RiscVILbuChild;
using ::mpact::sim::riscv::RV32::RiscVILhChild;
using ::mpact::sim::riscv::RV32::RiscVILhuChild;
using ::mpact::sim::riscv::RV32::RiscVILwChild;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

constexpr uint32_t kGoodLsuAddress = 0x00014000;
constexpr uint32_t kBadLsuAddress = 0x100000;
constexpr uint32_t kTestWord = 0xf0f0a5a5;
constexpr uint16_t kTestHalfWord = 0xa5a5;
constexpr uint8_t kTestByte = 0xa5;
constexpr uint64_t kNanBoxedTestWord =
    0xffffffff'00000000 | static_cast<uint64_t>(kTestWord);
constexpr uint32_t kLsuAccessStartAddress = 0x00010000;
constexpr uint32_t kLsuAccessLength = 0x8000;
constexpr uint32_t kItcmStartAddress = 0x00000000;
constexpr uint32_t kItcmLength = 0x2000;

class CoralNPUV2InstructionTest : public ::testing::Test {
 public:
  using SemanticFunction = std::function<void(Instruction*)>;

 protected:
  void SetUp() override {
    memory_ = std::make_unique<FlatDemandMemory>();
    state_ = std::make_unique<CoralNPUV2State>("CoralNPUV2", RiscVXlen::RV32,
                                               memory_.get());
    rv_vector_ = std::make_unique<RiscVVectorState>(
        state_.get(),
        /*byte_length=*/coralnpu::sim::kCoralNPUV2VectorByteLength);
    state_->set_rv_vector(rv_vector_.get());
    state_->AddMemoryRegion(kLsuAccessStartAddress, kLsuAccessLength,
                            MemoryPermission::kReadWrite);
    state_->AddMemoryRegion(kItcmStartAddress, kItcmLength,
                            MemoryPermission::kReadExecute);

    state_->AddMpauseHandler([this](const Instruction*) -> bool {
      was_mpause_handler_called_ = true;
      return true;
    });

    state_->set_on_trap([this](bool, uint64_t trap_value,
                               uint64_t exception_code, uint64_t epc,
                               const Instruction*) -> bool {
      was_trap_handler_called_ = true;
      exception_code_ = static_cast<ExceptionCode>(exception_code);
      trap_value_ = trap_value;
      epc_ = epc;
      return false;
    });

    // Add the scalar, floating point and vector registers to the state.
    std::string reg_name;
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat("x", i);
      x_regs_.push_back(new RV32Register(state_.get(), reg_name));
      state_->AddRegister(reg_name, x_regs_.back());
      ABSL_ASSERT_OK(state_->AddRegisterAlias<RV32Register>(reg_name,
                                                       kXRegisterAliases[i]));

      reg_name = absl::StrCat("f", i);
      f_regs_.push_back(new RVFpRegister(state_.get(), reg_name));
      state_->AddRegister(reg_name, f_regs_.back());
      ABSL_ASSERT_OK(state_->AddRegisterAlias<RVFpRegister>(reg_name,
                                                       kFRegisterAliases[i]));

      reg_name = absl::StrCat("v", i);
      v_regs_.push_back(new RVVectorRegister(
          state_.get(), reg_name,
          /*width=*/coralnpu::sim::kCoralNPUV2VectorByteLength));
      state_->AddRegister(reg_name, v_regs_.back());
    }
    pc_reg_ = std::make_unique<RV32Register>(state_.get(), "pc");
  }

  void AttachLoadChildInstruction(Instruction*, SemanticFunction);
  template <typename T>
  void SetMemoryContents(uint32_t address, T data);

  template <typename T>
  void SetMemoryContentsFromVector(uint32_t address,
                                   const std::vector<T>& test_data);

  void AttachSegmentLoadChildInstruction(Instruction* parent, int nf_value,
                                         int v_dest_start, int element_width);
  std::unique_ptr<Instruction> CreateLoadInstruction(
      SemanticFunction parent_semantic_function,
      SemanticFunction child_semantic_function);
  std::unique_ptr<Instruction> CreateStoreInstruction(
      SemanticFunction parent_semantic_function);
  std::unique_ptr<Instruction> CreateFloatLoadInstruction();
  std::unique_ptr<Instruction> CreateFloatStoreInstruction();
  std::unique_ptr<Instruction> CreateJalInstruction(int32_t offset);
  std::unique_ptr<Instruction> CreateJalrInstruction(int32_t offset);

  uint32_t GetXRegValue(RV32Register* reg) {
    return reg->data_buffer()->Get<uint32_t>(/*index=*/0);
  }

  void SetXRegValue(int reg_idx, uint32_t value) {
    x_regs_[reg_idx]->data_buffer()->Set<uint32_t>(/*index=*/0, value);
  }

  template <typename T>
  void SetVRegData(int reg_idx, const std::vector<T>& data) {
    DataBuffer* db = v_regs_[reg_idx]->data_buffer();
    int limit = std::min<int>(data.size(), db->size<T>());
    for (int i = 0; i < limit; i++) {
      db->Set<T>(i, data[i]);
    }
  }

  void SetupVectorState(uint32_t vtype, int vl) {
    rv_vector_->SetVectorType(vtype);
    rv_vector_->set_vector_length(vl);
  }

  template <typename T>
  T GetMemoryContents(uint32_t address) {
    DataBuffer* mem_db = state_->db_factory()->Allocate<T>(/*size=*/1);
    memory_->Load(address, mem_db, /*inst=*/nullptr, /*context=*/nullptr);
    T mem_value = mem_db->Get<T>(/*index=*/0);
    mem_db->DecRef();
    return mem_value;
  }

  std::unique_ptr<MemoryInterface> memory_;
  std::unique_ptr<CoralNPUV2State> state_;
  std::unique_ptr<RiscVVectorState> rv_vector_;
  // Registers are owned by the state_.
  std::vector<RV32Register*> x_regs_;
  std::vector<RVFpRegister*> f_regs_;
  std::vector<RVVectorRegister*> v_regs_;
  std::unique_ptr<RV32Register> pc_reg_;
  std::unique_ptr<Instruction> child_inst_;
  bool was_mpause_handler_called_ = false;
  bool was_trap_handler_called_ = false;
  uint64_t trap_value_ = 0;
  uint64_t epc_ = 0;
  ExceptionCode exception_code_ = ExceptionCode::kBreakpoint;
};

void CoralNPUV2InstructionTest::AttachLoadChildInstruction(
    Instruction* parent, SemanticFunction semantic_function) {
  child_inst_ = std::make_unique<Instruction>(parent->address(), state_.get());
  child_inst_->set_size(parent->size());
  child_inst_->set_semantic_function(semantic_function);
  parent->AppendChild(child_inst_.get());
}

template <typename T>
void CoralNPUV2InstructionTest::SetMemoryContents(uint32_t address, T data) {
  DataBuffer* db = state_->db_factory()->Allocate<T>(1);
  db->Set<T>(/*index=*/0, data);
  memory_->Store(address, db);
  db->DecRef();
}

template <typename T>
void CoralNPUV2InstructionTest::SetMemoryContentsFromVector(
    uint32_t address, const std::vector<T>& test_data) {
  DataBuffer* db = state_->db_factory()->Allocate<T>(test_data.size());
  for (int i = 0; i < test_data.size(); i++) {
    db->Set<T>(/*index=*/i, test_data[i]);
  }
  memory_->Store(address, db);
  db->DecRef();
}

void CoralNPUV2InstructionTest::AttachSegmentLoadChildInstruction(
    Instruction* parent, int nf_value, int v_dest_start, int element_width) {
  AttachLoadChildInstruction(parent,
                             absl::bind_front(&VlSegmentChild, element_width));
  child_inst_->AppendSource(new ImmediateOperand<int32_t>(nf_value));
  int num_regs = nf_value + 1;
  std::vector<RegisterBase*> dest_regs;
  std::string reg_name = absl::StrCat("v", v_dest_start);
  if (num_regs > 1) {
    absl::StrAppend(&reg_name, "-v", v_dest_start + num_regs - 1);
  }
  for (int i = 0; i < num_regs; i++) {
    dest_regs.push_back(v_regs_[v_dest_start + i]);
  }
  child_inst_->AppendDestination(new RV32VectorDestinationOperand(
      absl::MakeSpan(dest_regs), /*latency=*/0, reg_name));
}

std::unique_ptr<Instruction> CoralNPUV2InstructionTest::CreateLoadInstruction(
    SemanticFunction parent_semantic_function,
    SemanticFunction child_semantic_function) {
  // Source operand 1: x register containing the base address to load.
  // Source operand 2: immediate operand containing the offset to load.
  // Destination operand 1: x register to store the loaded data.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(parent_semantic_function);
  inst->AppendSource(x_regs_[1]->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  AttachLoadChildInstruction(inst.get(), child_semantic_function);
  child_inst_->AppendDestination(
      x_regs_[2]->CreateDestinationOperand(/*latency=*/0));
  return inst;
}

std::unique_ptr<Instruction> CoralNPUV2InstructionTest::CreateStoreInstruction(
    SemanticFunction parent_semantic_function) {
  // Source operand 1: x register containing the base address to store.
  // Source operand 2: immediate operand containing the offset to store.
  // Source operand 3: x register containing the data to store.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(parent_semantic_function);
  inst->AppendSource(x_regs_[1]->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  inst->AppendSource(x_regs_[2]->CreateSourceOperand());
  return inst;
}

std::unique_ptr<Instruction>
CoralNPUV2InstructionTest::CreateFloatLoadInstruction() {
  // Source operand 1: x register containing the base address to load.
  // Source operand 2: immediate operand containing the offset to load.
  // Destination operand 1: fp register to store the loaded data.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(CoralNPUV2Lw);
  inst->AppendSource(x_regs_[1]->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  AttachLoadChildInstruction(inst.get(), RiscVIFlwChild);
  child_inst_->AppendDestination(
      f_regs_[0]->CreateDestinationOperand(/*latency=*/0));
  return inst;
}

std::unique_ptr<Instruction>
CoralNPUV2InstructionTest::CreateFloatStoreInstruction() {
  // Source operand 1: x register containing the base address to store.
  // Source operand 2: immediate operand containing the offset to store.
  // Source operand 3: fp register containing the data to store.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(CoralNPUV2Fsw);
  inst->AppendSource(x_regs_[1]->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  inst->AppendSource(f_regs_[0]->CreateSourceOperand());
  return inst;
}

std::unique_ptr<Instruction> CoralNPUV2InstructionTest::CreateJalInstruction(
    int32_t offset) {
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(CoralNPUV2Jal);
  inst->AppendSource(new ImmediateOperand<int32_t>(offset));
  inst->AppendDestination(pc_reg_->CreateDestinationOperand(/*latency=*/0));
  inst->AppendDestination(x_regs_[1]->CreateDestinationOperand(/*latency=*/0));
  return inst;
}

std::unique_ptr<Instruction> CoralNPUV2InstructionTest::CreateJalrInstruction(
    int32_t offset) {
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(CoralNPUV2Jalr);
  inst->AppendSource(x_regs_[2]->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(offset));
  inst->AppendDestination(pc_reg_->CreateDestinationOperand(/*latency=*/0));
  inst->AppendDestination(x_regs_[1]->CreateDestinationOperand(/*latency=*/0));
  return inst;
}

TEST_F(CoralNPUV2InstructionTest, MPause_ExecutesHandler) {
  // Create a test instruction and execute it.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(CoralNPUV2Mpause);
  inst->Execute(/*context=*/nullptr);

  // Verify that the mpause test handler was called.
  EXPECT_TRUE(was_mpause_handler_called_);
}

TEST_F(CoralNPUV2InstructionTest, Lw_ValidAddress_Succeeds) {
  SetMemoryContents(kGoodLsuAddress, kTestWord);
  SetXRegValue(1, kGoodLsuAddress);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register contains the test data from memory.
  EXPECT_EQ(GetXRegValue(x_regs_[2]), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Lw_InvalidAddress_Traps) {
  SetMemoryContents(kBadLsuAddress, kTestWord);
  SetXRegValue(1, kBadLsuAddress);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the destination register does not contain the test data.
  EXPECT_NE(GetXRegValue(x_regs_[2]), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Sw_ValidAddress_Succeeds) {
  SetXRegValue(1, kGoodLsuAddress);
  SetXRegValue(2, kTestWord);

  // Create a test store word (sw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sw);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the memory contents were updated with the test data.
  EXPECT_EQ(GetMemoryContents<uint32_t>(kGoodLsuAddress), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Sw_InvalidAddress_Traps) {
  SetXRegValue(1, kBadLsuAddress);
  SetXRegValue(2, kTestWord);

  // Create a test store word (sw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sw);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the memory contents were not updated since the access was
  // invalid.
  EXPECT_NE(GetMemoryContents<uint32_t>(kGoodLsuAddress), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Lh_ValidAddress_Succeeds) {
  SetMemoryContents(kGoodLsuAddress, kTestHalfWord);
  SetXRegValue(1, kGoodLsuAddress);

  // Create a test load half word (lh) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lh, RiscVILhChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value =
      (kTestHalfWord & 0x8000) ? 0xffff'0000 | kTestHalfWord : kTestHalfWord;
  EXPECT_EQ(GetXRegValue(x_regs_[2]), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, Lh_InvalidAddress_Traps) {
  SetMemoryContents(kBadLsuAddress, kTestHalfWord);
  SetXRegValue(1, kBadLsuAddress);

  // Create a test load half word (lh) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lh, RiscVILhChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the destination register does not contain the test data.
  uint32_t unwanted_register_value =
      (kTestHalfWord & 0x8000) ? 0xffff'0000 | kTestHalfWord : kTestHalfWord;
  EXPECT_NE(GetXRegValue(x_regs_[2]), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, Sh_ValidAddress_Succeeds) {
  SetXRegValue(1, kGoodLsuAddress);
  SetXRegValue(2, kTestHalfWord);

  // Create a test store half word (sh) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sh);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the memory contents were updated with the register contents.
  EXPECT_EQ(GetMemoryContents<uint16_t>(kGoodLsuAddress), kTestHalfWord);
}

TEST_F(CoralNPUV2InstructionTest, Sh_InvalidAddress_Traps) {
  SetXRegValue(1, kBadLsuAddress);
  SetXRegValue(2, kTestHalfWord);

  // Create a test store half word (sh) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sh);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the memory contents were not updated with the register contents
  // since the access is invalid.
  EXPECT_NE(GetMemoryContents<uint16_t>(kBadLsuAddress), kTestHalfWord);
}

TEST_F(CoralNPUV2InstructionTest, Lhu_ValidAddress_Succeeds) {
  SetMemoryContents(kGoodLsuAddress, kTestHalfWord);
  SetXRegValue(1, kGoodLsuAddress);

  // Create a test load unsigned half word (lhu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lhu, RiscVILhuChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value = static_cast<uint32_t>(kTestHalfWord);
  EXPECT_EQ(GetXRegValue(x_regs_[2]), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, Lhu_InvalidAddress_Traps) {
  SetMemoryContents(kBadLsuAddress, kTestHalfWord);
  SetXRegValue(1, kBadLsuAddress);

  // Create a test load unsigned half word (lhu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lhu, RiscVILhuChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the destination register does not contain the test data since
  // the access is invalid.
  uint32_t unwanted_register_value = static_cast<uint32_t>(kTestHalfWord);
  EXPECT_NE(GetXRegValue(x_regs_[2]), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, Lb_ValidAddress_Succeeds) {
  SetMemoryContents(kGoodLsuAddress, kTestByte);
  SetXRegValue(1, kGoodLsuAddress);

  // Create a test load byte (lb) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lb, RiscVILbChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value =
      (kTestByte & 0x80) ? 0xffff'ff00 | kTestByte : kTestByte;
  EXPECT_EQ(GetXRegValue(x_regs_[2]), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, Lb_InvalidAddress_Traps) {
  SetMemoryContents(kBadLsuAddress, kTestByte);
  SetXRegValue(1, kBadLsuAddress);

  // Create a test load byte (lb) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lb, RiscVILbChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the destination register does not contain the test data.
  uint32_t unwanted_register_value =
      (kTestByte & 0x80) ? 0xffff'ff00 | kTestByte : kTestByte;
  EXPECT_NE(GetXRegValue(x_regs_[2]), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, Sb_ValidAddress_Succeeds) {
  SetXRegValue(1, kGoodLsuAddress);
  SetXRegValue(2, kTestByte);

  // Create a test store byte (sb) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sb);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the memory contents were updated with the register contents.
  EXPECT_EQ(GetMemoryContents<uint8_t>(kGoodLsuAddress), kTestByte);
}

TEST_F(CoralNPUV2InstructionTest, Sb_InvalidAddress_Traps) {
  SetXRegValue(1, kBadLsuAddress);
  SetXRegValue(2, kTestByte);

  // Create a test store byte (sb) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sb);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the memory contents were not updated with the register contents
  // since the access was invalid.
  EXPECT_NE(GetMemoryContents<uint8_t>(kBadLsuAddress), kTestByte);
}

TEST_F(CoralNPUV2InstructionTest, Lbu_ValidAddress_Succeeds) {
  SetMemoryContents(kGoodLsuAddress, kTestByte);
  SetXRegValue(1, kGoodLsuAddress);

  // Create a test load unsigned byte (lbu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lbu, RiscVILbuChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value = static_cast<uint32_t>(kTestByte);
  EXPECT_EQ(GetXRegValue(x_regs_[2]), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, Lbu_InvalidAddress_Traps) {
  SetMemoryContents(kBadLsuAddress, kTestByte);
  SetXRegValue(1, kBadLsuAddress);

  // Create a test load unsigned byte (lbu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lbu, RiscVILbuChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the destination register does not contain the test data.
  uint32_t unwanted_register_value = static_cast<uint32_t>(kTestByte);
  EXPECT_NE(GetXRegValue(x_regs_[2]), unwanted_register_value);
}

// Test with an address where the access spans across the valid LSU range
// boundary.
TEST_F(CoralNPUV2InstructionTest, Lw_AddressBeforeValidRange_Traps) {
  const uint32_t test_address = kLsuAccessStartAddress - 1;
  SetMemoryContents(test_address, kTestWord);
  SetXRegValue(1, test_address);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, test_address);

  // Verify that the destination register does not contain the test data.
  EXPECT_NE(GetXRegValue(x_regs_[2]), kTestWord);
}

// Test with an address where the access spans across the valid LSU range
// boundary.
TEST_F(CoralNPUV2InstructionTest, Lw_AddressSpanningEndBoundary_Traps) {
  const uint32_t test_address = kLsuAccessStartAddress + kLsuAccessLength - 3;
  SetMemoryContents(test_address, kTestWord);
  SetXRegValue(1, test_address);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, test_address);

  // Verify that the destination register does not contain the test data.
  EXPECT_NE(GetXRegValue(x_regs_[2]), kTestWord);
}

// Test load word at the end of valid LSU range.
TEST_F(CoralNPUV2InstructionTest, Lw_AddressAtEndOfValidRange_Succeeds) {
  const uint32_t test_address = kLsuAccessStartAddress + kLsuAccessLength - 4;
  SetMemoryContents(test_address, kTestWord);
  SetXRegValue(1, test_address);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register does contain the test data.
  EXPECT_EQ(GetXRegValue(x_regs_[2]), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Flw_ValidAddress_Succeeds) {
  SetMemoryContents(kGoodLsuAddress, kTestWord);
  SetXRegValue(1, kGoodLsuAddress);

  // Create a test float load word (flw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateFloatLoadInstruction();
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the destination register contains the test data from memory.
  uint64_t fp_word = f_regs_[0]->data_buffer()->Get<uint64_t>(0);
  EXPECT_EQ(static_cast<uint32_t>(fp_word), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Flw_InvalidAddress_Traps) {
  SetMemoryContents(kBadLsuAddress, kTestWord);
  SetXRegValue(1, kBadLsuAddress);

  // Create a test float load word (flw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateFloatLoadInstruction();
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the destination register does not contain the test data.
  uint64_t fp_word = f_regs_[0]->data_buffer()->Get<uint64_t>(0);
  EXPECT_NE(static_cast<uint32_t>(fp_word), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Fsw_ValidAddress_Succeeds) {
  SetXRegValue(1, kGoodLsuAddress);
  f_regs_[0]->data_buffer()->Set<uint64_t>(/*index=*/0, kNanBoxedTestWord);

  // Create a test float store word (fsw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateFloatStoreInstruction();
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);

  // Verify that the memory contents were updated with the test data.
  EXPECT_EQ(GetMemoryContents<uint32_t>(kGoodLsuAddress), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Fsw_InvalidAddress_Traps) {
  SetXRegValue(1, kBadLsuAddress);
  f_regs_[0]->data_buffer()->Set<uint64_t>(/*index=*/0, kNanBoxedTestWord);

  // Create a test float store word (fsw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateFloatStoreInstruction();
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kBadLsuAddress);

  // Verify that the memory contents were not updated since the access was
  // invalid.
  EXPECT_NE(GetMemoryContents<uint32_t>(kGoodLsuAddress), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, Jal_AlignedTarget_Succeeds) {
  int32_t offset = 4;
  std::unique_ptr<Instruction> inst = CreateJalInstruction(offset);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);
}

TEST_F(CoralNPUV2InstructionTest, Jal_InvalidTarget_Traps) {
  int32_t offset = kItcmLength;
  std::unique_ptr<Instruction> inst = CreateJalInstruction(offset);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kItcmLength);

  EXPECT_EQ(exception_code_, ExceptionCode::kInstructionAccessFault);
  // The destination register should NOT be updated if trap is taken for Jal.
  EXPECT_EQ(GetXRegValue(x_regs_[1]), 0);
}

TEST_F(CoralNPUV2InstructionTest, Jalr_AlignedTarget_Succeeds) {
  int32_t offset = 0;
  SetXRegValue(2, 4);
  std::unique_ptr<Instruction> inst = CreateJalrInstruction(offset);
  inst->Execute(/*context=*/nullptr);
  EXPECT_FALSE(was_trap_handler_called_);
}

TEST_F(CoralNPUV2InstructionTest, Jalr_InvalidTarget_Traps) {
  int32_t offset = 0;
  SetXRegValue(2, kItcmLength);
  std::unique_ptr<Instruction> inst = CreateJalrInstruction(offset);
  inst->Execute(/*context=*/nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  EXPECT_EQ(trap_value_, kItcmLength);

  EXPECT_EQ(exception_code_, ExceptionCode::kInstructionAccessFault);
  // The destination register should be updated even if trap is taken.
  EXPECT_EQ(GetXRegValue(x_regs_[1]), inst->address() + inst->size());
}

}  // namespace
