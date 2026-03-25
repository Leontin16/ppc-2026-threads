#pragma once

#include <tbb/tbb.h>

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"
#include "timur_a_cannon/common/include/common.hpp"

namespace timur_a_cannon {

class TimurACannonMatrixMultiplicationTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }

  TimurACannonMatrixMultiplicationTBB() = default;
  explicit TimurACannonMatrixMultiplicationTBB(const InType &in);
  ~TimurACannonMatrixMultiplicationTBB() override = default;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RotateBlocksA(std::vector<std::vector<Matrix>> &blocks, int grid_sz);
  static void RotateBlocksB(std::vector<std::vector<Matrix>> &blocks, int grid_sz);
  static void BlockMultiplyAccumulate(const Matrix &a, const Matrix &b, Matrix &c, int b_size);

  static void DistributeData(const Matrix &src_a, const Matrix &src_b, std::vector<std::vector<Matrix>> &bl_a,
                             std::vector<std::vector<Matrix>> &bl_b, int b_size, int grid_sz);

  static void CollectResult(const std::vector<std::vector<Matrix>> &bl_c, Matrix &res, int b_size, int grid_sz);
};

}  // namespace timur_a_cannon
