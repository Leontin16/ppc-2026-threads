#pragma once

#include "barkalova_m_mult_matrix_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace barkalova_m_mult_matrix_ccs {

class BarkalovaMMultMatrixCcsOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit BarkalovaMMultMatrixCcsOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct ThreadData {
    std::vector<Complex> values;
    std::vector<int> rows;
    std::vector<int> col_boundaries;
  };
};

}  // namespace barkalova_m_mult_matrix_ccs
