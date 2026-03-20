#include "liulin_y_complex_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "liulin_y_complex_ccs/common/include/common.hpp"

namespace liulin_y_complex_ccs {

LiulinYComplexCcsOmp::LiulinYComplexCcsOmp(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LiulinYComplexCcsOmp::ValidationImpl() {
  const auto &first = GetInput().first;
  const auto &second = GetInput().second;
  return first.count_cols == second.count_rows;
}

bool LiulinYComplexCcsOmp::PreProcessingImpl() {
  const auto &first = GetInput().first;
  const auto &second = GetInput().second;

  auto &result = GetOutput();
  result.count_rows = first.count_rows;
  result.count_cols = second.count_cols;
  result.values.clear();
  result.row_index.clear();
  result.col_index.assign(static_cast<size_t>(result.count_cols) + 1, 0);

  return true;
}

bool LiulinYComplexCcsOmp::RunImpl() {
  const auto &first = GetInput().first;
  const auto &second = GetInput().second;
  auto &result = GetOutput();

  const int num_rows = first.count_rows;
  const int num_cols = second.count_cols;

  std::vector<std::vector<std::complex<double>>> thread_values(num_cols);
  std::vector<std::vector<int>> thread_row_indices(num_cols);

#pragma omp parallel
  {
    std::vector<std::complex<double>> dense_col(num_rows, {0.0, 0.0});
    std::vector<int> active_rows;

    std::vector<int> marker(num_rows, -1);

#pragma omp for schedule(dynamic)
    for (int j = 0; j < num_cols; ++j) {
      for (int kb = second.col_index[j]; kb < second.col_index[j + 1]; ++kb) {
        int k = second.row_index[kb];
        std::complex<double> b_val = second.values[kb];

        for (int ka = first.col_index[k]; ka < first.col_index[k + 1]; ++ka) {
          int i = first.row_index[ka];

          if (marker[i] != j) {
            marker[i] = j;
            active_rows.push_back(i);
            dense_col[i] = first.values[ka] * b_val;
          } else {
            dense_col[i] += first.values[ka] * b_val;
          }
        }
      }

      std::ranges::sort(active_rows);

      for (int i : active_rows) {
        if (std::abs(dense_col[i]) > 1e-15) {
          thread_values[j].push_back(dense_col[i]);
          thread_row_indices[j].push_back(i);
        }
      }

      active_rows.clear();
    }
  }

  result.col_index[0] = 0;
  int total_nnz = 0;

  for (int j = 0; j < num_cols; ++j) {
    total_nnz += static_cast<int>(thread_row_indices[j].size());
    result.col_index[static_cast<size_t>(j) + 1] = total_nnz;
  }

  result.values.reserve(total_nnz);
  result.row_index.reserve(total_nnz);

  for (int j = 0; j < num_cols; ++j) {
    result.values.insert(result.values.end(), thread_values[j].begin(), thread_values[j].end());
    result.row_index.insert(result.row_index.end(), thread_row_indices[j].begin(), thread_row_indices[j].end());
  }

  return true;
}

bool LiulinYComplexCcsOmp::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_complex_ccs
