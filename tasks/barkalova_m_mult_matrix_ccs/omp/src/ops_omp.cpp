#include "barkalova_m_mult_matrix_ccs/omp/include/ops_omp.hpp"

#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

#include "barkalova_m_mult_matrix_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_mult_matrix_ccs {

BarkalovaMMultMatrixCcsOMP::BarkalovaMMultMatrixCcsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCSMatrix{};
}

bool BarkalovaMMultMatrixCcsOMP::ValidationImpl() {
  const auto &[A, B] = GetInput();
  if (A.cols != B.rows) {
    return false;
  }
  if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
    return false;
  }
  if (A.col_ptrs.size() != static_cast<size_t>(A.cols) + 1 || B.col_ptrs.size() != static_cast<size_t>(B.cols) + 1) {
    return false;
  }
  if (A.col_ptrs.empty() || A.col_ptrs[0] != 0 || B.col_ptrs.empty() || B.col_ptrs[0] != 0) {
    return false;
  }
  if (std::cmp_not_equal(A.nnz, A.values.size()) || std::cmp_not_equal(B.nnz, B.values.size())) {
    return false;
  }
  return true;
}

bool BarkalovaMMultMatrixCcsOMP::PreProcessingImpl() {
  return true;
}

namespace {
constexpr double kEpsilon = 1e-10;

void TransponirMatr(const CCSMatrix &a, CCSMatrix &at) {
  at.rows = a.cols;
  at.cols = a.rows;
  at.nnz = a.nnz;

  if (a.nnz == 0) {
    at.values.clear();
    at.row_indices.clear();
    at.col_ptrs.assign(at.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(at.cols, 0);
  for (int i = 0; i < a.nnz; i++) {
    row_count[a.row_indices[i]]++;
  }

  at.col_ptrs.resize(at.cols + 1);
  at.col_ptrs[0] = 0;
  for (int i = 0; i < at.cols; i++) {
    at.col_ptrs[i + 1] = at.col_ptrs[i] + row_count[i];
  }

  at.values.resize(a.nnz);
  at.row_indices.resize(a.nnz);

  std::vector<int> current_pos(at.cols, 0);
  for (int col = 0; col < a.cols; col++) {
    for (int i = a.col_ptrs[col]; i < a.col_ptrs[col + 1]; i++) {
      int row = a.row_indices[i];
      Complex val = a.values[i];

      int pos = at.col_ptrs[row] + current_pos[row];
      at.values[pos] = val;
      at.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

Complex ComputeScalarProduct(const CCSMatrix &at, const CCSMatrix &b, int row_a, int col_b) {
  Complex sum = Complex(0.0, 0.0);

  int ks = at.col_ptrs[row_a];
  int ls = b.col_ptrs[col_b];
  int kf = at.col_ptrs[row_a + 1];
  int lf = b.col_ptrs[col_b + 1];

  while ((ks < kf) && (ls < lf)) {
    if (at.row_indices[ks] < b.row_indices[ls]) {
      ks++;
    } else if (at.row_indices[ks] > b.row_indices[ls]) {
      ls++;
    } else {
      sum += at.values[ks] * b.values[ls];
      ks++;
      ls++;
    }
  }

  return sum;
}

bool IsNonZero(const Complex &val) {
  return std::abs(val.real()) > kEpsilon || std::abs(val.imag()) > kEpsilon;
}

}  // namespace

bool BarkalovaMMultMatrixCcsOMP::RunImpl() {
  const auto &[a, b] = GetInput();

  try {
    CCSMatrix c;
    CCSMatrix at;
    TransponirMatr(a, at);

    c.rows = a.rows;
    c.cols = b.cols;

    int num_threads = omp_get_max_threads();
    std::vector<ThreadData> thread_data(num_threads);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      ThreadData &local = thread_data[tid];

      std::vector<Complex> local_values;
      std::vector<int> local_rows;
      std::vector<int> boundaries(c.cols + 1, 0);

#pragma omp for schedule(dynamic)
      for (int j = 0; j < c.cols; j++) {
        for (int i = 0; i < at.cols; i++) {
          Complex sum = ComputeScalarProduct(at, b, i, j);
          if (IsNonZero(sum)) {
            local_values.push_back(sum);
            local_rows.push_back(i);
          }
        }
        boundaries[j + 1] = local_values.size();
      }

      for (int j = 1; j <= c.cols; j++) {
        if (boundaries[j] == 0) {
          boundaries[j] = boundaries[j - 1];
        }
      }

      local.values = std::move(local_values);
      local.rows = std::move(local_rows);
      local.col_boundaries = std::move(boundaries);
    }

    std::vector<int> col_ptrs(c.cols + 1, 0);

    for (int t = 0; t < num_threads; t++) {
      const auto &boundaries = thread_data[t].col_boundaries;
      for (int j = 0; j < c.cols; j++) {
        col_ptrs[j + 1] += boundaries[j + 1] - boundaries[j];
      }
    }

    for (int j = 1; j <= c.cols; j++) {
      col_ptrs[j] += col_ptrs[j - 1];
    }

    int total_nnz = col_ptrs[c.cols];

    std::vector<Complex> values(total_nnz);
    std::vector<int> rows(total_nnz);

    std::vector<int> thread_pos(num_threads, 0);

    for (int j = 0; j < c.cols; j++) {
      int write_pos = col_ptrs[j];
      for (int t = 0; t < num_threads; t++) {
        int start_idx = thread_data[t].col_boundaries[j];
        int end_idx = thread_data[t].col_boundaries[j + 1];
        int count = end_idx - start_idx;
        if (count > 0) {
          if (start_idx + count <= static_cast<int>(thread_data[t].values.size()) &&
              start_idx + count <= static_cast<int>(thread_data[t].rows.size())) {
            std::copy(thread_data[t].values.begin() + start_idx, thread_data[t].values.begin() + end_idx,
                      values.begin() + write_pos);
            std::copy(thread_data[t].rows.begin() + start_idx, thread_data[t].rows.begin() + end_idx,
                      rows.begin() + write_pos);
            write_pos += count;
          }
        }
      }
    }

    c.values = std::move(values);
    c.row_indices = std::move(rows);
    c.col_ptrs = std::move(col_ptrs);
    c.nnz = total_nnz;

    GetOutput() = c;
    return true;

  } catch (const std::exception &) {
    return false;
  }
}

bool BarkalovaMMultMatrixCcsOMP::PostProcessingImpl() {
  const auto &c = GetOutput();
  return c.rows > 0 && c.cols > 0 && c.col_ptrs.size() == static_cast<size_t>(c.cols) + 1;
}

}  // namespace barkalova_m_mult_matrix_ccs
