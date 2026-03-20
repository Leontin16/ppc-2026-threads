#include "tasks/gusev_d_double_sort_even_odd_batcher_omp/omp/include/ops_omp.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads {
namespace {

constexpr int kRadixPasses = 8;
constexpr int kBitsPerByte = 8;
constexpr size_t kRadixBuckets = 256;

uint64_t DoubleToSortableKey(ValueType value) {
  const auto bits = std::bit_cast<uint64_t>(value);
  const auto sign_mask = uint64_t{1} << 63;
  return (bits & sign_mask) == 0 ? bits ^ sign_mask : ~bits;
}

void RadixSortDoubles(std::vector<ValueType>& data) {
  if (data.size() < 2) {
    return;
  }

  std::vector<ValueType> buffer(data.size());
  auto* src = &data;
  auto* dst = &buffer;

  for (int byte = 0; byte < kRadixPasses; ++byte) {
    std::array<size_t, kRadixBuckets> count{};
    const auto shift = byte * kBitsPerByte;

    for (ValueType value : *src) {
      const auto bucket = static_cast<uint8_t>((DoubleToSortableKey(value) >> shift) & 0xFFULL);
      count.at(bucket)++;
    }

    size_t prefix = 0;
    for (auto& value : count) {
      const auto current = value;
      value = prefix;
      prefix += current;
    }

    for (ValueType value : *src) {
      const auto bucket = static_cast<uint8_t>((DoubleToSortableKey(value) >> shift) & 0xFFULL);
      (*dst)[count.at(bucket)++] = value;
    }

    std::swap(src, dst);
  }

  if (src != &data) {
    data = std::move(*src);
  }
}

void SplitByGlobalParity(const std::vector<ValueType>& source, size_t global_offset, std::vector<ValueType>& even,
                         std::vector<ValueType>& odd) {
  even.clear();
  odd.clear();
  even.reserve((source.size() + 1) / 2);
  odd.reserve(source.size() / 2);

  for (size_t i = 0; i < source.size(); ++i) {
    if (((global_offset + i) & 1U) == 0U) {
      even.push_back(source[i]);
    } else {
      odd.push_back(source[i]);
    }
  }
}

std::vector<ValueType> InterleaveParityGroups(size_t total_size, const std::vector<ValueType>& even,
                                              const std::vector<ValueType>& odd) {
  std::vector<ValueType> result(total_size);
  size_t even_index = 0;
  size_t odd_index = 0;

  for (size_t i = 0; i < total_size; ++i) {
    if ((i & 1U) == 0U) {
      result[i] = even[even_index++];
    } else {
      result[i] = odd[odd_index++];
    }
  }

  return result;
}

void OddEvenFinalize(std::vector<ValueType>& result) {
  for (size_t phase = 0; phase < result.size(); ++phase) {
    const auto start = phase & 1U;
    for (size_t i = start; i + 1 < result.size(); i += 2) {
      if (result[i] > result[i + 1]) {
        std::swap(result[i], result[i + 1]);
      }
    }
  }
}

std::vector<ValueType> MergeBatcherEvenOdd(const std::vector<ValueType>& left, const std::vector<ValueType>& right) {
  std::vector<ValueType> left_even;
  std::vector<ValueType> left_odd;
  std::vector<ValueType> right_even;
  std::vector<ValueType> right_odd;

  SplitByGlobalParity(left, 0, left_even, left_odd);
  SplitByGlobalParity(right, left.size(), right_even, right_odd);

  std::vector<ValueType> merged_even;
  std::vector<ValueType> merged_odd;
  merged_even.reserve(left_even.size() + right_even.size());
  merged_odd.reserve(left_odd.size() + right_odd.size());

  std::ranges::merge(left_even, right_even, std::back_inserter(merged_even));
  std::ranges::merge(left_odd, right_odd, std::back_inserter(merged_odd));

  auto result = InterleaveParityGroups(left.size() + right.size(), merged_even, merged_odd);
  OddEvenFinalize(result);
  return result;
}

size_t GetBlockCount(size_t input_size) {
  if (input_size == 0) {
    return 0;
  }

  const auto omp_threads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
  return std::max<size_t>(1, std::min(input_size, omp_threads));
}

std::vector<std::vector<ValueType>> MakeSortedBlocks(const std::vector<ValueType>& input) {
  const auto block_count = GetBlockCount(input.size());

  std::vector<std::vector<ValueType>> blocks(block_count);

#pragma omp parallel for schedule(static) if(block_count > 1)
  for (long long block = 0; block < static_cast<long long>(block_count); ++block) {
    const auto index = static_cast<size_t>(block);
    const auto begin = (index * input.size()) / block_count;
    const auto end = ((index + 1) * input.size()) / block_count;

    blocks[index].assign(input.begin() + static_cast<std::ptrdiff_t>(begin), input.begin() + static_cast<std::ptrdiff_t>(end));
    RadixSortDoubles(blocks[index]);
  }

  return blocks;
}

std::vector<ValueType> MergeBlocks(std::vector<std::vector<ValueType>> blocks) {
  if (blocks.empty()) {
    return {};
  }

  while (blocks.size() > 1) {
    const auto pair_count = blocks.size() / 2;
    std::vector<std::vector<ValueType>> next((blocks.size() + 1) / 2);

#pragma omp parallel for schedule(static) if(pair_count > 1)
    for (long long pair = 0; pair < static_cast<long long>(pair_count); ++pair) {
      const auto index = static_cast<size_t>(pair);
      next[index] = MergeBatcherEvenOdd(blocks[index * 2], blocks[(index * 2) + 1]);
    }

    if ((blocks.size() & 1U) != 0U) {
      next.back() = std::move(blocks.back());
    }

    blocks = std::move(next);
  }

  return blocks.empty() ? std::vector<double>{} : std::move(blocks.front());
}

}  // namespace

DoubleSortEvenOddBatcherOMP::DoubleSortEvenOddBatcherOMP(const InType& in) : BaseTask(in) {
  internal_order_test_ = true;
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool DoubleSortEvenOddBatcherOMP::ValidationImpl() { return GetOutput().empty(); }

bool DoubleSortEvenOddBatcherOMP::PreProcessingImpl() {
  input_data_ = GetInput();
  result_data_.clear();
  return true;
}

bool DoubleSortEvenOddBatcherOMP::RunImpl() {
  if (input_data_.empty()) {
    result_data_.clear();
    return true;
  }

  auto blocks = MakeSortedBlocks(input_data_);
  result_data_ = MergeBlocks(std::move(blocks));
  return true;
}

bool DoubleSortEvenOddBatcherOMP::PostProcessingImpl() {
  SetOutput(result_data_);
  return true;
}

}  // namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads
