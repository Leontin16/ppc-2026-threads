#include "gasenin_l_djstra/stl/include/ops_stl.hpp"

#include <algorithm>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

#include "gasenin_l_djstra/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_djstra {

namespace {

void ThreadFindMin(int thread_id, int num_threads, InType n, InType inf, const std::vector<InType> &dist,
                   const std::vector<char> &visited, std::vector<InType> &local_min, std::vector<InType> &local_vert) {
  InType t_min = inf;
  InType t_vert = -1;
  for (int idx = thread_id; idx < n; idx += num_threads) {
    if (visited[idx] == 0 && dist[idx] < t_min) {
      t_min = dist[idx];
      t_vert = idx;
    }
  }
  local_min[thread_id] = t_min;
  local_vert[thread_id] = t_vert;
}

void ThreadRelaxEdges(int thread_id, int num_threads, InType n, InType inf, InType global_vertex,
                      std::vector<InType> &dist, const std::vector<char> &visited) {
  for (int vertex = thread_id; vertex < n; vertex += num_threads) {
    if (visited[vertex] == 0 && vertex != global_vertex && dist[global_vertex] != inf) {
      dist[vertex] = std::min(dist[vertex], dist[global_vertex] + std::abs(global_vertex - vertex));
    }
  }
}

InType ReduceAndMark(int num_threads, InType inf, std::vector<InType> &local_min, std::vector<InType> &local_vert,
                     std::vector<char> &visited) {
  InType global_min = inf;
  InType global_vertex = -1;
  for (int ti = 0; ti < num_threads; ++ti) {
    if (local_min[ti] < global_min) {
      global_min = local_min[ti];
      global_vertex = local_vert[ti];
    }
  }
  if (global_vertex != -1 && global_min != inf) {
    visited[global_vertex] = 1;
  } else {
    global_vertex = -1;
  }
  return global_vertex;
}

}  // namespace

GaseninLDjstraSTL::GaseninLDjstraSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLDjstraSTL::ValidationImpl() {
  return GetInput() > 0;
}

bool GaseninLDjstraSTL::PreProcessingImpl() {
  const InType n = GetInput();
  const InType inf = std::numeric_limits<InType>::max();

  dist_.assign(n, inf);
  visited_.assign(n, 0);

  dist_[0] = 0;
  return true;
}

bool GaseninLDjstraSTL::RunImpl() {
  const InType n = GetInput();
  const InType inf = std::numeric_limits<InType>::max();
  const int num_threads = ppc::util::GetNumThreads();

  std::vector<InType> local_min(num_threads, inf);
  std::vector<InType> local_vert(num_threads, -1);
  InType global_vertex = -1;

  auto on_find_sync = [&]() noexcept {
    global_vertex = ReduceAndMark(num_threads, inf, local_min, local_vert, visited_);
    std::fill(local_min.begin(), local_min.end(), inf);
    std::fill(local_vert.begin(), local_vert.end(), -1);
  };

  std::barrier<decltype(on_find_sync)> bar_find(num_threads, std::move(on_find_sync));
  std::barrier<> bar_relax(num_threads);

  auto worker = [&](int thread_id) {
    for (int iter = 0; iter < n; ++iter) {
      ThreadFindMin(thread_id, num_threads, n, inf, dist_, visited_, local_min, local_vert);
      bar_find.arrive_and_wait();
      if (global_vertex == -1) {
        break;
      }
      ThreadRelaxEdges(thread_id, num_threads, n, inf, global_vertex, dist_, visited_);
      bar_relax.arrive_and_wait();
    }
  };

  std::vector<std::thread> threads(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads[thread_id] = std::thread(worker, thread_id);
  }
  for (auto &th : threads) {
    th.join();
  }

  int64_t total_sum = 0;
  for (int idx = 0; idx < n; ++idx) {
    if (dist_[idx] != inf) {
      total_sum += dist_[idx];
    }
  }
  GetOutput() = static_cast<OutType>(total_sum);
  return true;
}

bool GaseninLDjstraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_djstra
