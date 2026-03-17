#include "gasenin_l_djstra/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

#include "gasenin_l_djstra/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_djstra {

namespace {

InType FindMinVertexSTL(const InType n, const InType inf, const std::vector<InType> &dist, std::vector<char> &visited,
                        const int num_threads) {
  std::vector<InType> local_min(num_threads, inf);
  std::vector<InType> local_vertex(num_threads, -1);

  std::vector<std::thread> threads(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([&, t]() {
      InType t_min = inf;
      InType t_vertex = -1;

      for (int i = t; i < n; i += num_threads) {
        if (visited[i] == 0 && dist[i] < t_min) {
          t_min = dist[i];
          t_vertex = i;
        }
      }

      local_min[t] = t_min;
      local_vertex[t] = t_vertex;
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  InType global_min = inf;
  InType global_vertex = -1;

  for (int t = 0; t < num_threads; ++t) {
    if (local_min[t] < global_min) {
      global_min = local_min[t];
      global_vertex = local_vertex[t];
    }
  }

  if (global_vertex != -1 && global_min != inf) {
    visited[global_vertex] = 1;
  } else {
    global_vertex = -1;
  }

  return global_vertex;
}

void RelaxEdgesSTL(const InType n, const InType inf, const InType u, std::vector<InType> &dist,
                   const std::vector<char> &visited, const int num_threads) {
  std::vector<std::thread> threads(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([&, t]() {
      for (int v = t; v < n; v += num_threads) {
        if (visited[v] == 0 && v != u) {
          const InType weight = std::abs(u - v);

          if (dist[u] != inf) {
            const InType new_dist = dist[u] + weight;
            if (new_dist < dist[v]) {
              dist[v] = new_dist;
            }
          }
        }
      }
    });
  }

  for (auto &th : threads) {
    th.join();
  }
}

int64_t CalculateTotalSumSTL(const InType n, const InType inf, const std::vector<InType> &dist, const int num_threads) {
  std::vector<int64_t> partial(num_threads, 0);
  std::vector<std::thread> threads(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([&, t]() {
      int64_t s = 0;
      for (int i = t; i < n; i += num_threads) {
        if (dist[i] != inf) {
          s += dist[i];
        }
      }
      partial[t] = s;
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  int64_t total = 0;
  for (const int64_t part : partial) {
    total += part;
  }
  return total;
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

  for (int iteration = 0; iteration < n; ++iteration) {
    const InType u = FindMinVertexSTL(n, inf, dist_, visited_, num_threads);

    if (u == -1) {
      break;
    }

    RelaxEdgesSTL(n, inf, u, dist_, visited_, num_threads);
  }

  const int64_t total_sum = CalculateTotalSumSTL(n, inf, dist_, num_threads);

  GetOutput() = static_cast<OutType>(total_sum);
  return true;
}

bool GaseninLDjstraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_djstra
