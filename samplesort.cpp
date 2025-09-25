// samplesort_ppp.cpp
// Point-to-point sample sort (no collective MPI calls).
// Compile: mpic++ samplesort_ppp.cpp -O2 -o samplesort_ppp
// Run: mpirun --hostfile hostfile.txt -np <P> ./samplesort_ppp

#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>

bool isLocalSorted(const std::vector<long long> &arr) {
    for (size_t i = 1; i < arr.size(); ++i)
        if (arr[i-1] > arr[i]) return false;
    return true;
}

bool verifyGlobalSort(const std::vector<long long> &local, int rank, int size, MPI_Comm comm) {
    // 1) Local sortedness -> gather flags to root using point-to-point
    int localFlag = isLocalSorted(local) ? 1 : 0;
    int globalFlag = 1;
    if (rank == 0) {
        if (localFlag == 0) globalFlag = 0;
        for (int r = 1; r < size; ++r) {
            int recvFlag;
            MPI_Recv(&recvFlag, 1, MPI_INT, r, 10, comm, MPI_STATUS_IGNORE);
            if (recvFlag == 0) globalFlag = 0;
        }
        for (int r = 1; r < size; ++r) {
            MPI_Send(&globalFlag, 1, MPI_INT, r, 11, comm);
        }
    } else {
        MPI_Send(&localFlag, 1, MPI_INT, 0, 10, comm);
        MPI_Recv(&globalFlag, 1, MPI_INT, 0, 11, comm, MPI_STATUS_IGNORE);
    }
    if (globalFlag == 0) return false;

    // 2) Boundary check between ranks (rank i sends its last to i+1; i>0 receives from i-1)
    if (rank < size - 1 && !local.empty()) {
        long long myLast = local.back();
        MPI_Send(&myLast, 1, MPI_LONG_LONG, rank+1, 20, comm);
    }
    if (rank > 0) {
        long long prevLast = 0;
        MPI_Recv(&prevLast, 1, MPI_LONG_LONG, rank-1, 20, comm, MPI_STATUS_IGNORE);
        if (!local.empty() && prevLast > local.front()) return false;
    }
    return true;
}

void parallelSampleSort(long long n, int rank, int size, MPI_Comm comm, std::vector<long long> &localData) {
    double tstart_all = MPI_Wtime();

    // Step A: generate distributed data locally (no collectives)
    long long local_n = n / size;
    if (rank < (n % size)) local_n++;
    localData.resize(local_n);

    // seed RNG differently per rank
    std::srand((unsigned int)(42 + rank*17));
    for (long long i = 0; i < local_n; ++i)
        localData[i] = (long long) (std::rand()) % 1000000000LL;

    //double t_gen = MPI_Wtime();

    // Step B: local sort
    std::sort(localData.begin(), localData.end());
    //double t_sort = MPI_Wtime();

    // Step C: choose local samples (s = P-1)
    int s = size - 1;
    std::vector<long long> local_samples;
    if (local_n > 0 && s > 0) {
        local_samples.reserve(s);
        for (int i = 1; i <= s; ++i) {
            long long idx = (long long)local_n * i / (s+1);
            if (idx < 0) idx = 0;
            if (idx >= local_n) idx = local_n - 1;
            local_samples.push_back(localData[(size_t)idx]);
        }
    } else {
        local_samples.assign(s, 0);
    }
    //double t_sample = MPI_Wtime();

    // Step D: gather samples to root (point-to-point)
    std::vector<long long> allSamples;
    if (rank == 0) allSamples.resize((size_t)s * size);
    if (rank == 0) {
        // copy root's samples first
        for (int i = 0; i < s; ++i) allSamples[i] = local_samples[i];
        int idx = s;
        for (int src = 1; src < size; ++src) {
            MPI_Recv(allSamples.data() + idx, s, MPI_LONG_LONG, src, 1, comm, MPI_STATUS_IGNORE);
            idx += s;
        }
    } else {
        MPI_Send(local_samples.data(), s, MPI_LONG_LONG, 0, 1, comm);
    }
    //double t_gatherSamples = MPI_Wtime();

    // Step E: root picks pivots and distributes them (point-to-point)
    std::vector<long long> pivots(s, 0);
    if (rank == 0) {
        std::sort(allSamples.begin(), allSamples.end());
        for (int i = 1; i <= s; ++i) {
            size_t pos = (size_t)allSamples.size() * i / (s+1);
            if (pos >= allSamples.size()) pos = allSamples.size() - 1;
            pivots[i-1] = allSamples[pos];
        }
        // send pivots to others
        for (int dst = 1; dst < size; ++dst) {
            MPI_Send(pivots.data(), s, MPI_LONG_LONG, dst, 2, comm);
        }
    } else {
        MPI_Recv(pivots.data(), s, MPI_LONG_LONG, 0, 2, comm, MPI_STATUS_IGNORE);
    }
    //double t_pivots = MPI_Wtime();

    // Step F: partition local data into buckets using corrected logic (reset idx per element)
    std::vector<std::vector<long long>> buckets(size);
    for (auto &v : localData) {
        int idx = 0;
        while (idx < s && v > pivots[idx]) ++idx;
        buckets[idx].push_back(v);
    }

    // Step G: prepare send counts and exchange counts (point-to-point non-blocking)
    std::vector<int> send_counts(size, 0), recv_counts(size, 0);
    for (int r = 0; r < size; ++r) send_counts[r] = (int) buckets[r].size();

    // Post non-blocking receives for counts
n=100, time=7.0944e-05s, correct=✅
    std::vector<MPI_Request> creq_recv(size), creq_send(size);
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            creq_recv[r] = MPI_REQUEST_NULL;
            creq_send[r] = MPI_REQUEST_NULL;
            recv_counts[r] = send_counts[r];
            continue;
        }
        MPI_Irecv(&recv_counts[r], 1, MPI_INT, r, 100, comm, &creq_recv[r]);
    }
    // Post sends for counts
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        MPI_Isend(&send_counts[r], 1, MPI_INT, r, 100, comm, &creq_send[r]);
    }
    // Wait for all count exchanges
    MPI_Waitall(size, creq_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(size, creq_send.data(), MPI_STATUSES_IGNORE);

    // compute recv displs and total receive
    std::vector<int> send_displs(size,0), recv_displs(size,0);
    for (int i = 1; i < size; ++i) send_displs[i] = send_displs[i-1] + send_counts[i-1];
    int total_recv = (recv_counts.size()>0) ? recv_counts[0] : 0;
    for (int i = 1; i < size; ++i) {
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
        total_recv += recv_counts[i];
    }

    // flatten send buffer
    std::vector<long long> send_buffer;
    send_buffer.reserve((size_t) std::accumulate(send_counts.begin(), send_counts.end(), 0));
    for (int r = 0; r < size; ++r) {
        send_buffer.insert(send_buffer.end(), buckets[r].begin(), buckets[r].end());
    }

    // Step H: exchange data non-blocking
    std::vector<long long> recv_buffer(total_recv);
    std::vector<MPI_Request> dreq_recv(size), dreq_send(size);

    // post receives
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            if (recv_counts[r] > 0) {
                // copy own bucket directly from send_buffer (it's contiguous at send_displs[r])
                int sdisp = send_displs[r];
                for (int k = 0; k < recv_counts[r]; ++k) {
                    recv_buffer[recv_displs[r] + k] = send_buffer[sdisp + k];
                }
            }
            dreq_recv[r] = MPI_REQUEST_NULL;
            dreq_send[r] = MPI_REQUEST_NULL;
            continue;
        }
        if (recv_counts[r] > 0) {
            MPI_Irecv(recv_buffer.data() + recv_displs[r], recv_counts[r], MPI_LONG_LONG, r, 101, comm, &dreq_recv[r]);
        } else {
            dreq_recv[r] = MPI_REQUEST_NULL;
        }
    }
    // post sends
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        if (send_counts[r] > 0) {
            MPI_Isend(send_buffer.data() + send_displs[r], send_counts[r], MPI_LONG_LONG, r, 101, comm, &dreq_send[r]);
        } else {
            dreq_send[r] = MPI_REQUEST_NULL;
        }
    }

    // wait for data transfers
    MPI_Waitall(size, dreq_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(size, dreq_send.data(), MPI_STATUSES_IGNORE);

    //double t_exchange = MPI_Wtime();

    // Step I: final local sort of received data
    std::sort(recv_buffer.begin(), recv_buffer.end());
    localData.swap(recv_buffer);
    //double t_finalsort = MPI_Wtime();

    /*if (rank == 0) {
        std::cout << "[timings] gen=" << (t_gen - tstart_all)
                  << " sort=" << (t_sort - t_gen)
                  << " sample=" << (t_sample - t_sort)
                  << " gatherSamples=" << (t_gatherSamples - t_sample)
                  << " pivots=" << (t_pivots - t_gatherSamples)
                  << " exchange=" << (t_exchange - t_pivots)
                  << " finalsort=" << (t_finalsort - t_exchange)
                  << " total=" << (t_finalsort - tstart_all) << " s\n";
    }*/
}

// Scaling experiment
void testScaling(int rank, int size, MPI_Comm comm) {
    std::vector<long long> localData;
    std::vector<long long> testSizes = {100, 1000, 10000, 100000,1000000}; 
    // You can add 1000000 later, but test step by step.

    for (auto n : testSizes) {
        // Synchronize before timing
        MPI_Barrier(comm);
        double start = MPI_Wtime();

        parallelSampleSort(n, rank, size, comm, localData);

        MPI_Barrier(comm);
        double end = MPI_Wtime();

        // Verify correctness
        bool correct = verifyGlobalSort(localData, rank, size, comm);

        if (rank == 0) {
            std::cout << "n=" << n 
                      << ", time=" << (end - start) << "s"
                      << ", correct=" << (correct ? "✅" : "❌")
                      << std::endl;
        }
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    testScaling(rank, size, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
