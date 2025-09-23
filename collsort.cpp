#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <climits>

// Check if local array is sorted
bool isLocalSorted(const std::vector<long long> &arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i-1] > arr[i]) return false;
    }
    return true;
}

// Global correctness check
bool verifyGlobalSort(const std::vector<long long> &local, int rank, int size, MPI_Comm comm) {
    int localFlag = isLocalSorted(local) ? 1 : 0;
    int globalFlag;
    MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_MIN, comm);
    if (globalFlag == 0) return false;

    // check boundaries
    long long myLast = local.empty() ? LLONG_MIN : local.back();
    long long myFirst = local.empty() ? LLONG_MAX : local.front();
    long long prevLast;
    if (rank > 0) {
        MPI_Recv(&prevLast, 1, MPI_LONG_LONG, rank-1, 100, comm, MPI_STATUS_IGNORE);
        if (prevLast > myFirst) return false;
    }
    if (rank < size-1 && !local.empty()) {
        MPI_Send(&myLast, 1, MPI_LONG_LONG, rank+1, 100, comm);
    }
    return true;
}

// Parallel sample sort using collectives
void parallelSampleSort(long long n, int rank, int size, MPI_Comm comm, std::vector<long long> &localData) {
    // Step 1: generate local data
    long long local_n = n / size;
    if (rank < n % size) local_n++;
    localData.resize(local_n);
    srand(time(NULL) + rank);
    for (long long i = 0; i < local_n; i++)
        localData[i] = rand() % 1000000000LL;

    // Step 2: local sort
    std::sort(localData.begin(), localData.end());

    // Step 3: pick local samples
    int s = size - 1;
    std::vector<long long> samples(s);
    for (int i = 0; i < s; i++)
        samples[i] = localData[local_n * (i+1) / (s+1)];

    // Step 4: gather samples at root
    std::vector<long long> allSamples;
    if (rank == 0) allSamples.resize(s * size);
    MPI_Gather(samples.data(), s, MPI_LONG_LONG, allSamples.data(), s, MPI_LONG_LONG, 0, comm);

    // Step 5: choose pivots and broadcast
    std::vector<long long> pivots(s);
    if (rank == 0) {
        std::sort(allSamples.begin(), allSamples.end());
        for (int i = 0; i < s; i++)
            pivots[i] = allSamples[allSamples.size() * (i+1) / (s+1)];
    }
    MPI_Bcast(pivots.data(), s, MPI_LONG_LONG, 0, comm);

    // Step 6: partition local data into buckets
    std::vector<int> sendCounts(size, 0);
    std::vector<std::vector<long long>> buckets(size);
    int idx = 0;
    for (auto v : localData) {
        while (idx < s && v > pivots[idx]) idx++;
        buckets[idx].push_back(v);
    }
    for (int i = 0; i < size; i++)
        sendCounts[i] = buckets[i].size();

    // Step 7: exchange sizes
    std::vector<int> recvCounts(size);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);

    // Step 8: exchange data using Alltoallv
    std::vector<long long> sendBuf, recvBuf;
    std::vector<int> sdispls(size, 0), rdispls(size, 0);
    for (int i = 0; i < size; i++) {
        sdispls[i] = sendBuf.size();
        sendBuf.insert(sendBuf.end(), buckets[i].begin(), buckets[i].end());
    }
    int totalRecv = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    recvBuf.resize(totalRecv);
    for (int i = 1; i < size; i++) rdispls[i] = rdispls[i-1] + recvCounts[i-1];

    MPI_Alltoallv(sendBuf.data(), sendCounts.data(), sdispls.data(), MPI_LONG_LONG,
                  recvBuf.data(), recvCounts.data(), rdispls.data(), MPI_LONG_LONG, comm);

    // Step 9: final local sort
    std::sort(recvBuf.begin(), recvBuf.end());
    localData.swap(recvBuf);
}

// Scaling experiment
void testScaling(int rank, int size, MPI_Comm comm) {
    std::vector<long long> localData;
    std::vector<int> testSizes = {100}; // adjust for larger tests
    for (auto n : testSizes) {
        double start = MPI_Wtime();
        parallelSampleSort(n, rank, size, comm, localData);
        double end = MPI_Wtime();

        bool correct = verifyGlobalSort(localData, rank, size, comm);
        if (rank == 0) {
            std::cout << "n=" << n << ", time=" << (end-start) 
                      << "s, correct=" << (correct ? "✅" : "❌") << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    testScaling(rank, size, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
