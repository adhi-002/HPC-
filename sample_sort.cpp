#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

void print_vector(const std::vector<int>& v, const std::string& msg, int rank) {
    std::cout << "Rank " << rank << " " << msg << ": ";
    for (int val : v) std::cout << val << " ";
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000; // Total elements to sort

    std::vector<int> data;
    int base_local_n = N / size;
    int remainder = N % size;
    int local_n = base_local_n + (rank < remainder ? 1 : 0);

    // Root process initializes big data vector
    if (rank == 0) {
        data.resize(N);
        std::srand(42);
        for (int i = 0; i < N; ++i) {
            data[i] = std::rand() % 1000000;
        }
    }

    // Prepare scatter counts and displacements
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = base_local_n + (i < remainder ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Allocate local data buffer and scatter data
    std::vector<int> local_data(local_n);
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_INT,
                 local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 1: Sort local data
    std::sort(local_data.begin(), local_data.end());

    // Step 2: Choose samples (p-1 samples evenly spaced)
    int samples_count = size - 1;
    std::vector<int> local_samples(samples_count);
    for (int i = 0; i < samples_count; ++i) {
        int idx = (i + 1) * local_n / (samples_count + 1);
        local_samples[i] = local_data[idx];
    }

    // Step 3: Gather all samples to root
    std::vector<int> all_samples;
    if (rank == 0) all_samples.resize(samples_count * size);
    MPI_Gather(local_samples.data(), samples_count, MPI_INT,
               all_samples.data(), samples_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 4: Root sorts all samples and picks pivots
    std::vector<int> pivots(samples_count);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 0; i < samples_count; ++i) {
            pivots[i] = all_samples[(i + 1) * samples_count];
        }
    }

    // Step 5: Broadcast pivots to all processes
    MPI_Bcast(pivots.data(), samples_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 6: Partition local data by pivots
    std::vector<int> send_counts(size, 0);
    for (int val : local_data) {
        int idx = 0;
        while (idx < samples_count && val > pivots[idx]) idx++;
        send_counts[idx]++;
    }

    std::vector<int> send_displs(size, 0);
    for (int i = 1; i < size; ++i) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    }

    std::vector<int> send_buffer(local_n);
    std::vector<int> current_pos = send_displs;

    for (int val : local_data) {
        int idx = 0;
        while (idx < samples_count && val > pivots[idx]) idx++;
        send_buffer[current_pos[idx]++] = val;
    }

    // Step 7: All-to-all exchange of send_counts to get recv_counts
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> recv_displs(size, 0);
    int total_recv = recv_counts[0];
    for (int i = 1; i < size; ++i) {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        total_recv += recv_counts[i];
    }

    std::vector<int> recv_buffer(total_recv);

    // Step 8: Alltoallv exchange of data
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Step 9: Final local sort of received data
    std::sort(recv_buffer.begin(), recv_buffer.end());

    // Optional: Print small sorted arrays (commented out for large N)
    /*
    print_vector(recv_buffer, "final sorted chunk", rank);
    */

    MPI_Finalize();
    return 0;
}
