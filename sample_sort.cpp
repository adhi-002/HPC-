#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Compare function for qsort
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 16;  // total data size
    int *data = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = n / size;
    int *local_data = malloc(local_n * sizeof(int));

    if (rank == 0) {
        data = malloc(n * sizeof(int));
        srand(time(NULL));
        printf("Unsorted data: ");
        for (int i = 0; i < n; i++) {
            data[i] = rand() % 100;
            printf("%d ", data[i]);
        }
        printf("\n");

        // Scatter manually using point-to-point
        for (int p = 1; p < size; p++) {
            MPI_Send(data + p * local_n, local_n, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
        // Copy my own chunk
        for (int i = 0; i < local_n; i++)
            local_data[i] = data[i];
    } else {
        MPI_Recv(local_data, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Step 1: Local sort
    qsort(local_data, local_n, sizeof(int), compare);

    // Step 2: Choose local samples
    int s = size - 1;  // number of samples per process
    int *samples = malloc(s * sizeof(int));
    for (int i = 0; i < s; i++) {
        samples[i] = local_data[(i + 1) * local_n / size];
    }

    // Gather samples at root
    int *all_samples = NULL;
    if (rank == 0) all_samples = malloc(s * size * sizeof(int));

    if (rank == 0) {
        // Copy root samples
        for (int i = 0; i < s; i++)
            all_samples[i] = samples[i];

        // Receive from others
        for (int p = 1; p < size; p++) {
            MPI_Recv(all_samples + p * s, s, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(samples, s, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    // Step 3: Root chooses splitters
    int *splitters = malloc((size - 1) * sizeof(int));
    if (rank == 0) {
        qsort(all_samples, s * size, sizeof(int), compare);
        for (int i = 0; i < size - 1; i++) {
            splitters[i] = all_samples[(i + 1) * size];
        }
    }

    // Broadcast splitters manually
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            MPI_Send(splitters, size - 1, MPI_INT, p, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(splitters, size - 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Step 4: Partition local data based on splitters
    int *send_counts = calloc(size, sizeof(int));
    int *positions = calloc(size, sizeof(int));
    for (int i = 0; i < local_n; i++) {
        int val = local_data[i];
        int p = 0;
        while (p < size - 1 && val > splitters[p]) p++;
        send_counts[p]++;
    }

    int **send_data = malloc(size * sizeof(int *));
    for (int p = 0; p < size; p++) {
        send_data[p] = malloc(send_counts[p] * sizeof(int));
    }

    for (int i = 0; i < local_n; i++) {
        int val = local_data[i];
        int p = 0;
        while (p < size - 1 && val > splitters[p]) p++;
        send_data[p][positions[p]++] = val;
    }

    // Step 5: Exchange partitions
    int *recv_counts = malloc(size * sizeof(int));
    for (int p = 0; p < size; p++) {
        if (p != rank) {
            MPI_Send(&send_counts[p], 1, MPI_INT, p, 3, MPI_COMM_WORLD);
            MPI_Recv(&recv_counts[p], 1, MPI_INT, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            recv_counts[p] = send_counts[p];
        }
    }

    int total_recv = 0;
    for (int p = 0; p < size; p++) total_recv += recv_counts[p];
    int *recv_data = malloc(total_recv * sizeof(int));

    int offset = 0;
    for (int p = 0; p < size; p++) {
        if (p != rank) {
            MPI_Send(send_data[p], send_counts[p], MPI_INT, p, 4, MPI_COMM_WORLD);
            MPI_Recv(recv_data + offset, recv_counts[p], MPI_INT, p, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += recv_counts[p];
        } else {
            for (int i = 0; i < recv_counts[p]; i++)
                recv_data[offset++] = send_data[p][i];
        }
    }

    // Step 6: Final sort
    qsort(recv_data, total_recv, sizeof(int), compare);

    // Gather results back to root for display
    if (rank == 0) {
        int *sizes = malloc(size * sizeof(int));
        sizes[0] = total_recv;
        for (int p = 1; p < size; p++) {
            MPI_Recv(&sizes[p], 1, MPI_INT, p, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int total = sizes[0];
        for (int p = 1; p < size; p++) total += sizes[p];

        int *final = malloc(total * sizeof(int));
        int pos = 0;
        for (int i = 0; i < total_recv; i++) final[pos++] = recv_data[i];

        for (int p = 1; p < size; p++) {
            MPI_Recv(final + pos, sizes[p], MPI_INT, p, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pos += sizes[p];
        }

        printf("Sorted data: ");
        for (int i = 0; i < total; i++) printf("%d ", final[i]);
        printf("\n");

    } else {
        MPI_Send(&total_recv, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
        MPI_Send(recv_data, total_recv, MPI_INT, 0, 6, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}