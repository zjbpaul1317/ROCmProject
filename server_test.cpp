#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>

// Define IOCTL commands
#define XDMA_CONFIG _IOW('d', 1, struct dma_transfer)
#define XDMA_START _IO('d', 2)
#define XDMA_WAIT _IO('d', 3)

// DMA transfer structure
struct dma_transfer
{
    uint64_t src_addr; // Source address (host memory or GPU memory IOVA address)
    uint64_t dst_addr; // Destination address (host memory or GPU memory IOVA address)
    uint32_t length;   // Data length (in bytes)
    uint32_t control;  // Control flags (transfer direction)
};

// Transfer direction enum
enum
{
    DMA_WRITE = 1, // Transfer direction: Host to FPGA or FPGA to GPU
    DMA_READ = 2   // Transfer direction: GPU to FPGA or FPGA to Host
};

// DMA device file paths (adjust according to the actual device paths)
#define DMA_DEV_H2C "/dev/xdma0_h2c_0" // Host to FPGA
#define DMA_DEV_C2H "/dev/xdma0_c2h_0" // FPGA to Host (including FPGA to GPU and GPU to FPGA)

// FPGA data cache physical address
#define FPGA_CACHE_ADDR 0x20000000 // Example address, replace with actual address

/**
 * @brief Configure DMA transfer
 *
 * @param fd DMA device file descriptor
 * @param transfer DMA transfer parameters
 * @return int 0 on success, -1 on failure
 */
int configure_dma(int fd, struct dma_transfer &transfer)
{
    if (ioctl(fd, XDMA_CONFIG, &transfer) < 0)
    {
        perror("ioctl XDMA_CONFIG");
        return -1;
    }
    return 0;
}

/**
 * @brief Start DMA transfer
 *
 * @param fd DMA device file descriptor
 * @return int 0 on success, -1 on failure
 */
int start_dma(int fd)
{
    if (ioctl(fd, XDMA_START, NULL) < 0)
    {
        perror("ioctl XDMA_START");
        return -1;
    }
    return 0;
}

/**
 * @brief Wait for DMA transfer to complete
 *
 * @param fd DMA device file descriptor
 * @return int 0 on success, -1 on failure
 */
int wait_dma(int fd)
{
    if (ioctl(fd, XDMA_WAIT, NULL) < 0)
    {
        perror("ioctl XDMA_WAIT");
        return -1;
    }
    return 0;
}

/**
 * @brief Verify the correctness of the data transfer
 *
 * @param original Pointer to the original data
 * @param received Pointer to the received data
 * @param N Number of data elements
 * @param processed Whether the data has been processed (e.g., multiplied by 2)
 * @return bool Returns true if the data matches, false otherwise
 */
bool verify_data(const float *original, const float *received, size_t N, bool processed = false)
{
    for (size_t i = 0; i < N; ++i)
    {
        float expected = processed ? original[i] * 2.0f : original[i];
        if (received[i] != expected)
        {
            std::cerr << "Data mismatch at index " << i << ": expected = "
                      << expected << ", received = " << received[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // Define send and receive data
    size_t N = 1024; // Number of data elements
    size_t size = N * sizeof(float);

    float *h_send_data;
    float *h_receive_data;

    // Allocate pinned host memory (suitable for DMA transfers)
    hipError_t err = hipHostMalloc(&h_send_data, size, hipHostMallocDefault);
    if (err != hipSuccess)
    {
        std::cerr << "hipHostMalloc for h_send_data failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipHostMalloc(&h_receive_data, size, hipHostMallocDefault);
    if (err != hipSuccess)
    {
        std::cerr << "hipHostMalloc for h_receive_data failed: " << hipGetErrorString(err) << std::endl;
        hipHostFree(h_send_data);
        return -1;
    }

    // Initialize send data
    for (size_t i = 0; i < N; ++i)
    {
        h_send_data[i] = static_cast<float>(i);
    }
    std::cout << "Host send data initialization complete." << std::endl;

    // Allocate GPU memory
    float *d_data;
    err = hipMalloc(&d_data, size);
    if (err != hipSuccess)
    {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "Allocated " << size << " bytes of GPU memory." << std::endl;

    // Copy host send data to GPU memory (initialization, optional)
    err = hipMemcpy(d_data, h_send_data, size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        std::cerr << "hipMemcpy failed to send to GPU memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "Host send data copied to GPU memory." << std::endl;

    // Get GPU memory IOVA address
    uint64_t iova = 0;
    hipDeviceptr_t device_ptr = reinterpret_cast<hipDeviceptr_t>(d_data);
    err = hipExtGetDeviceAddress(&iova, device_ptr);
    if (err != hipSuccess)
    {
        std::cerr << "hipExtGetDeviceAddress failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "GPU memory IOVA address: 0x" << std::hex << iova << std::dec << std::endl;

    // Step-by-step DMA transfer process

    // Host memory to FPGA data cache
    std::cout << "\nStep 1: Host memory to FPGA data cache DMA transfer." << std::endl;
    std::cout << "Press Enter to proceed with the host to FPGA transfer...";
    std::cin.ignore(); // Wait for user to press Enter

    int fd_h2c = open(DMA_DEV_H2C, O_RDWR);
    if (fd_h2c < 0)
    {
        perror("Failed to open " DMA_DEV_H2C);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "Opened " << DMA_DEV_H2C << " for host to FPGA transfer." << std::endl;

    struct dma_transfer transfer_h2c;
    std::memset(&transfer_h2c, 0, sizeof(transfer_h2c));
    transfer_h2c.src_addr = reinterpret_cast<uint64_t>(h_send_data); // Host memory address
    transfer_h2c.dst_addr = FPGA_CACHE_ADDR;                         // FPGA data cache address
    transfer_h2c.length = size;                                      // Data length
    transfer_h2c.control = DMA_WRITE;                                // Transfer direction: Host to FPGA

    if (configure_dma(fd_h2c, transfer_h2c) != 0)
    {
        close(fd_h2c);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "DMA transfer parameters configured: Host to FPGA." << std::endl;

    if (start_dma(fd_h2c) != 0)
    {
        close(fd_h2c);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "DMA transfer started: Host to FPGA." << std::endl;

    if (wait_dma(fd_h2c) != 0)
    {
        close(fd_h2c);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "DMA transfer completed: Host to FPGA." << std::endl;

    close(fd_h2c); // Close H2C device file

    // FPGA data cache to GPU memory
    std::cout << "\nStep 2: FPGA data cache to GPU memory DMA transfer." << std::endl;
    std::cout << "Press Enter to proceed with the FPGA to GPU transfer...";
    std::cin.ignore(); // Wait for user to press Enter

    int fd_c2h = open(DMA_DEV_C2H, O_RDWR);
    if (fd_c2h < 0)
    {
        perror("Failed to open " DMA_DEV_C2H);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "Opened " << DMA_DEV_C2H << " for FPGA to GPU transfer." << std::endl;

    struct dma_transfer transfer_c2h;
    std::memset(&transfer_c2h, 0, sizeof(transfer_c2h));
    transfer_c2h.src_addr = FPGA_CACHE_ADDR; // FPGA data cache address
    transfer_c2h.dst_addr = iova;            // GPU memory IOVA address
    transfer_c2h.length = size;              // Data length
    transfer_c2h.control = DMA_READ;         // Transfer direction: FPGA to GPU

    if (configure_dma(fd_c2h, transfer_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "DMA transfer parameters configured: FPGA to GPU." << std::endl;

    if (start_dma(fd_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "DMA transfer started: FPGA to GPU." << std::endl;

    if (wait_dma(fd_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "DMA transfer completed: FPGA to GPU." << std::endl;

    close(fd_c2h); // Close C2H device file

    // GPU memory to Host memory
    std::cout << "\nStep 3: GPU memory to Host memory DMA transfer." << std::endl;
    std::cout << "Press Enter to proceed with the GPU to Host transfer...";
    std::cin.ignore(); // Wait for user to press Enter

    err = hipMemcpy(h_receive_data, d_data, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipHostFree(h_send_data);
        hipHostFree(h_receive_data);
        return -1;
    }
    std::cout << "GPU memory copied to host memory." << std::endl;

    // Verify data correctness
    bool is_verified = verify_data(h_send_data, h_receive_data, N, true); // Verifying that the data has been processed correctly
    if (is_verified)
    {
        std::cout << "DMA transfer verified successfully." << std::endl;
    }
    else
    {
        std::cerr << "DMA transfer verification failed." << std::endl;
    }

    // Free resources
    hipFree(d_data);
    hipHostFree(h_send_data);
    hipHostFree(h_receive_data);

    return 0;
}
