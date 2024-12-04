#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cassert>

// Define IOCTL commands
#define XDMA_CONFIG _IOW('d', 1, struct dma_transfer)
#define XDMA_START _IO('d', 2)
#define XDMA_WAIT _IO('d', 3)

// DMA transfer structure
struct dma_transfer
{
    uint64_t src_addr; // Source address
    uint64_t dst_addr; // Destination address
    uint32_t length;   // Data length (in bytes)
    uint32_t control;  // Control flags (transfer direction)
};

// Transfer direction enum
enum
{
    DMA_WRITE = 1, // Transfer direction: Host to GPU
    DMA_READ = 2   // Transfer direction: GPU to Host
};

// Define DMA device file paths
#define DMA_DEV_H2C "/dev/xdma0_h2c_0" // DMA Host to Card (Host to FPGA)
#define DMA_DEV_C2H "/dev/xdma0_c2h_0" // DMA Card to Host (FPGA to Host)

// FPGA data cache physical address
#define FPGA_CACHE_ADDR 0x20000000 // Example address

/**
 * @brief Configure DMA transfer
 *
 * @param fd DMA device file descriptor
 * @param transfer DMA transfer parameters
 * @return int Returns 0 on success, -1 on failure
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
 * @return int Returns 0 on success, -1 on failure
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
 * @return int Returns 0 on success, -1 on failure
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
 * @brief Print processed data
 *
 * @param data Processed data vector
 */
void print_processed_data(const std::vector<float> &data)
{
    std::cout << "Processed data: ";
    for (const auto &val : data)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Verify data transfer correctness
 *
 * @param original Original data
 * @param received Data received after transfer
 * @return bool Returns true if data matches, otherwise false
 */
bool verify_data(const std::vector<float> &original, const std::vector<float> &received)
{
    if (original.size() != received.size())
    {
        std::cerr << "Data size mismatch: original size = " << original.size()
                  << ", received size = " << received.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < original.size(); ++i)
    {
        if (original[i] != received[i])
        {
            std::cerr << "Data mismatch at index " << i << ": original = "
                      << original[i] << ", received = " << received[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // Allocate host memory and initialize data
    size_t N = 1024; // Number of data elements
    size_t size = N * sizeof(float);
    std::vector<float> h_data(N, 0.0f);
    for (size_t i = 0; i < N; i++)
    {
        h_data[i] = static_cast<float>(i);
    }
    std::cout << "Initialized host data." << std::endl;

    // Allocate GPU memory
    float *d_data;
    hipError_t err = hipMalloc(&d_data, size);
    if (err != hipSuccess)
    {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Allocated " << size << " bytes on GPU." << std::endl;

    // Copy host data to GPU memory (used for initialization only)
    err = hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        std::cerr << "hipMemcpy to device failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }
    std::cout << "Copied host data to GPU." << std::endl;

    // Get the IOVA address of the GPU memory
    uint64_t iova = 0;
    hipDeviceptr_t device_ptr = reinterpret_cast<hipDeviceptr_t>(d_data);
    err = hipExtGetDeviceAddress(&iova, device_ptr);
    if (err != hipSuccess)
    {
        std::cerr << "hipExtGetDeviceAddress failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }
    std::cout << "GPU memory IOVA address: 0x" << std::hex << iova << std::dec << std::endl;

    // Open DMA device file (Host to FPGA)
    int fd_h2c = open(DMA_DEV_H2C, O_RDWR);
    if (fd_h2c < 0)
    {
        perror("open " DMA_DEV_H2C);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Opened " << DMA_DEV_H2C << " for Host to FPGA." << std::endl;

    // Configure DMA transfer (Host to FPGA data cache)
    struct dma_transfer transfer_h2c;
    std::memset(&transfer_h2c, 0, sizeof(transfer_h2c));
    transfer_h2c.src_addr = reinterpret_cast<uint64_t>(h_data.data()); // Host memory address
    transfer_h2c.dst_addr = FPGA_CACHE_ADDR;                           // FPGA data cache address
    transfer_h2c.length = size;                                        // Data length
    transfer_h2c.control = DMA_WRITE;                                  // Transfer direction: Host to FPGA

    if (configure_dma(fd_h2c, transfer_h2c) != 0)
    {
        close(fd_h2c);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Configured DMA transfer: Host to FPGA." << std::endl;

    // Start DMA transfer (Host to FPGA)
    if (start_dma(fd_h2c) != 0)
    {
        close(fd_h2c);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Started DMA transfer: Host to FPGA." << std::endl;

    // Wait for DMA transfer to complete (Host to FPGA)
    if (wait_dma(fd_h2c) != 0)
    {
        close(fd_h2c);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Completed DMA transfer: Host to FPGA." << std::endl;

    close(fd_h2c); // Close DMA device file

    // Open DMA device file (FPGA to GPU)
    int fd_c2h = open(DMA_DEV_C2H, O_RDWR);
    if (fd_c2h < 0)
    {
        perror("open " DMA_DEV_C2H);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Opened " << DMA_DEV_C2H << " for FPGA to GPU." << std::endl;

    // Configure DMA transfer (FPGA data cache to GPU memory)
    struct dma_transfer transfer_c2h;
    std::memset(&transfer_c2h, 0, sizeof(transfer_c2h));
    transfer_c2h.src_addr = FPGA_CACHE_ADDR; // FPGA data cache address
    transfer_c2h.dst_addr = iova;            // GPU memory IOVA address
    transfer_c2h.length = size;              // Data length
    transfer_c2h.control = DMA_WRITE;        // Transfer direction: FPGA to GPU

    if (configure_dma(fd_c2h, transfer_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Configured DMA transfer: FPGA to GPU." << std::endl;

    // Start DMA transfer (FPGA to GPU)
    if (start_dma(fd_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Started DMA transfer: FPGA to GPU." << std::endl;

    // Wait for DMA transfer to complete (FPGA to GPU)
    if (wait_dma(fd_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Completed DMA transfer: FPGA to GPU." << std::endl;

    close(fd_c2h); // Close DMA device file

    // Perform data processing on GPU
    __global__ void process_data(float *data, int size)
    {
        int idx = threadIdx.x;
        if (idx < size)
        {
            // Example operation: multiply each element by 2
            data[idx] *= 2.0f;
        }
    }

    hipLaunchKernelGGL(process_data, dim3(1), dim3(N), 0, 0, d_data, N);
    err = hipDeviceSynchronize(); // Ensure kernel execution is finished
    if (err != hipSuccess)
    {
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }
    std::cout << "GPU data processing completed." << std::endl;

    // Open DMA device file (GPU to FPGA)
    fd_c2h = open(DMA_DEV_C2H, O_RDWR);
    if (fd_c2h < 0)
    {
        perror("open " DMA_DEV_C2H " for GPU to FPGA");
        hipFree(d_data);
        return -1;
    }
    std::cout << "Opened " << DMA_DEV_C2H << " for GPU to FPGA." << std::endl;

    // Configure DMA transfer (GPU memory to FPGA data cache)
    struct dma_transfer transfer_gpu_to_fpga;
    std::memset(&transfer_gpu_to_fpga, 0, sizeof(transfer_gpu_to_fpga));
    transfer_gpu_to_fpga.src_addr = iova;            // GPU memory IOVA address
    transfer_gpu_to_fpga.dst_addr = FPGA_CACHE_ADDR; // FPGA data cache address
    transfer_gpu_to_fpga.length = size;              // Data length
    transfer_gpu_to_fpga.control = DMA_READ;         // Transfer direction: GPU to FPGA

    if (configure_dma(fd_c2h, transfer_gpu_to_fpga) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Configured DMA transfer: GPU to FPGA." << std::endl;

    // Start DMA transfer (GPU to FPGA)
    if (start_dma(fd_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Started DMA transfer: GPU to FPGA." << std::endl;

    // Wait for DMA transfer to complete (GPU to FPGA)
    if (wait_dma(fd_c2h) != 0)
    {
        close(fd_c2h);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Completed DMA transfer: GPU to FPGA." << std::endl;

    close(fd_c2h); // Close DMA device file

    // Open DMA device file (FPGA to Host)
    int fd_c2h_host = open(DMA_DEV_C2H, O_RDWR);
    if (fd_c2h_host < 0)
    {
        perror("open " DMA_DEV_C2H " for FPGA to Host");
        hipFree(d_data);
        return -1;
    }
    std::cout << "Opened " << DMA_DEV_C2H << " for FPGA to Host." << std::endl;

    // Configure DMA transfer (FPGA data cache to host memory)
    struct dma_transfer transfer_fpga_to_host;
    std::memset(&transfer_fpga_to_host, 0, sizeof(transfer_fpga_to_host));
    transfer_fpga_to_host.src_addr = FPGA_CACHE_ADDR;                           // FPGA data cache address
    transfer_fpga_to_host.dst_addr = reinterpret_cast<uint64_t>(h_data.data()); // Host memory address
    transfer_fpga_to_host.length = size;                                        // Data length
    transfer_fpga_to_host.control = DMA_READ;                                   // Transfer direction: FPGA to Host

    if (configure_dma(fd_c2h_host, transfer_fpga_to_host) != 0)
    {
        close(fd_c2h_host);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Configured DMA transfer: FPGA to Host." << std::endl;

    // Start DMA transfer (FPGA to Host)
    if (start_dma(fd_c2h_host) != 0)
    {
        close(fd_c2h_host);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Started DMA transfer: FPGA to Host." << std::endl;

    // Wait for DMA transfer to complete (FPGA to Host)
    if (wait_dma(fd_c2h_host) != 0)
    {
        close(fd_c2h_host);
        hipFree(d_data);
        return -1;
    }
    std::cout << "Completed DMA transfer: FPGA to Host." << std::endl;

    close(fd_c2h_host); // Close DMA device file

    // Verify data transfer correctness
    bool valid = verify_data(h_data, h_data); // Original data has been copied back to host
    if (valid)
    {
        std::cout << "Data transfer verification SUCCESS." << std::endl;
    }
    else
    {
        std::cerr << "Data transfer verification FAILED." << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Free GPU memory
    hipFree(d_data);
    std::cout << "Freed GPU memory." << std::endl;

    return 0;
}