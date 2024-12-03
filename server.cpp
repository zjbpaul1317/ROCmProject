#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>

// Define IOCTL commands
#define BLUE_DMA_CONFIG _IOW('d', 1, struct dma_transfer)
#define BLUE_DMA_START _IO('d', 2)
#define BLUE_DMA_WAIT _IO('d', 3)

// Define the DMA transfer structure
struct dma_transfer
{
    uint64_t src_addr; // Source address (FPGA source address or GPU IOVA address)
    uint64_t dst_addr; // Destination address (GPU IOVA address or FPGA destination address)
    uint32_t length;   // Data length (in bytes)
    uint32_t control;  // Control flags (e.g., transfer direction)
};

// Define transfer direction enum
enum
{
    DMA_WRITE = 1, // Transfer direction: from FPGA to GPU
    DMA_READ = 2   // Transfer direction: from GPU to FPGA
};

// Define constants and functions for BAR interface (placeholders)
#define BAR_SIZE 4096                            // BAR space size
#define BDMA_CONTROL_OFFSET 0x0                  // FPGA control register offset
#define BDMA_DMA_CONTROL_OFFSET 0x8              // DMA control register offset
#define BDMA_CONTROL_REG "/dev/blue_dma_control" // FPGA control register device file path

/**
 * @brief Abstract interface to configure FPGA registers (placeholder)
 *
 * @param gpu_iova GPU memory IOVA address
 * @param dma_control_flag DMA transfer control flag
 */
// Map the FPGA's BAR address space and write to registers
void configure_fpga_registers(uint64_t gpu_iova, uint32_t dma_control_flag)
{
    int bar_fd = open(BDMA_CONTROL_REG, O_RDWR | O_SYNC);
    if (bar_fd < 0)
    {
        perror("open /dev/blue_dma_control");
        return;
    }

    // Map the BAR address space
    void *bar_addr = mmap(NULL, BAR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, bar_fd, 0);
    if (bar_addr == MAP_FAILED)
    {
        perror("mmap");
        close(bar_fd);
        return;
    }

    // Write the IOVA address to FPGA control register (offset 0x0)
    *((volatile uint64_t *)(static_cast<char *>(bar_addr) + BDMA_CONTROL_OFFSET)) = gpu_iova;

    // Write DMA control flag to FPGA DMA control register (offset 0x8)
    *((volatile uint32_t *)(static_cast<char *>(bar_addr) + BDMA_DMA_CONTROL_OFFSET)) = dma_control_flag;

    // Unmap the address space and close the file descriptor
    munmap(bar_addr, BAR_SIZE);
    close(bar_fd);

    std::cout << "Configured FPGA registers via BAR: IOVA = 0x"
              << std::hex << gpu_iova << ", Control Flag = " << std::dec
              << dma_control_flag << std::endl;
}

/**
 * @brief Print the processed data
 *
 * @param data The processed data vector
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

int main()
{
    // Allocate GPU memory
    float *d_data;
    size_t N = 1024; // Number of data elements
    size_t size = N * sizeof(float);
    hipError_t err = hipMalloc(&d_data, size);
    if (err != hipSuccess)
    {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Allocated " << size << " bytes on GPU." << std::endl;

    // Initialize host data and copy to GPU
    std::vector<float> h_data(N, 0.0f);
    for (size_t i = 0; i < N; i++)
    {
        h_data[i] = static_cast<float>(i);
    }

    err = hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        std::cerr << "hipMemcpy to device failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }
    std::cout << "Copied data to GPU." << std::endl;

    // Get GPU memory IOVA address
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

    // Configure FPGA registers through the abstract interface (placeholder)
    configure_fpga_registers(iova, DMA_WRITE);

    // Interact with FPGA and configure DMA transfer
    int fd = open("/dev/blue_dma", O_RDWR);
    if (fd < 0)
    {
        perror("open /dev/blue_dma");
        hipFree(d_data);
        return -1;
    }
    std::cout << "Opened /dev/blue_dma." << std::endl;

    uint64_t fpga_src_addr = 0x10000000; // FPGA source address

    // Configure DMA transfer
    struct dma_transfer transfer;
    std::memset(&transfer, 0, sizeof(transfer));
    transfer.src_addr = fpga_src_addr; // FPGA source address
    transfer.dst_addr = iova;          // GPU memory IOVA address
    transfer.length = size;            // Data length
    transfer.control = DMA_WRITE;      // Transfer direction: from FPGA to GPU

    // Send configuration command
    int ret = ioctl(fd, BLUE_DMA_CONFIG, &transfer);
    if (ret < 0)
    {
        perror("ioctl BLUE_DMA_CONFIG");
        close(fd);
        hipFree(d_data);
        return -1;
    }
    std::cout << "DMA transfer configured successfully." << std::endl;

    // Start DMA transfer
    ret = ioctl(fd, BLUE_DMA_START, NULL);
    if (ret < 0)
    {
        perror("ioctl BLUE_DMA_START");
        close(fd);
        hipFree(d_data);
        return -1;
    }
    std::cout << "DMA transfer started." << std::endl;

    // Wait for DMA transfer to complete
    ret = ioctl(fd, BLUE_DMA_WAIT, NULL);
    if (ret < 0)
    {
        perror("ioctl BLUE_DMA_WAIT");
        close(fd);
        hipFree(d_data);
        return -1;
    }
    std::cout << "DMA transfer completed." << std::endl;

    close(fd); // Close the DMA device file

    // Process data on GPU
    // Define and launch a kernel function to process or verify data
    __global__ void process_data(float *data, int size)
    {
        int idx = threadIdx.x;
        if (idx < size)
        {
            // Multiply each element by 2
            data[idx] *= 2.0f;
        }
    }

    // Launch the kernel function
    hipLaunchKernelGGL(process_data, dim3(1), dim3(N), 0, 0, d_data, N);
    err = hipDeviceSynchronize(); // Ensure the kernel has completed
    if (err != hipSuccess)
    {
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }
    std::cout << "GPU data processing completed." << std::endl;

    // Copy the processed data back to host and print
    std::vector<float> h_result(N);
    err = hipMemcpy(h_result.data(), d_data, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        std::cerr << "hipMemcpy to host failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    print_processed_data(h_result);

    // Free GPU memory
    hipFree(d_data);
    std::cout << "Freed GPU memory." << std::endl;

    return 0;
}
