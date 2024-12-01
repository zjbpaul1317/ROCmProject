#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

// Example IOCTL commands (should be defined according to the actual blue-dmac-driver)
#define BLUE_DMA_CONFIG _IOW('d', 1, struct dma_transfer)
#define BLUE_DMA_START _IO('d', 2)
#define BLUE_DMA_WAIT _IO('d', 3)

#define N 1024

// DMA transfer structure
struct dma_transfer
{
    uint64_t src_addr; // Source address
    uint64_t dst_addr; // Destination address
    uint32_t length;   // Data length
    uint32_t control;  // Control flags (such as transfer direction)
};

enum
{
    DMA_WRITE = 1, // Transfer direction: from FPGA to GPU
    DMA_READ = 2   // Transfer direction: from GPU to FPGA
};

int main()
{
    // Allocate GPU memory
    float *d_data;
    size_t size = N * sizeof(float);
    hipError_t err = hipMalloc(&d_data, size);
    if (err != hipSuccess)
    {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Get the IOVA address of the GPU memory
    uint64_t iova;
    hipDeviceptr_t device_ptr = reinterpret_cast<hipDeviceptr_t>(d_data);
    err = hipExtGetDeviceAddress(&iova, device_ptr);
    if (err != hipSuccess)
    {
        std::cerr << "hipExtGetDeviceAddress failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Interact with the FPGA to configure the DMA transfer
    int fd = open("/dev/blue_dma", O_RDWR);
    if (fd < 0)
    {
        perror("open /dev/blue_dma");
        hipFree(d_data);
        return -1;
    }

    // Assuming the FPGA source address is a constant or obtained through other means
    uint64_t fpga_src_addr = 0x10000000; // Example FPGA source address

    // Configure the DMA transfer
    dma_transfer transfer;
    transfer.src_addr = fpga_src_addr; // FPGA source address
    transfer.dst_addr = iova;          // GPU memory IOVA address
    transfer.length = size;            // Data length
    transfer.control = DMA_WRITE;      // Transfer direction: from FPGA to GPU memory

    int ret = ioctl(fd, BLUE_DMA_CONFIG, &transfer);
    if (ret < 0)
    {
        perror("ioctl BLUE_DMA_CONFIG");
        close(fd);
        hipFree(d_data);
        return -1;
    }

    // Start the DMA transfer
    ret = ioctl(fd, BLUE_DMA_START, NULL);
    if (ret < 0)
    {
        perror("ioctl BLUE_DMA_START");
        close(fd);
        hipFree(d_data);
        return -1;
    }

    // Wait for the DMA transfer to complete
    ret = ioctl(fd, BLUE_DMA_WAIT, NULL);
    if (ret < 0)
    {
        perror("ioctl BLUE_DMA_WAIT");
        close(fd);
        hipFree(d_data);
        return -1;
    }

    close(fd); // Close the DMA device

    // Process the data on the GPU
    // Define a kernel function for data processing or verification
    __global__ void process_data(float *data, int size)
    {
        int idx = threadIdx.x;
        if (idx < size)
        {
            // Process data, for example, multiply each element by 2
            data[idx] *= 2.0f;
        }
    }

    // Launch the kernel for data processing
    hipLaunchKernelGGL(process_data, dim3(1), dim3(N), 0, 0, d_data, N);
    err = hipDeviceSynchronize(); // Ensure the kernel finishes execution
    if (err != hipSuccess)
    {
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Copy the result back to the host and print it
    std::vector<float> h_data(N);
    err = hipMemcpy(h_data.data(), d_data, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Print the processed data
    std::cout << "Processed data: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    hipFree(d_data);

    return 0;
}
