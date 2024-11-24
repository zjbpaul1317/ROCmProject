#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define N 1024

// abstract interface to tell FPGA what data to read from
void fpga_register_write(uint64_t address, uint32_t value)
{
    // A placeholder and will be filled when implemented
    std::cout << "FPGA register write (stub): Address = " << address << ", Value = " << value << std::endl;
}

// simple addition operation
__global__ void gpu_addition(float *data, int size)
{
    int idx = threadIdx.x;
    if (idx < size)
    {
        data[idx] += 1.0f; // add 1 to each data
    }
}

int main()
{
    // allocate GPU memory
    float *d_data;
    hipMalloc(&d_data, N * sizeof(float));

    // initialize data and copy it to the GPU
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++)
    {
        h_data[i] = static_cast<float>(i);
    }

    hipMemcpy(d_data, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice);

    // execute GPU computation
    hipLaunchKernelGGL(gpu_addition, dim3(1), dim3(N), 0, 0, d_data, N);

    // copy the result back to the host
    hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);

    // print the result
    std::cout << "Result from GPU: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // an abstract interface for placeholder
    uint64_t fake_address = 0x1000;                // assume GPU memory
    uint32_t fake_value = 12345;                   // assume values to be written.
    fpga_register_write(fake_address, fake_value); // abstract interface to write to FPGA registers

    // release memory
    hipFree(d_data);

    return 0;
}
