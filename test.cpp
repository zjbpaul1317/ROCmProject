#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define N 1024

// 抽象的接口，用来告诉FPGA读取什么地方的数据
void fpga_register_write(uint64_t address, uint32_t value)
{
    // 当前接口只是占位符，等具体实现时填充功能
    std::cout << "FPGA register write (stub): Address = " << address << ", Value = " << value << std::endl;
}

// 简单的加法操作
__global__ void gpu_addition(float *data, int size)
{
    int idx = threadIdx.x;
    if (idx < size)
    {
        data[idx] += 1.0f; // 给每个数据加1
    }
}

int main()
{
    // 分配GPU显存
    float *d_data;
    hipMalloc(&d_data, N * sizeof(float));

    // 初始化数据并复制到GPU
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++)
    {
        h_data[i] = static_cast<float>(i);
    }

    hipMemcpy(d_data, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice);

    // 执行GPU计算（加法操作）
    hipLaunchKernelGGL(gpu_addition, dim3(1), dim3(N), 0, 0, d_data, N);

    // 从GPU复制结果回主机
    hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Result from GPU: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // 使用抽象接口（占位符）
    uint64_t fake_address = 0x1000;                // 假设的GPU显存地址
    uint32_t fake_value = 12345;                   // 假设的要写入的值
    fpga_register_write(fake_address, fake_value); // 写入FPGA寄存器的抽象接口

    // 释放显存
    hipFree(d_data);

    return 0;
}
