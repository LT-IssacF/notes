## 课程来源: [CUDA在现代C++中如何运用？](https://www.bilibili.com/video/BV16b4y1E74f/ "双笙子佯谬") ##
## 前置知识 ##
* `__global__`核函数，从CPU端通过三重尖括号调用，在GPU端上执行，不可以有返回值
* `__device__`设备函数，在GPU上调用和执行，不需要尖括号，可以有返回值
* `host`可以调用`global`，`global`可以调用`device`，`device`可以调用`device`
* `cudaDeviceSynchronize()`同步函数，令`host`等待`device`执行完
* `__inline__`同C++的内敛函数声明
* `__host__ __device__`可以将函数同时定义在CPU和GPU上并调用，如下：
```C++
__host__ __device__ void say_hello() {
#ifdef __CUDA_ARCH__
    printf("GPU\n"); // 从GPU调用
#else
    printf("CPU\n"); // 从CPU调用
#endif
} // 一个函数可以通过__CUDA_ARCH宏在CPU和GPU上重载

__global__ void kernel() {
    say_hello();
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    say_hello();
    return 0;
}
```
`__CUDA_ARCH__`这个宏是本设备的版号，且只有**定义了(#ifdef)**才可以使用
```cmake
set(CMAKE_CUDA_ARCHITECTURES 52;61;75;86;89)
```
CMakeLists.txt中需要加上自己设备的版号

---
## 板块 ##
* 一个板块有多个线程，线程<板块<网格
* `blockDim`板块中线程数量，`gridDim`板块数量，`threadIdx`线程在板块中的编号，`blockIdx`板块编号，
* <<<gridDim, blockDim>>>
* `checkCudaErrors(cudaFunc())`可以检查错误，需要引入`helper_cuda.h`头文件
* `cudaMalloc(void **ptr, size_t n)`在GPU上分配空间，CPU无法访问
`cudaMemcpy(void *dst, const void *src, size_t cnt, enum cudaMemcpyKind kind)`在不同端之间拷贝数据，会自动同步
`cudaFree(void *ptr)`释放GPU上的空间
* `cudaMallocManaged(void **ptr, size_t n)`统一内存地址技术，CPU和GPU都能访问，需要同步

---
多个线程并行处理数组
```C++
__global__ void kernel(int *arr, int n) {
    int i = threadIdx.x;
    arr[i] = i;
}

int main() {
    int n = 32, *arr;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    kernel<<<1, n>>>(arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(arr);
    ...
}
```

网格跨步循环
```C++
for(int i = threadIdx.x; i < n; i += blockDim.x) {
    ...
} // 例子：处理长度为8的数组，只分配4个线程，两轮就执行结束
```

---
## 从线程到板块 ##
当需要处理的数据`n`远远大于硬件的线程数`blockDim`时，可以用`n / blockDim`作为`gridDim`；但是当`n`不能整除`blockDim`时就会向下取整从而导致有余下的数据无法使用线程，有两种解决办法，一是**向上取整**，即`gridDim = (n + blockDim - 1) / blockDim`，二还是**采用网格跨步循环**，如下：
```C++
for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    ...
} // 一次跨步为总的线程数量
```

---
## 在GPU上使用vector ##
vector的构造具有隐含的第二个模板参数，`std::vector<T, std::allocator<T>>`，vector会调用`std::allocator<T>`的`allocate()`/`deallocate()`函数，用以分配和释放内存。如果可以自定义一套基于GPU的allocator，那就可以在GPU上使用vector了，如下：
```C++
template <class T>
struct cudaAllocator {
    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }
    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }
    template <class ...Args>
    /**
    * @brief
    * vector在初始化时会对所有元素进行无参构造，但此过程是在CPU上进行的
    * 故在本例中要禁用，通过给allocator添加construct成员函数来改变vector的构造
    * 如果有参数且是传统类型(如int、char等)，则跳过其无参构造
    */
    void construct(T *p, Args &&...agrs) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void*)p) T(std::forward<Args>(args)...);
    }
}
```

---
## 核函数也可以是模板函数 ##
```C++
template <int N, class T>
__global__ void kernel(T *arr) {...}

kernel<n><<<32, 128>>>(arr.data());
// 调用时能自动通过arr的类型推出第二个模板参数类型
```
## 核函数也可以接受仿函数 ##
```C++
template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

struct MyFunctor { // 仿函数：其对象可以被调用
    __device__ void operator()(int i) const {...}
}

int main() {
    int n = 65536;
    std::vector<int, cudaAllocator<int>> arr(n);

    parallel_for<<<32, 128>>>(n, MyFunctor{}); // 模板参数Func可由仿函数自动推导
    ...
}
```

也可以直接使用lambda，前提是在CMakeLists.txt文件中添加下面这句
```cmake
target_compile_options(TARGET PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
```

```C++
parallel_for<<<32, 128>>>(n, [] __device__ (int i) {...});
```
但是有一点需要注意，如果在lambda中意欲使用`[&]`传引用捕获外部变量比如`arr`，虽然其空间被分配在GPU上，但其本身的地址依旧在CPU的堆栈内，此时捕获的只能是`arr`变量本身

为了防止vector的深拷贝，可以使用`[=]`传值捕获：
```C++
int *arr_data = arr.data(); // data()返回一个起始地址，指针可以被浅拷贝
parallel_for<<<32, 128>>>(n, [=] __device__ (int i) { arr_data[i] = i; });
```

又或者可以更简便地在`[]`中写自由捕获表达式，使用同一变量名(`arr`由外部的vector指针退化为lambda内部指向`arr`起始地址的指针)
```C++
parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__ (int i) { arr[i] = i; });
```

测试用例：x[i] = a * x[i] + y[i]
```C++
    int n = 1 << 25;
    float a = 3.14f;
    std::vector<float> x1(n), y1(n);
    std::vector<float, cudaAllocator<float>> x2(n), y2(n);

    for(int i = 0; i < n; i++) {
        x1[i] = std::rand() * (1.f / RAND_MAX);
        y1[i] = std::rand() * (1.f / RAND_MAX);
        x2[i] = std::rand() * (1.f / RAND_MAX);
        y2[i] = std::rand() * (1.f / RAND_MAX);
    }

    TICK(cpu);
    for(int i = 0; i < n; i++) {
        x1[i] = a * x1[i] + y1[i];
    }
    TOCK(cpu);

    TICK(gpu);
    parallel_for<<<n / 512, 128>>>(n, [a, x2 = x2.data(), y2 = y2.data()] __device__ (int i) {
        x2[i] = a * x2[i] + y2[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(gpu);
```

---
## thrust 库 ##
* `universal_vector`统一地址分配，`host_vector`内存分配，`device_vector`显存分配。可以通过`=`运算符在`host_vector`与`device_vector`之间拷贝数据
* `generate()`用于在容器内批量生成一系列数字，例如：
    ```C++
    thrust::generate(vector_host.begin(), vector_host.end(), [] {
        return std::rand() * (1.f / RAND_MAX)
    });
    ```
* `for_each()`可以批量修改容器内的数据，例如：
    ```C++
    thrust::for_each(vector_device.begin(), vector_device.end(),
        [] __device__ (float &x) { x += 100.f; });

    thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
        [] __device__ (int i) { std::cout << i << std::endl; }); // 相当于for区间
    ```
类似的还有很多标准库函数

---
## 原子操作 ##
众所周知，被`__global__`修饰的核函数不能返回值，但可以用指针获取函数内求的结果，比如：
```C++
__global__ void parallel_sum(int *sum, const int *arr, int n) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum[0] += arr[i];
    }
}

int main() {
    int n = 65536;
    std::vector<int, cudaAllocator<int>> arr(n), sum(1);

    parallel_sum<<<n / 512, 128>>>(sum.data(), arr.data(), n);

    ...
}
```
但是，学过汇编的都知道，`sum[0] += arr[i]`这个操作会被分为四步，分别是1.取出`sum[0]`的值到寄存器A 2.取出`i`以及`arr[i]`的值到寄存器B 3.将`sum[0]`与`arr[i]`相加并存入寄存器B 4.将值写回寄存器A。由于使用多线程运行，不可避免会出现取到旧值的情况，所以要为这个操作**上锁**，即使用原子操作：
* `atomicAdd(&sum[0], arr[i]);`

`aomicAdd()`会返回旧值，相应的也会有其他的运算，如加减、异或与和最大最小值
但是这样也就让并行变为了串行，如果使用老版本cuda，这样做与在cpu中直接计算并无差距

* 解决办法之一：TLS(Thread Local Storage)
```C++
__global__ void parallel_sum(int *sum, const int *arr, int n) {
    int local_sum = 0;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    } atomicAdd(&sum[0], local_sum);
}

int main() {
    ...
    parallel_sum<<<n / 2056, 128>>>(sum.data(), arr.data(), n);
    ...
}
```
举个例子，n = 8192，那么此时的`gridDim`就是4，而`blockDim`则为128，即有4个板块，每个板块有128个线程，总共512个线程，**执行512次原子操作**，每个线程最多进行16次迭代即可退出循环（事实上也只有位于板块0的线程0才会完成最多的16次迭代）；而假如将原子操作放在循环体内，如`atomicAdd(&sum[0], arr[i]);`，那么会完完整整执行8192次原子操作。使用TLS可以减少`2056 / 128 = 16`倍的原子操作数，也就意味着要尽可能地使`blockDim * gridDim`远小于`n`

* 还有一种办法无需原子操作：
```C++
__global__ void parallel_sum(int *sum, const int *arr, int n) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum = 0;
        for(int j = i * 1024; j < (i + 1) * 1024; j++) {
            local_sum += arr[j];
        } // 每个线程负责原数组中相邻1024个元素的求和
        sum[i] = local_sum;
    }
}

int main() {
    ...
    std::vector<int, cudaAllocator<int>> sum(n / 1024); // 开比n小1024倍的辅助数组
    ...
    parallel_sum<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n); // 启动n / 1024个线程
    ...
    int final_sum = 0;
    for(int i = 0; i < n / 1024; i++) {
        final_sum += sum[i];
    }
    ...
}
```
核心思想就是使用一个比原数组小x倍的辅助数组，开共n/x个线程，每个线程串行处理x个数据，将结果储存到辅助数组中，最后cpu再串行地处理辅助数组即可（辅助数组的大小在接受范围内）
