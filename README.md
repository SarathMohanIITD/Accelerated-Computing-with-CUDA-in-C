# Accelerated Computing with CUDA in C
   In this we will be looking at some basic commands for programming in CUDA for creating acceleratied appliccation. So lets see what is cuda.
 CUDA is a parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of the GPU. 
 
 (The commands listed here are for the jupyter notebook environment)
> For retrieving the NVIDIA GPU info - `!nvidia-smi`
### Points to note when writting a GPU code
- The gpu accelerated fucntons are saved with an extension > "**.cu**"
- The gpu functions should be  always declared as a global function `__global__ void GPUFunction  `<br />One thing to note here is the return type. The return type should be -void- since it is declared as global
- During the function call it should follow the syntax `GPUFunction<<<1, 1>>>();` <br /> We will be looking into this syntax in detail.But, basic idea is when we are calling a gpu function (which we call as kernel) `<<< >>>` provides an execution config for the GPU.
- When you are executing a GPU function, it happens asynchronously with the CPU process. So we need to have something which will ensure the synchronization between CPU and GPU process. `cudaDeviceSynchronize();` will ensure this. Add this immidiately after you call a GPU function.
- Putting all together, we can write a gpu code as given below 
``` 
#include <stdio.h>
__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{

  helloGPU<<<1, 1>>>();

  cudaDeviceSynchronize();
} 
```
### Launching Parallel Kernels
- Earlier we saw how to call a GPU fucntion . `helloGPU<<<1, 1>>>();`. So what does <<< >>> actually do? The execution configuration allows programmers to specifiy how many groups of threads - called thread blocks, or just blocks - and how many threads they would like each thread block to contain.
- `<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>` where BLOCKS is a collection of threads and each blocks contain equal amount of threads.

#### Accelerating for loops
- The following codes will show the effectiveness of using GPU instead of CPU for a for loop.
```
#include <stdio.h>

void CPUloop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{

  int N = 10;
  CPUloop(N);
}
```
Now lets see how we can use GPUs to accelerate this for loop

```
#include <stdio.h>

__global__ void loop()
{

  printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{

  loop<<<1, 10>>>();
  cudaDeviceSynchronize();
}
```
- Notice the `threadIdx.x` which was used in the above code
- CUDA kernels have access to special variables identifying both the index of the thread (within the block) that is executing the kernel, and, the index of the block (within the grid) that the thread is within. These variables are **threadIdx.x** and **blockIdx.x** respectively.

### More Parallelization
- The maximum number of blocks that can be inside a block is 1024. In order to increase the amount of parallelism in accelerated applications, we must be able to coordinate among multiple thread blocks.
- `blockdim.x` says how many threads are there in a block. With this we can find the unique indexing for the threads < br/> `threadIdx.x + blockIdx.x * blockDim.x`
- Lets see an example code for more clarity
```
#include <stdio.h>

__global__ void loop()
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

int main()
{

  loop<<<2, 5>>>();
  cudaDeviceSynchronize();
}
```
### Dynamic memory allocation
We can allocate memory such that it can be accessed from GPU as well as CPU by which the data tranfer process will occur much faster
- Dynamic memory allocation in the CPU `a = (int *)malloc(size);`
- Shared memory allocation `cudaMallocManaged(&a, size);`
- 
### Data Sets Larger Than the Grid
- These days it is so often to see the datasets which has more than the total number of threads . So we nned something that can make a loop and reuse the threads when current procrss is over. This is known a **Grid Stride Loop**
- Lets see an example code which is used to doubles element values in an arrya using Grid Stride Loop 
```
#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}


int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);  // Allocationg shared memory

  init(a, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);

  cudaFree(a);  //To free the allocated memory
}
```










