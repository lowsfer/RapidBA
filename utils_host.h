/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

//
// Created by yao on 9/10/17.
//

#pragma once
#include "cuda_hint.cuh"
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cstdlib>
#include <new>
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <algorithm>
#include <functional>
#include <cassert>
#include <queue>
#include <unordered_set>
#include <boost/numeric/conversion/cast.hpp>
#include <mutex>
#include <atomic>
#include <variant>

// unit tests require (USE_MANAGED_MEM == 1), but USE_MANAGED_MEM=0 is fine for public API / demo
#define USE_MANAGED_MEM 1

//@fixme: change all host integer casts to numeric_cast
using boost::numeric_cast;
template <typename Source>
inline uint32_t cast32u(Source src) {
    return boost::numeric_cast<uint32_t, Source>(src);
}

inline void checkCudaErrors(cudaError_t error){
    if(error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(error));
    }
}

#define checkEarlyReturn(expr) do{cudaError_t tmpErrCode = (expr); if (tmpErrCode != cudaSuccess) {checkCudaErrors(tmpErrCode); return tmpErrCode;}}while(false)

struct CudaDeviceDeleter {
    void operator()(void* p){
        cudaFree(p);
    }
};

struct CudaHostDeleter {
    void operator()(void* p){
        cudaFreeHost(p);
    }
};

// allocated memory is not initialized
template <typename T>
std::unique_ptr<T, CudaDeviceDeleter> deviceAlloc(size_t nbElems){
    void* ptr = nullptr;
    if (nbElems != 0) {
        checkCudaErrors(cudaMalloc(&ptr, sizeof(T)*nbElems));
    }
    return std::unique_ptr<T, CudaDeviceDeleter>{static_cast<T*>(ptr)};
}

// allocated memory is not initialized
template <typename T>
std::unique_ptr<T, CudaDeviceDeleter> managedAlloc(size_t nbElems){
    void* ptr = nullptr;
    if (nbElems != 0) {
        checkCudaErrors(cudaMallocManaged(&ptr, sizeof(T)*nbElems));
    }
    return std::unique_ptr<T, CudaDeviceDeleter>{static_cast<T*>(ptr)};
}

// allocated memory is not initialized
template <typename T>
std::unique_ptr<T, CudaHostDeleter> hostAlloc(size_t nbElems){
    void* ptr = nullptr;
    if (nbElems != 0) {
        checkCudaErrors(cudaMallocHost(&ptr, sizeof(T)*nbElems));
    }
    return std::unique_ptr<T, CudaHostDeleter>{static_cast<T*>(ptr)};
}

struct CudaStreamDeleter {
    void operator()(cudaStream_t s){
        cudaStreamDestroy(s);
    }
};

inline std::unique_ptr<CUstream_st, CudaStreamDeleter> makeCudaStream(){
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return std::unique_ptr<CUstream_st, CudaStreamDeleter>{stream};
}

struct CudaEventDeleter {
    void operator()(cudaEvent_t e){
        cudaEventDestroy(e);
    }
};

inline std::unique_ptr<CUevent_st, CudaEventDeleter> makeCudaEvent(unsigned flags = cudaEventBlockingSync | cudaEventDisableTiming){
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreateWithFlags(&event, flags));
    return std::unique_ptr<CUevent_st, CudaEventDeleter>{event};
}


template <class T>
struct CudaHostAllocator {
    typedef T value_type;
    CudaHostAllocator(){}
    template <class U>
    constexpr CudaHostAllocator(const CudaHostAllocator<U>& other) noexcept{}
    T* allocate(std::size_t n) {
        return hostAlloc<T>(n).release();
    }
    void deallocate(T* ptr, std::size_t) noexcept { cudaFreeHost(ptr); }
};
template <class T, class U>
bool operator==(const CudaHostAllocator<T>& a, const CudaHostAllocator<U>& b) { return true; }
template <class T, class U>
bool operator!=(const CudaHostAllocator<T>& a, const CudaHostAllocator<U>& b) { return false; }

template <class T>
struct CudaManagedAllocator {
    typedef T value_type;
    CudaManagedAllocator(){}
    template <class U>
    constexpr CudaManagedAllocator(const CudaManagedAllocator<U>& other) noexcept{}
    T* allocate(std::size_t n) {
        return managedAlloc<T>(n).release();
    }
    void deallocate(T* ptr, std::size_t) noexcept { cudaFree(ptr); }
};
template <class T, class U>
bool operator==(const CudaManagedAllocator<T>& a, const CudaManagedAllocator<U>& b) { return true; }
template <class T, class U>
bool operator!=(const CudaManagedAllocator<T>& a, const CudaManagedAllocator<U>& b) { return false; }


//void CUDART_CB callback_functor(cudaStream_t stream, cudaError_t status, void *data);
//template<typename Func>
//void stream_add_callback(cudaStream_t stream, Func&& func){
//    auto callback = new std::function<void()>{std::forward<Func>(func)};
//    const cudaError_t err = cudaStreamAddCallback(stream, callback_functor, callback, 0);
//    if(err != cudaSuccess)
//        delete callback;
//    checkCudaError(err);
//};

inline void require(bool val){
    if(!val)
        throw std::runtime_error("assertion failure");
}

template<typename T>
void copyPitched(const T *src, size_t pitch_src, T *dst, size_t pitch_dst, size_t width, size_t height){
    for(size_t i = 0; i < height; i++){
        std::copy_n(src, width, dst);
        src = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(src) + pitch_src);
        dst = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(dst) + pitch_dst);
    }
}

// k-way merge and remove duplicates. Input data with multiple (nbRanges) sorted ranges is stored in src.
template <typename T>
void mergeUnique(std::vector<T>& dst, T* src, const std::vector<uint32_t>& ranges){
    assert(!ranges.empty());
    const auto nbRanges = uint32_t(ranges.size() - 1);
    std::vector<bool> nonEmptyRanges(nbRanges, false);
    uint32_t nbNonEmptyRanges = 0;
    for (unsigned i = 0; i < nbRanges; i++){
        if (ranges[i] != ranges[i+1]){
            nonEmptyRanges[i] = true;
            nbNonEmptyRanges++;
        }
    }

    struct Item{
        T data;
        uint32_t idxSrc;
        bool operator>(const Item& other) const{return data > other.data;}
    };
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;

    std::vector<uint32_t> rangePos(ranges.begin(), ranges.begin() + nbRanges);
    const auto rangeEnd = ranges.begin() + 1;
    std::unordered_set<T> heapElems;
    heapElems.reserve(nbRanges);
    auto heapNewItem = [&](uint32_t idxSrc) {
        uint32_t &pos = rangePos[idxSrc];
        const uint32_t posEnd = rangeEnd[idxSrc];
        while (pos != posEnd) {
            const T data = src[pos];
            if ((dst.empty() || data != dst.back()) && heapElems.find(data) == heapElems.end()){
                heap.emplace(Item{data, idxSrc});
                heapElems.emplace(data);
                assert(heap.size() == heapElems.size());
                break;
            }
            pos++;
        }
    };
    auto mergeItem = [&]() -> uint32_t {
        Item top = heap.top();
        dst.emplace_back(std::move(top.data));
        heapElems.erase(top.data);
        heap.pop();
        return top.idxSrc;
    };

    // initialize
    for (unsigned i = 0; i < nbRanges; i++){
        heapNewItem(i);
    }
    dst.clear();
    //merge
    while(!heap.empty()) {
        const uint32_t idxSrcMerged = mergeItem();
        heapNewItem(idxSrcMerged);
    }
};

inline int getCudaDevice() {
    int id;
    checkCudaErrors(cudaGetDevice(&id));
    return id;
};
inline cudaError_t prefetchManagedMem(const void* ptr, size_t bytes, int dstDevice, cudaStream_t stream)
{
    if (ptr == nullptr){
        assert(bytes == 0);
        return cudaSuccess;
    }
    return cudaMemPrefetchAsync(ptr, bytes, dstDevice, stream);
}

template <typename T>
class DevVector{
public:
    using Value = T;
    DevVector() = default;
    explicit DevVector(size_t size) {resize(size);}
    explicit DevVector(const std::vector<T>& hostData) {
        assign(hostData);
    }
    void assign(const std::vector<T>& hostData) {
        resize(hostData.size());
        checkCudaErrors(cudaMemcpy(mData.get(), hostData.data(), sizeof(T) * mSize, cudaMemcpyHostToDevice));
    }
    std::vector<T> getHostCopy() const {
        std::vector<T> hostCopy(mSize);
        checkCudaErrors(cudaMemcpy(hostCopy.data(), data(), sizeof(T) * size(), cudaMemcpyDeviceToHost));
        return hostCopy;
    }
    Value* data() const {return mData.get();}
    size_t size() const {return mSize;}
    Value* dataEnd() const {return data() + size();}
    bool empty() const {return size() == 0;}
    // We don't preserve data when resizing
    void resize(size_t size) {
        mData = deviceAlloc<T>(size);
        mSize = size;
    }
    void clear() {
        resize(0);
    }
private:
    size_t mSize{0};
    std::unique_ptr<T, CudaDeviceDeleter> mData;
};

#if USE_MANAGED_MEM || defined(__CUDACC__)
template <typename T>
class MngVector : public std::vector<T, CudaManagedAllocator<T>>{
public:
    using std::vector<T, CudaManagedAllocator<T>>::vector;
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const{
        if (this->empty())
            return cudaSuccess;
        return prefetchManagedMem(this->data(), sizeof(T) * this->size(), deviceId, stream);
    }

    MngVector<T> getHostCopy() const {
        const cudaStream_t stream = nullptr;
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(migrateToDevice(cudaCpuDeviceId, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        
        return MngVector<T> (this->begin(), this->end());
    }
};
#else
template <typename T>
class FlexVector{
public:
    using value_type = T;
    template <typename... Args>
    explicit FlexVector(Args&&... args) : mData{std::vector<T>(std::forward<Args>(args)...)}{}
    explicit FlexVector(const std::initializer_list<T>& x) : mData(std::vector<T>(x)){}

    void resize(size_t size) {hostVec().resize(size);}
    void push_back(const T& val) {hostVec().push_back(val);}
    template <typename... Args>
    void emplace_back(Args&&... args) {hostVec().emplace_back(std::forward<Args>(args)...);}
    T* hostData() {return hostVec().data();}
    const T* hostData() const {return hostVec().data();}
    T* devData() const {return devVec().data();}
    const T* data() const {return getLoc() == Location::kHost ? hostData() : devData();}
    T* data() {return getLoc() == Location::kHost ? hostData() : devData();}

    size_t size() const {
        return getLoc() == Location::kHost ? hostVec().size() : devVec().size();
    }
    size_t empty() const {return size() == 0;}
    void clear() {mData = std::vector<T>{};}
    void reserve(size_t capacity) {hostVec().reserve(capacity);}

    T& at(size_t i) {return hostVec().at(i);}
    T& operator[](size_t i){return hostVec()[i];}
    const T& at(size_t i) const {return hostVec().at(i);}
    const T& operator[](size_t i) const {return hostVec()[i];}

    T& back() {return hostVec().back();}
    const T& back() const {return hostVec().back();}
    using iterator = typename std::vector<T>::iterator;
    iterator begin() {return hostVec().begin();}
    iterator end() {return hostVec().end();}
    template <typename Iter>
    void insert(iterator pos, Iter first, Iter last) {
        hostVec().insert(pos, first, last);
    }

    enum class Location{kHost, kGpu};
    Location getLoc() const { return mData.index() == 0 ? Location::kHost : Location::kGpu; }
    void assertLoc(Location loc = Location::kHost) const {assert(getLoc() == loc);}

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const{
        require(getCudaDevice() == deviceId || cudaCpuDeviceId == deviceId);
        const Location dstLoc = (deviceId == cudaCpuDeviceId ? Location::kHost : Location::kGpu);
        if (this->empty() || getLoc() == dstLoc)
            return cudaSuccess;
        require(dstLoc == Location::kGpu); // migrating back to CPU invalidates device pointers. May not be safe.
        checkCudaErrors(cudaStreamSynchronize(stream));
        if (getLoc() == Location::kHost) {
            std::vector<T> hostData = std::move(hostVec());
            mData = DevVector<T>(hostData);
        }
        else {
            mData = devVec().getHostCopy();
        }
        return cudaSuccess;
    }

    std::vector<T>& hostVec(){ return std::get<std::vector<T>>(mData);}
    const std::vector<T>& hostVec() const { return std::get<std::vector<T>>(mData);}
    DevVector<T>& devVec(){ return std::get<DevVector<T>>(mData);}
    const DevVector<T>& devVec() const{ return std::get<DevVector<T>>(mData);}

    FlexVector<T> getHostCopy() const {
        return FlexVector<T>{(getLoc() == Location::kHost) ? hostVec() : devVec().getHostCopy()};
    }
private:
    mutable std::variant<std::vector<T>, DevVector<T>> mData;
};

template <typename T>
using MngVector = FlexVector<T>;
#endif

class StreamPool
{
public:
    static constexpr uint32_t nbWorkers = 4U;

    class WorkerStream
    {
    public:
        WorkerStream(const StreamPool& pool, uint32_t idx) : mPool{&pool}, mIndex{idx} {
            mPool->syncMasterWorker(idx);
        }
        WorkerStream(const WorkerStream&) = delete;
        WorkerStream& operator=(const WorkerStream&) = delete;
        WorkerStream(WorkerStream&& src): mPool{src.mPool}, mIndex{src.mIndex} { src.mPool = nullptr; }
        WorkerStream& operator=(WorkerStream&& src) {
            mPool = src.mPool;
            mIndex = src.mIndex;
            src.mPool = nullptr;
            return *this;
        }
        ~WorkerStream() {if (mPool != nullptr) mPool->syncWorkerMaster(mIndex);}
        cudaStream_t get() const {
            if (mPool != nullptr){
                throw std::runtime_error("invalid WorkerStream");
            }
            return mPool->worker(mIndex);
        }

    private:
        const StreamPool* mPool;
        uint32_t mIndex;
    };

    WorkerStream getWorker() { return {*this, mIdxNext.fetch_add(1u, std::memory_order_relaxed) % nbWorkers}; }

private:
    cudaStream_t master() const { return mMaster.get(); }
    cudaStream_t worker(uint32_t idx) const { return workers[idx % workers.size()].get(); }
    void syncMasterWorker(uint32_t idx) const{
        std::lock_guard<std::mutex> lock(mLock);
        checkCudaErrors(cudaEventRecord(mSyncEvent.get(), master()));
        checkCudaErrors(cudaStreamWaitEvent(worker(idx), mSyncEvent.get(), 0));
    }
    void syncWorkerMaster(uint32_t idx) const{
        std::lock_guard<std::mutex> lock(mLock);
        checkCudaErrors(cudaEventRecord(mSyncEvent.get(), worker(idx)));
        checkCudaErrors(cudaStreamWaitEvent(master(), mSyncEvent.get(), 0));
    }
private:
    mutable std::mutex mLock;
    std::unique_ptr<CUstream_st, CudaStreamDeleter> mMaster = makeCudaStream();
    std::array<std::unique_ptr<CUstream_st, CudaStreamDeleter>, nbWorkers> workers = {
            makeCudaStream(),
            makeCudaStream(),
            makeCudaStream(),
            makeCudaStream()
    };
    std::unique_ptr<CUevent_st, CudaEventDeleter> mSyncEvent = makeCudaEvent();
    std::atomic_uint32_t mIdxNext {0u};
};

#if __CUDACC_VER_MAJOR__ >= 10
struct CudaGraphDeleter{
    void operator()(cudaGraph_t g){
        cudaGraphDestroy(g);
    }
};
using UniqueCudaGraph = std::unique_ptr<CUgraph_st, CudaGraphDeleter>;

inline UniqueCudaGraph makeCudaGraph(){
    cudaGraph_t g;
    checkCudaErrors(cudaGraphCreate(&g, 0));
    return UniqueCudaGraph{g};
}

struct CudaGraphExecDeleter{
    void operator()(cudaGraphExec_t g){
        cudaGraphExecDestroy(g);
    }
};
using UniqueCudaGraphExec = std::unique_ptr<CUgraphExec_st, CudaGraphExecDeleter>;

inline UniqueCudaGraphExec instCudaGraph(cudaGraph_t g){
    cudaGraphExec_t inst = nullptr;
    cudaGraphNode_t node = nullptr;
    const size_t logBuffSize = 128;
    char logBuff[logBuffSize];
    checkCudaErrors(cudaGraphInstantiate(&inst, g, &node, logBuff, logBuffSize));
    return UniqueCudaGraphExec{inst};
}

#endif
