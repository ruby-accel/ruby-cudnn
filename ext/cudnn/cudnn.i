%module "cudnn"

%include std_vector.i

%{
#include <cuda_runtime.h>
#include "cudnn.h"
%}

%template(StdVectorInt)    std::vector<int>;
%template(StdVectorDouble) std::vector<double>;
%template(StdVectorFloat)  std::vector<float>;

%inline %{
 
  namespace RubyCuDNN {

    typedef enum {
        CUDNN_DATA_FLOAT  = ::CUDNN_DATA_FLOAT,
        CUDNN_DATA_DOUBLE = ::CUDNN_DATA_DOUBLE,
        CUDNN_DATA_HALF   = ::CUDNN_DATA_HALF,
      } cudnnDataType_t;

    typedef enum {
      CUDNN_NOT_PROPAGATE_NAN  = ::CUDNN_NOT_PROPAGATE_NAN,
      CUDNN_PROPAGATE_NAN      = ::CUDNN_PROPAGATE_NAN,
    } cudnnNanPropagation_t;

    typedef enum
      {
        CUDNN_TENSOR_NCHW = ::CUDNN_TENSOR_NCHW,   /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC = ::CUDNN_TENSOR_NHWC    /* feature maps interleaved ( cStride = 1 )*/
      } cudnnTensorFormat_t;

    typedef enum
      {
        CUDNN_CONVOLUTION       = ::CUDNN_CONVOLUTION,
        CUDNN_CROSS_CORRELATION = ::CUDNN_CROSS_CORRELATION
      } cudnnConvolutionMode_t;

    typedef enum
      {
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = ::CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = ::CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = ::CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
      } cudnnConvolutionFwdPreference_t;


    typedef enum
      {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = ::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = ::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = ::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = ::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = ::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = ::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
      } cudnnConvolutionFwdAlgo_t;

    typedef enum
      {
        CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = ::CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = ::CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = ::CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
      } cudnnConvolutionBwdFilterPreference_t;

    typedef enum
      {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = ::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,  // non-deterministic
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = ::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = ::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3         = ::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3   // non-deterministic, algo0 with workspace
      } cudnnConvolutionBwdFilterAlgo_t;

    typedef enum
      {
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE             = ::CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST           = ::CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = ::CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
      } cudnnConvolutionBwdDataPreference_t;
    
    typedef enum
      {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0          = ::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, // non-deterministic
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1          = ::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT        = ::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = ::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
      } cudnnConvolutionBwdDataAlgo_t;

    typedef enum
      {
        CUDNN_SOFTMAX_FAST     = ::CUDNN_SOFTMAX_FAST,
        CUDNN_SOFTMAX_ACCURATE = ::CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_LOG      = ::CUDNN_SOFTMAX_LOG
      } cudnnSoftmaxAlgorithm_t;

    typedef enum
      {
        CUDNN_SOFTMAX_MODE_INSTANCE = CUDNN_SOFTMAX_MODE_INSTANCE,
        CUDNN_SOFTMAX_MODE_CHANNEL  = CUDNN_SOFTMAX_MODE_CHANNEL
      } cudnnSoftmaxMode_t;

    typedef enum
      {
        CUDNN_POOLING_MAX     = ::CUDNN_POOLING_MAX,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = ::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = ::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
      } cudnnPoolingMode_t;

    size_t GetVersion(){
      return ::cudnnGetVersion();
    }
    void error_check(cudnnStatus_t s) { }
    void error_check(cudaError_t s) { }

    void setDevice(int n){
      error_check( cudaSetDevice(n) );
    }

    template<class T>
    class DeviceVector {
    public:
      T* ptr;
      int size;
      int bytesize;

      DeviceVector(int n) {
        size = n;
        bytesize = n*sizeof(T);
        error_check( cudaMalloc((void **)&ptr, sizeof(T) * n) );
      }

      DeviceVector(const std::vector<T>& v) {
        size = v.size();
        error_check( cudaMalloc((void **)&ptr, sizeof(T) * v.size()) );
        error_check( cudaMemcpy((void *)ptr, (void *)v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice) );
      }

      ~DeviceVector() {
        cudaFree(ptr);
      }

      std::vector<T> to_hostvec() {
        std::vector<T> v(size);
        error_check( cudaMemcpy((void *)v.data(), (void *)ptr, sizeof(T) * size, cudaMemcpyDeviceToHost) );
        return v;
      }
    };
    typedef DeviceVector<double> DeviceVectorDouble;
    typedef DeviceVector<float> DeviceVectorFloat;

    const char* GetErrorString(cudnnStatus_t status) {
      return ::cudnnGetErrorString(status);
    }

    cudaStream_t* StreamCreate() {
      cudaStream_t* pStream=0;
      error_check( cudaStreamCreate( pStream ) );
      return pStream;
    }

    class TensorDescriptor {
    public:
      ::cudnnTensorDescriptor_t desc;
      int degree;
      bool with_stride;
      std::vector<int> dim;
      cudnnDataType_t dtype;
      cudnnTensorFormat_t format;

      TensorDescriptor(int n, int c, int h, int w,
                       cudnnTensorFormat_t format, cudnnDataType_t dataType) :
        degree(4), with_stride(false), dim(4) {
        dim[0] = n, dim[1] = c, dim[2] = h, dim[3] = w;
        error_check( cudnnCreateTensorDescriptor(&desc) );
        error_check( cudnnSetTensor4dDescriptor(desc, (::cudnnTensorFormat_t ) format,
                                                (::cudnnDataType_t) dataType,
                                                n, c, h, w ) );
      }

      TensorDescriptor(int n, int c, int h, int w,
                       int nStride, int cStride, int hStride, int wStride,
                       cudnnDataType_t dataType ) {
        degree = 4;
        with_stride = true;
        error_check( cudnnCreateTensorDescriptor(&desc) );
        error_check( cudnnSetTensor4dDescriptorEx(desc, (::cudnnDataType_t) dataType,
                                                  n,  c,  h,  w,
                                                  nStride,  cStride,  hStride,  wStride) );
      }

      TensorDescriptor(int nbDims, std::vector<int> dimA, std::vector<int> strideA, cudnnDataType_t dataType) {
        degree = nbDims;
        with_stride = true;
        error_check( cudnnCreateTensorDescriptor(&desc) );
        error_check( cudnnSetTensorNdDescriptor(desc, (::cudnnDataType_t) dataType,
                                                nbDims,
                                                dimA.data(),
                                                strideA.data()) );
      }

      ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc);
      }

      std::vector<int> shape() {
        std::vector<int> v(8);
        ::cudnnDataType_t dummy;
        cudnnGetTensor4dDescriptor(desc, &dummy,
                                   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
        return v;
      }
    };

    class FilterDescriptor {
    public:
      ::cudnnFilterDescriptor_t desc;
      int degree;
      std::vector<int> filterDim;
      cudnnTensorFormat_t format;
      cudnnDataType_t dtype;

      FilterDescriptor(int k, int c, int h, int w,
                       cudnnTensorFormat_t fmat, cudnnDataType_t dataType) :
        degree(4), filterDim(4), format(fmat), dtype(dataType) {
        filterDim[0] = k, filterDim[1] = c, filterDim[2] = h, filterDim[3] = w;        
        error_check( cudnnCreateFilterDescriptor(&desc) );
        error_check( cudnnSetFilter4dDescriptor_v4(desc,
                                                   (::cudnnDataType_t) dataType,
                                                   (::cudnnTensorFormat_t ) fmat,
                                                   k, c, h, w) );
      }

      FilterDescriptor(int nbDims, const std::vector<int>& filterDimA, 
                       cudnnTensorFormat_t fmat, cudnnDataType_t dataType) :
        degree(nbDims), filterDim(nbDims), format(fmat), dtype(dataType) {
        filterDim = filterDimA;
        error_check( cudnnCreateFilterDescriptor(&desc) );
        error_check( cudnnSetFilterNdDescriptor_v4(desc, (::cudnnDataType_t) dataType, 
                                                   (::cudnnTensorFormat_t ) fmat,
                                                   nbDims, filterDimA.data()) );
      };

      ~FilterDescriptor() {
        cudnnDestroyFilterDescriptor(desc);
      }
    };

    class ConvolutionDescriptor {
    public:
      ::cudnnConvolutionDescriptor_t desc;
      int degree;
      std::vector<int> pad;
      std::vector<int> filterStride;
      std::vector<int> upscale;

      ConvolutionDescriptor(int pad_h, int pad_w,
                            int u, int v, int upscalex, int upscaley,
                            cudnnConvolutionMode_t mode) :
        pad(2), filterStride(2), upscale(2) {
        degree = 2;
        pad[0] = pad_h, pad[1] = pad_w, filterStride[0] = u, filterStride[1] = v;
        upscale[0] = upscalex, upscale[1] = upscaley;
        error_check( cudnnCreateConvolutionDescriptor(&desc) );
        error_check( cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, u, v, upscalex, upscaley,
                                                     (::cudnnConvolutionMode_t) mode) );
      }

      ConvolutionDescriptor(int arrayLength,
                            const std::vector<int>& padA,
                            const std::vector<int>& filterStrideA,
                            const std::vector<int>& upscaleA,
                            cudnnConvolutionMode_t mode, cudnnDataType_t dataType) : 
        pad(arrayLength), filterStride(arrayLength), upscale(arrayLength) {
        degree = arrayLength;
        pad = padA, filterStride = filterStrideA, upscale = upscaleA;
        error_check( cudnnCreateConvolutionDescriptor(&desc) );
        error_check( cudnnSetConvolutionNdDescriptor_v3(desc, arrayLength,
                                                        padA.data(),
                                                        filterStrideA.data(),
                                                        upscaleA.data(),
                                                        (::cudnnConvolutionMode_t) mode,
                                                        (::cudnnDataType_t) dataType) );
      }

      ~ConvolutionDescriptor() {
        cudnnDestroyConvolutionDescriptor(desc);
      }

      std::vector<int> getConvolution2dForwardOutputDim( const TensorDescriptor& inputTensorDesc,
                                                         const FilterDescriptor& filterDesc) {
        std::vector<int> v(4);
        error_check( cudnnGetConvolution2dForwardOutputDim(desc, inputTensorDesc.desc, filterDesc.desc,
                                                           &v[0], &v[1], &v[2], &v[3] ) );
        return v;
      }

    };

    class Handle {
    public:
      cudnnHandle_t handle;
      Handle(){ 
        error_check( cudnnCreate(&handle) ); 
      }
      ~Handle(){ 
        error_check( cudnnDestroy(handle) ); 
      }

      void setStream(cudaStream_t streamId) {
        error_check( ::cudnnSetStream(handle, streamId) );
      }

      cudaStream_t* getStream() {
        cudaStream_t *streamId=0;
        error_check( ::cudnnGetStream(handle, streamId) );
        return streamId;
      }
    };

    cudnnConvolutionFwdAlgo_t
    getConvolutionForwardAlgorithm(Handle h,
                                   const TensorDescriptor& xDesc,
                                   const FilterDescriptor& filDesc,
                                   const ConvolutionDescriptor& convDesc,
                                   const TensorDescriptor& yDesc,
                                   cudnnConvolutionFwdPreference_t pref,
                                   size_t memlimit) {
      ::cudnnConvolutionFwdAlgo_t algo;
      error_check( cudnnGetConvolutionForwardAlgorithm(h.handle, xDesc.desc, filDesc.desc,
                                                       convDesc.desc, yDesc.desc,
                                                       (::cudnnConvolutionFwdPreference_t) pref,
                                                       memlimit, &algo) );
      return (cudnnConvolutionFwdAlgo_t) algo;
    }

    size_t getConvolutionForwardWorkspaceSize(Handle h,
                                              const TensorDescriptor& xDesc,
                                              const FilterDescriptor& filDesc,
                                              const ConvolutionDescriptor& convDesc,
                                              const TensorDescriptor& yDesc,
                                              cudnnConvolutionFwdAlgo_t algo) {
      size_t s;
      error_check( cudnnGetConvolutionForwardWorkspaceSize(h.handle, xDesc.desc, filDesc.desc,
                                                           convDesc.desc, yDesc.desc,
                                                           (::cudnnConvolutionFwdAlgo_t) algo, &s) );
      return s;
    }

    void covolutionForward(Handle h, float alpha, float beta,
                           const TensorDescriptor& xDesc,
                           const DeviceVectorFloat& x,
                           const FilterDescriptor& filDesc,
                           const DeviceVectorFloat& filter,
                           const ConvolutionDescriptor& convDesc,
                           cudnnConvolutionFwdAlgo_t algo,
                           DeviceVectorFloat& workspace,
                           const TensorDescriptor& yDesc,
                           DeviceVectorFloat& y) {
      error_check( cudnnConvolutionForward(h.handle, &alpha,
                                           xDesc.desc, x.ptr,
                                           filDesc.desc, filter.ptr,
                                           convDesc.desc, (::cudnnConvolutionFwdAlgo_t) algo,
                                           workspace.ptr, workspace.bytesize,
                                           &beta,
                                           yDesc.desc, y.ptr) );
    }
    
    void addTensor(Handle h,float alpha, float beta,
                   const TensorDescriptor& bDesc,
                   const DeviceVectorFloat& b,
                   const TensorDescriptor& yDesc,
                   DeviceVectorFloat& y) {
      error_check( cudnnAddTensor_v3(h.handle, &alpha, bDesc.desc, b.ptr, &beta, yDesc.desc, y.ptr) );
    }
    
    void convolutionBackwardBias(Handle h,float alpha, float beta,
                                 const TensorDescriptor& dyDesc,
                                 DeviceVectorFloat& dy,
                                 const TensorDescriptor& dbDesc,
                                 DeviceVectorFloat& db) {
      error_check( cudnnConvolutionBackwardBias(h.handle, &alpha, dyDesc.desc, dy.ptr, &beta, dbDesc.desc, db.ptr) );
    }

    cudnnConvolutionBwdFilterAlgo_t
    getConvolutionBackwardFilterAlgorithm(Handle h,
                                          const TensorDescriptor& xDesc,
                                          const TensorDescriptor& dyDesc,
                                          const ConvolutionDescriptor& convDesc,
                                          const FilterDescriptor& filDesc,
                                          cudnnConvolutionBwdFilterPreference_t pref,
                                          size_t memlimit) {
      ::cudnnConvolutionBwdFilterAlgo_t algo;
      error_check( cudnnGetConvolutionBackwardFilterAlgorithm(h.handle, xDesc.desc, dyDesc.desc, convDesc.desc,
                                                              filDesc.desc,
                                                              (::cudnnConvolutionBwdFilterPreference_t) pref,
                                                              memlimit, &algo) );
      return (cudnnConvolutionBwdFilterAlgo_t) algo;
    }
    
    cudnnConvolutionBwdDataAlgo_t
    getConvolutionBackwardDataAlgorithm(Handle h,
                                        const FilterDescriptor& filDesc,
                                        const TensorDescriptor& dyDesc,
                                        const ConvolutionDescriptor& convDesc,
                                        const TensorDescriptor& dxDesc,
                                        cudnnConvolutionBwdDataPreference_t pref,
                                        size_t memlimit) {
      ::cudnnConvolutionBwdDataAlgo_t algo;
      error_check( cudnnGetConvolutionBackwardDataAlgorithm(h.handle, filDesc.desc, dyDesc.desc, convDesc.desc,
                                                            dxDesc.desc,
                                                            (::cudnnConvolutionBwdDataPreference_t) pref,
                                                            memlimit, &algo) );
      return (cudnnConvolutionBwdDataAlgo_t) algo;
    }

    size_t getConvolutionBackwardFilterWorkspaceSize(Handle h,
                                                     const TensorDescriptor& xDesc,
                                                     const TensorDescriptor& dyDesc,
                                                     const ConvolutionDescriptor& convDesc,
                                                     const FilterDescriptor& filDesc,
                                                     cudnnConvolutionBwdFilterAlgo_t algo) {
      size_t s;
      error_check( cudnnGetConvolutionBackwardFilterWorkspaceSize(h.handle, xDesc.desc, dyDesc.desc,
                                                                  convDesc.desc, filDesc.desc,
                                                                  (::cudnnConvolutionBwdFilterAlgo_t) algo,
                                                                  &s) );
      return s;
    }

    void convolutionBackwardFilter(Handle h,
                                   float alpha,
                                   const TensorDescriptor& xDesc,
                                   DeviceVectorFloat& x,
                                   const TensorDescriptor& dyDesc,
                                   DeviceVectorFloat& dy,
                                   const ConvolutionDescriptor& convDesc,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   DeviceVectorFloat& workspace,
                                   float beta,
                                   const FilterDescriptor& dfilDesc,
                                   DeviceVectorFloat& dfil) {
      error_check( cudnnConvolutionBackwardFilter_v3(h.handle, &alpha, xDesc.desc, x.ptr, dyDesc.desc, dy.ptr,
                                                     convDesc.desc, (::cudnnConvolutionBwdFilterAlgo_t) algo,
                                                     workspace.ptr, workspace.bytesize,
                                                     &beta,
                                                     dfilDesc.desc, dfil.ptr) );
    }

    void convolutionBackwardData(Handle h,
                                 float alpha,
                                 const FilterDescriptor& filDesc,
                                 DeviceVectorFloat& fil,
                                 const TensorDescriptor& dyDesc,
                                 DeviceVectorFloat& dy,
                                 const ConvolutionDescriptor& convDesc,
                                 cudnnConvolutionBwdDataAlgo_t algo,
                                 DeviceVectorFloat& workspace,
                                 float beta,
                                 const TensorDescriptor& dxDesc,
                                 DeviceVectorFloat& dx) {

      error_check( cudnnConvolutionBackwardData_v3(h.handle, &alpha,
                                                   filDesc.desc, fil.ptr, dyDesc.desc, dy.ptr,
                                                   convDesc.desc, (::cudnnConvolutionBwdDataAlgo_t ) algo,
                                                   workspace.ptr, workspace.bytesize, &beta,
                                                   dxDesc.desc, dx.ptr) );
    }

    void softmaxForward(Handle h,
                        cudnnSoftmaxAlgorithm_t algo,
                        cudnnSoftmaxMode_t mode,
                        float alpha,
                        const TensorDescriptor& xDesc,
                        DeviceVectorFloat& x,
                        float beta,
                        const TensorDescriptor yDesc,
                        DeviceVectorFloat& y) {
      error_check( cudnnSoftmaxForward(h.handle, (::cudnnSoftmaxAlgorithm_t) algo,
                                       (::cudnnSoftmaxMode_t) mode, &alpha,
                                       xDesc.desc, x.ptr, &beta,
                                       yDesc.desc, y.ptr) );
    }

    void softmaxBackward(Handle handle,
                         cudnnSoftmaxAlgorithm_t algo,
                         cudnnSoftmaxMode_t mode,
                         float alpha,
                         const TensorDescriptor& yDesc,
                         DeviceVectorFloat& y,
                         const TensorDescriptor& dyDesc,
                         DeviceVectorFloat& dy,
                         float beta,
                         const TensorDescriptor& dxDesc,
                         DeviceVectorFloat& dx){
      error_check( cudnnSoftmaxBackward(handle.handle, (::cudnnSoftmaxAlgorithm_t) algo,
                                        (::cudnnSoftmaxMode_t) mode, &alpha,
                                        yDesc.desc, y.ptr, dyDesc.desc, dy.ptr, &beta,
                                        dxDesc.desc, dx.ptr) );
    }

    class PoolingDescriptor {
    public:
      cudnnPoolingDescriptor_t desc;
      int degree;
      cudnnPoolingMode_t mode;
      int windowHeight;
      int windowWidth;
      int verticalPadding;
      int horizontalPadding;
      int verticalStride;
      int horizontalStride;

      PoolingDescriptor(cudnnPoolingMode_t m, 
                        int h, int w, 
                        int vPad, int hPad,
                        int vStr, int hStr) : 
        degree(2), mode(m), windowHeight(h), windowWidth(w), 
        verticalPadding(vPad), horizontalPadding(hPad), 
        verticalStride(vStr), horizontalStride(hStr) {
        error_check( cudnnCreatePoolingDescriptor(&desc) );
        error_check( cudnnSetPooling2dDescriptor( (::cudnnPoolingDescriptor_t) desc,
                                                  (::cudnnPoolingMode_t) m,
                                                  h, w, vPad, hPad, vStr, hStr) );
      }
      ~PoolingDescriptor() {
        error_check( cudnnDestroyPoolingDescriptor(desc) );
      }
    };
  }; // end of namespace RubyCuDNN

%}

namespace RubyCuDNN {
  %template(DeviceVectorDouble) DeviceVector<double>;
  %template(DeviceVectorFloat) DeviceVector<float>;
};
