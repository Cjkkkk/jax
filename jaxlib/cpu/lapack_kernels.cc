/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/cpu/lapack_kernels.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "jaxlib/ffi_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

static_assert(sizeof(jax::lapack_int) == sizeof(int32_t),
              "Expected LAPACK integers to be 32-bit");

namespace ffi = xla::ffi;

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::Side);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::Transpose);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::Diag);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::UpLo);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::svd::ComputationMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::eig::ComputationMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::schur::ComputationMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::schur::Sort);

namespace jax {

template <typename T>
inline T CastNoOverflow(int64_t value, std::string_view source = __FILE__) {
  auto result = MaybeCastNoOverflow<T>(value, source);
  if (!result.ok()) {
    throw std::overflow_error{std::string(result.status().message())};
  }
  return result.value();
}

template <ffi::DataType dtype>
void CopyIfDiffBuffer(ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out) {
  if (x.typed_data() != x_out->typed_data()) {
    const auto x_size = x.element_count();
    std::copy_n(x.typed_data(), x_size, x_out->typed_data());
  }
}

//== Triangular System Solver ==//

template <ffi::DataType dtype>
ffi::Error TriMatrixEquationSolver<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<dtype> y,
    // TODO(b/397715595): Remove RemainingArgs no earlier than 180 days after
    // the release of JAX 0.5.2.
    ffi::RemainingArgs, ffi::ResultBuffer<dtype> y_out, MatrixParams::Side side,
    MatrixParams::UpLo uplo, MatrixParams::Transpose trans_x,
    MatrixParams::Diag diag) {
  CopyIfDiffBuffer(y, y_out);
  FFI_ASSIGN_OR_RETURN((auto [batch_count, y_rows, y_cols]),
                       SplitBatch2D(y.dimensions()));
  auto* y_out_data = y_out->typed_data();
  lapack_int x_leading_dim_v =
      side == MatrixParams::Side::kLeft ? y_rows : y_cols;
  lapack_int y_leading_dim_v = y_rows;

  auto side_v = static_cast<char>(side);
  auto uplo_v = static_cast<char>(uplo);
  auto trans_x_v = static_cast<char>(trans_x);
  auto diag_v = static_cast<char>(diag);
  FFI_ASSIGN_OR_RETURN(auto y_rows_v, MaybeCastNoOverflow<lapack_int>(y_rows));
  FFI_ASSIGN_OR_RETURN(auto y_cols_v, MaybeCastNoOverflow<lapack_int>(y_cols));

  auto* x_data = x.typed_data();
  const int64_t y_out_step{y_rows * y_cols};
  const int64_t x_step{x_leading_dim_v * x_leading_dim_v};
  ffi::NativeType<dtype> alpha = static_cast<ffi::NativeType<dtype>>(1);
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&side_v, &uplo_v, &trans_x_v, &diag_v, &y_rows_v, &y_cols_v, &alpha,
       x_data, &x_leading_dim_v, y_out_data, &y_leading_dim_v);

    y_out_data += y_out_step;
    x_data += x_step;
  }
  return ffi::Error::Success();
}

template struct TriMatrixEquationSolver<ffi::DataType::F32>;
template struct TriMatrixEquationSolver<ffi::DataType::F64>;
template struct TriMatrixEquationSolver<ffi::DataType::C64>;
template struct TriMatrixEquationSolver<ffi::DataType::C128>;

//== LU Decomposition ==//

template <ffi::DataType dtype>
ffi::Error LuDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<LapackIntDtype> ipiv,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* ipiv_data = ipiv->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t ipiv_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, ipiv_data,
       info_data);
    x_out_data += x_out_step;
    ipiv_data += ipiv_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct LuDecomposition<ffi::DataType::F32>;
template struct LuDecomposition<ffi::DataType::F64>;
template struct LuDecomposition<ffi::DataType::C64>;
template struct LuDecomposition<ffi::DataType::C128>;

//== QR Factorization ==//

template <ffi::DataType dtype>
ffi::Error QrFactorization<dtype>::Kernel(ffi::Buffer<dtype> x,
                                          ffi::ResultBuffer<dtype> x_out,
                                          ffi::ResultBuffer<dtype> tau) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* tau_data = tau->typed_data();
  lapack_int info;
  const int64_t work_size = GetWorkspaceSize(x_rows, x_cols);
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  CopyIfDiffBuffer(x, x_out);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, tau_data,
       work_data.get(), &workspace_dim_v, &info);
    x_out_data += x_out_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t QrFactorization<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                 lapack_int x_cols) {
  ValueType optimal_size{};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct QrFactorization<ffi::DataType::F32>;
template struct QrFactorization<ffi::DataType::F64>;
template struct QrFactorization<ffi::DataType::C64>;
template struct QrFactorization<ffi::DataType::C128>;

//== Column Pivoting QR Factorization ==//

// lapack geqp3
template <ffi::DataType dtype>
ffi::Error PivotingQrFactorization<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<LapackIntDtype> jpvt,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> jpvt_out,
    ffi::ResultBuffer<dtype> tau) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* jpvt_out_data = jpvt_out->typed_data();
  auto* tau_data = tau->typed_data();
  lapack_int info;
  const int64_t work_size = GetWorkspaceSize(x_rows, x_cols);
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  constexpr bool is_complex_dtype = ffi::IsComplexType<dtype>();
  std::unique_ptr<RealType[]> rwork_data;
  if constexpr (is_complex_dtype) {
    rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(2 * x_cols);
  }

  CopyIfDiffBuffer(x, x_out);
  CopyIfDiffBuffer(jpvt, jpvt_out);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t jpvt_step{x_cols};
  const int64_t tau_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    if constexpr (is_complex_dtype) {
      fn(&x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, jpvt_out_data,
         tau_data, work_data.get(), &workspace_dim_v, rwork_data.get(), &info);
    } else {
      fn(&x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, jpvt_out_data,
         tau_data, work_data.get(), &workspace_dim_v, &info);
    }
    x_out_data += x_out_step;
    jpvt_out_data += jpvt_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t PivotingQrFactorization<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                         lapack_int x_cols) {
  ValueType optimal_size{};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  if constexpr (ffi::IsComplexType<dtype>()) {
    fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
       &optimal_size, &workspace_query, nullptr, &info);
  } else {
    fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
       &optimal_size, &workspace_query, &info);
  }
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct PivotingQrFactorization<ffi::DataType::F32>;
template struct PivotingQrFactorization<ffi::DataType::F64>;
template struct PivotingQrFactorization<ffi::DataType::C64>;
template struct PivotingQrFactorization<ffi::DataType::C128>;

//== Orthogonal QR                                      ==//
//== Computes orthogonal matrix Q from QR Decomposition ==//

template <ffi::DataType dtype>
ffi::Error OrthogonalQr<dtype>::Kernel(ffi::Buffer<dtype> x,
                                       ffi::Buffer<dtype> tau,
                                       ffi::ResultBuffer<dtype> x_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* tau_data = tau.typed_data();
  auto* x_out_data = x_out->typed_data();
  lapack_int info;

  CopyIfDiffBuffer(x, x_out);

  // Prepare LAPACK workspaces.
  int64_t work_size = GetWorkspaceSize(x_rows, x_cols, tau.dimensions().back());
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto tau_size_v, MaybeCastNoOverflow<lapack_int>(
                                            tau.dimensions().back()));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{tau_size_v};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, &tau_size_v, x_out_data, &x_leading_dim_v,
       tau_data, work_data.get(), &work_size_v, &info);
    x_out_data += x_out_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t OrthogonalQr<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                              lapack_int x_cols,
                                              lapack_int tau_size) {
  ValueType optimal_size = {};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  fn(&x_rows, &x_cols, &tau_size, nullptr, &x_leading_dim_v, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct OrthogonalQr<ffi::DataType::F32>;
template struct OrthogonalQr<ffi::DataType::F64>;
template struct OrthogonalQr<ffi::DataType::C64>;
template struct OrthogonalQr<ffi::DataType::C128>;

//== Cholesky Factorization ==//

template <ffi::DataType dtype>
ffi::Error CholeskyFactorization<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_order_v,
                       MaybeCastNoOverflow<lapack_int>(x.dimensions().back()));
  auto x_leading_dim_v = x_order_v;

  const int64_t x_out_step{x_rows * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, info_data);
    x_out_data += x_out_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct CholeskyFactorization<ffi::DataType::F32>;
template struct CholeskyFactorization<ffi::DataType::F64>;
template struct CholeskyFactorization<ffi::DataType::C64>;
template struct CholeskyFactorization<ffi::DataType::C128>;

//== Singular Value Decomposition (SVD) ==//
//== using a divide and conquer method  ==//

namespace internal {

template <ffi::DataType dtype>
static ffi::Error SvdKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (mode == svd::ComputationMode::kComputeVtOverwriteXPartialU) [[unlikely]] {
    return ffi::Error(
        XLA_FFI_Error_Code_UNIMPLEMENTED,
        "Current implementation does not support this computation mode");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* singular_values_data = singular_values->typed_data();
  auto* u_data = u->typed_data();
  auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();

  // Prepare LAPACK workspaces.
  FFI_ASSIGN_OR_RETURN(
      const auto work_size,
      svd::SVDType<dtype>::GetWorkspaceSize(x_rows, x_cols, mode));
  FFI_ASSIGN_OR_RETURN(const auto iwork_size,
                       svd::GetIntWorkspaceSize(x_rows, x_cols));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  auto iwork_data = AllocateScratchMemory<LapackIntDtype>(iwork_size);
  using RealType = typename svd::SVDType<dtype>::RealType;
  std::unique_ptr<RealType[]> rwork;
  if constexpr (ffi::IsComplexType<dtype>()) {
    FFI_ASSIGN_OR_RETURN(const auto rwork_size,
                         svd::GetRealWorkspaceSize(x_rows, x_cols, mode));
    rwork = AllocateScratchMemory<ffi::ToReal(dtype)>(rwork_size);
  }

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto mode_v = static_cast<char>(mode);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto x_leading_dim_v = x_rows_v;
  auto u_leading_dim_v = x_rows_v;

  auto u_dims = u->dimensions().last(2);
  auto vt_dims = vt->dimensions().last(2);
  FFI_ASSIGN_OR_RETURN(auto vt_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(vt_dims.front()));

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t singular_values_step{singular_values->dimensions().back()};
  const int64_t u_step{u_dims.front() * u_dims.back()};
  const int64_t vt_step{vt_leading_dim_v * vt_dims.back()};

  for (int64_t i = 0; i < batch_count; ++i) {
    if constexpr (ffi::IsComplexType<dtype>()) {
      svd::SVDType<dtype>::fn(&mode_v, &x_rows_v, &x_cols_v, x_out_data,
                              &x_leading_dim_v, singular_values_data, u_data,
                              &u_leading_dim_v, vt_data, &vt_leading_dim_v,
                              work_data.get(), &workspace_dim_v, rwork.get(),
                              iwork_data.get(), info_data);
    } else {
      svd::SVDType<dtype>::fn(&mode_v, &x_rows_v, &x_cols_v, x_out_data,
                              &x_leading_dim_v, singular_values_data, u_data,
                              &u_leading_dim_v, vt_data, &vt_leading_dim_v,
                              work_data.get(), &workspace_dim_v,
                              iwork_data.get(), info_data);
    }

    // Suppress MSAN warnings when using a copy of LAPACK uninstrumented by
    // MSAN.
    using T [[maybe_unused]] = typename svd::SVDType<dtype>::ValueType;
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(*info_data));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data,
                                        x_cols_v * x_leading_dim_v * sizeof(T));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        singular_values_data, std::min(x_rows_v, x_cols_v) * sizeof(RealType));
    if (mode_v == 'A') {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          u_data, u_leading_dim_v * x_rows_v * sizeof(T));
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          vt_data, vt_leading_dim_v * x_cols_v * sizeof(T));
    } else if (mode_v == 'O') {
      if (x_rows_v < x_cols_v) {
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
            u_data, u_leading_dim_v * x_rows_v * sizeof(T));
      } else {
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
            vt_data, vt_leading_dim_v * x_cols_v * sizeof(T));
      }
    } else if (mode_v == 'S') {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          u_data, u_leading_dim_v * std::min(x_rows_v, x_cols_v) * sizeof(T));
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          vt_data, vt_leading_dim_v * x_cols_v * sizeof(T));
    }

    x_out_data += x_out_step;
    singular_values_data += singular_values_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
static int64_t SvdGetWorkspaceSize(lapack_int x_rows, lapack_int x_cols,
                                   svd::ComputationMode mode) {
  ffi::NativeType<dtype> optimal_size = {};
  lapack_int info = 0;
  lapack_int workspace_query = -1;

  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows;
  auto u_leading_dim_v = x_rows;
  auto vt_leading_dim_v = mode == svd::ComputationMode::kComputeFullUVt
                              ? x_cols
                              : std::min(x_rows, x_cols);
  if constexpr (ffi::IsComplexType<dtype>()) {
    svd::SVDType<dtype>::fn(
        &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
        &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, nullptr, nullptr, &info);
  } else {
    svd::SVDType<dtype>::fn(&mode_v, &x_rows, &x_cols, nullptr,
                            &x_leading_dim_v, nullptr, nullptr,
                            &u_leading_dim_v, nullptr, &vt_leading_dim_v,
                            &optimal_size, &workspace_query, nullptr, &info);
  }
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template <ffi::DataType dtype>
static ffi::Error SvdQRKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (mode == svd::ComputationMode::kComputeVtOverwriteXPartialU) [[unlikely]] {
    return ffi::Error(
        XLA_FFI_Error_Code_UNIMPLEMENTED,
        "SVD: Current implementation does not support this computation mode");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* singular_values_data = singular_values->typed_data();
  auto* u_data = u->typed_data();
  auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();

  // Prepare LAPACK workspaces.
  FFI_ASSIGN_OR_RETURN(
      const auto work_size,
      svd::SVDQRType<dtype>::GetWorkspaceSize(x_rows, x_cols, mode));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  using RealType = typename svd::SVDType<dtype>::RealType;
  std::unique_ptr<RealType[]> rwork;
  if constexpr (ffi::IsComplexType<dtype>()) {
    FFI_ASSIGN_OR_RETURN(const auto rwork_size,
                         svd::GetRealWorkspaceSizeQR(x_rows, x_cols));
    rwork = AllocateScratchMemory<ffi::ToReal(dtype)>(rwork_size);
  }

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto mode_v = static_cast<char>(mode);
  auto workspace_dim_v = work_size;
  auto x_leading_dim_v = x_rows_v;
  auto u_leading_dim_v = x_rows_v;

  auto u_dims = u->dimensions().last(2);
  auto vt_dims = vt->dimensions().last(2);
  FFI_ASSIGN_OR_RETURN(auto vt_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(vt_dims.front()));

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t singular_values_step{singular_values->dimensions().back()};
  const int64_t u_step{u_dims.front() * u_dims.back()};
  const int64_t vt_step{vt_leading_dim_v * vt_dims.back()};

  for (int64_t i = 0; i < batch_count; ++i) {
    if constexpr (ffi::IsComplexType<dtype>()) {
      svd::SVDQRType<dtype>::fn(&mode_v, &mode_v, &x_rows_v, &x_cols_v,
                                x_out_data, &x_leading_dim_v,
                                singular_values_data, u_data, &u_leading_dim_v,
                                vt_data, &vt_leading_dim_v, work_data.get(),
                                &workspace_dim_v, rwork.get(), info_data);
    } else {
      svd::SVDQRType<dtype>::fn(
          &mode_v, &mode_v, &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v,
          singular_values_data, u_data, &u_leading_dim_v, vt_data,
          &vt_leading_dim_v, work_data.get(), &workspace_dim_v, info_data);
    }
    x_out_data += x_out_step;
    singular_values_data += singular_values_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
static absl::StatusOr<lapack_int> SvdQRGetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  ffi::NativeType<dtype> optimal_size = {};
  lapack_int info = 0;
  lapack_int workspace_query = -1;

  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows;
  auto u_leading_dim_v = x_rows;
  auto vt_leading_dim_v = mode == svd::ComputationMode::kComputeFullUVt
                              ? x_cols
                              : std::min(x_rows, x_cols);
  if constexpr (ffi::IsComplexType<dtype>()) {
    svd::SVDQRType<dtype>::fn(&mode_v, &mode_v, &x_rows, &x_cols, nullptr,
                              &x_leading_dim_v, nullptr, nullptr,
                              &u_leading_dim_v, nullptr, &vt_leading_dim_v,
                              &optimal_size, &workspace_query, nullptr, &info);
  } else {
    svd::SVDQRType<dtype>::fn(&mode_v, &mode_v, &x_rows, &x_cols, nullptr,
                              &x_leading_dim_v, nullptr, nullptr,
                              &u_leading_dim_v, nullptr, &vt_leading_dim_v,
                              &optimal_size, &workspace_query, &info);
  }
  return info == 0 ? MaybeCastNoOverflow<lapack_int>(std::real(optimal_size))
                   : -1;
}

}  // namespace internal

template <ffi::DataType dtype>
ffi::Error SingularValueDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> singular_values, ffi::ResultBuffer<dtype> u,
    ffi::ResultBuffer<dtype> vt, ffi::ResultBuffer<LapackIntDtype> info,
    svd::ComputationMode mode) {
  return internal::SvdKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                    mode);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  return internal::SvdKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                    mode);
}

template <ffi::DataType dtype>
absl::StatusOr<int64_t> SingularValueDecomposition<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype>
absl::StatusOr<int64_t>
SingularValueDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionQR<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> singular_values, ffi::ResultBuffer<dtype> u,
    ffi::ResultBuffer<dtype> vt, ffi::ResultBuffer<LapackIntDtype> info,
    svd::ComputationMode mode) {
  return internal::SvdQRKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                      mode);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionQRComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  return internal::SvdQRKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                      mode);
}

template <ffi::DataType dtype>
absl::StatusOr<lapack_int>
SingularValueDecompositionQR<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdQRGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype>
absl::StatusOr<lapack_int>
SingularValueDecompositionQRComplex<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdQRGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

absl::StatusOr<lapack_int> svd::GetRealWorkspaceSize(
    int64_t x_rows, int64_t x_cols, svd::ComputationMode mode) {
  const auto min_dim = std::min(x_rows, x_cols);
  if (!ComputesUV(mode)) {
    return MaybeCastNoOverflow<lapack_int>(7 * min_dim);
  }
  const auto max_dim = std::max(x_rows, x_cols);
  return MaybeCastNoOverflow<lapack_int>(
      std::max(5 * min_dim * min_dim + 5 * min_dim,
               2 * max_dim * min_dim + 2 * min_dim * min_dim + min_dim));
}

absl::StatusOr<lapack_int> svd::GetRealWorkspaceSizeQR(int64_t x_rows,
                                                       int64_t x_cols) {
  return CastNoOverflow<lapack_int>(5 * std::min(x_rows, x_cols));
}

absl::StatusOr<lapack_int> svd::GetIntWorkspaceSize(int64_t x_rows,
                                                    int64_t x_cols) {
  return CastNoOverflow<lapack_int>(8 * std::min(x_rows, x_cols));
}

template struct SingularValueDecomposition<ffi::DataType::F32>;
template struct SingularValueDecomposition<ffi::DataType::F64>;
template struct SingularValueDecompositionComplex<ffi::DataType::C64>;
template struct SingularValueDecompositionComplex<ffi::DataType::C128>;

template struct SingularValueDecompositionQR<ffi::DataType::F32>;
template struct SingularValueDecompositionQR<ffi::DataType::F64>;
template struct SingularValueDecompositionQRComplex<ffi::DataType::C64>;
template struct SingularValueDecompositionQRComplex<ffi::DataType::C128>;

//== Eigenvalues and eigenvectors ==//

absl::StatusOr<lapack_int> eig::GetWorkspaceSize(int64_t x_cols,
                                                 ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(2 * x_cols + 1);
    case ComputationMode::kComputeEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(1 + 6 * x_cols +
                                             2 * x_cols * x_cols);
  }
}

absl::StatusOr<lapack_int> eig::GetIntWorkspaceSize(int64_t x_cols,
                                                    ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return 1;
    case ComputationMode::kComputeEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(3 + 5 * x_cols);
  }
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionSymmetric<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* eigenvalues_data = eigenvalues->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto mode_v = static_cast<char>(mode);
  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(x_cols));
  // Prepare LAPACK workspaces.
  FFI_ASSIGN_OR_RETURN(lapack_int work_size_v,
                       eig::GetWorkspaceSize(x_cols, mode));
  FFI_ASSIGN_OR_RETURN(lapack_int iwork_size_v,
                       eig::GetIntWorkspaceSize(x_cols, mode));
  auto work_data = AllocateScratchMemory<dtype>(work_size_v);
  auto iwork_data = AllocateScratchMemory<LapackIntDtype>(iwork_size_v);

  const int64_t x_out_step{x_cols * x_cols};
  const int64_t eigenvalues_step{x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &uplo_v, &x_cols_v, x_out_data, &x_leading_dim_v,
       eigenvalues_data, work_data.get(), &work_size_v, iwork_data.get(),
       &iwork_size_v, info_data);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data,
                                        sizeof(*x_out_data) * x_cols * x_cols);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigenvalues_data,
                                        sizeof(*eigenvalues_data) * x_cols);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
    x_out_data += x_out_step;
    eigenvalues_data += eigenvalues_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

namespace eig {

absl::StatusOr<lapack_int> GetComplexWorkspaceSize(int64_t x_cols,
                                                   ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(x_cols + 1);
    case ComputationMode::kComputeEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(2 * x_cols + x_cols * x_cols);
  }
}

absl::StatusOr<lapack_int> GetRealWorkspaceSize(int64_t x_cols,
                                                ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(std::max(x_cols, int64_t{1}));
    case ComputationMode::kComputeEigenvectors:
      return MaybeCastNoOverflow<lapack_int>(1 + 5 * x_cols +
                                             2 * x_cols * x_cols);
  }
}

}  // namespace eig

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionHermitian<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* eigenvalues_data = eigenvalues->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto mode_v = static_cast<char>(mode);
  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(x_cols));
  // Prepare LAPACK workspaces.
  FFI_ASSIGN_OR_RETURN(lapack_int work_size_v,
                       eig::GetComplexWorkspaceSize(x_cols, mode));
  FFI_ASSIGN_OR_RETURN(lapack_int rwork_size_v,
                       eig::GetRealWorkspaceSize(x_cols, mode));
  FFI_ASSIGN_OR_RETURN(lapack_int iwork_size_v,
                       eig::GetIntWorkspaceSize(x_cols, mode));
  auto work_data = AllocateScratchMemory<dtype>(work_size_v);
  auto iwork_data = AllocateScratchMemory<LapackIntDtype>(iwork_size_v);
  auto rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(rwork_size_v);

  const int64_t x_out_step{x_cols * x_cols};
  const int64_t eigenvalues_step{x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &uplo_v, &x_cols_v, x_out_data, &x_leading_dim_v,
       eigenvalues_data, work_data.get(), &work_size_v, rwork_data.get(),
       &rwork_size_v, iwork_data.get(), &iwork_size_v, info_data);
    x_out_data += x_out_step;
    eigenvalues_data += eigenvalues_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct EigenvalueDecompositionSymmetric<ffi::DataType::F32>;
template struct EigenvalueDecompositionSymmetric<ffi::DataType::F64>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C64>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C128>;

template <ffi::DataType dtype>
ffi::Error EigenvalueDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_left,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));

  const auto* x_data = x.typed_data();
  auto* eigvecs_left_data = eigvecs_left->typed_data();
  auto* eigvecs_right_data = eigvecs_right->typed_data();
  auto* eigvals_real_data = eigvals_real->typed_data();
  auto* eigvals_imag_data = eigvals_imag->typed_data();
  auto* info_data = info->typed_data();

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  // Prepare LAPACK workspaces.
  int64_t work_size = GetWorkspaceSize(x_cols_v, compute_left, compute_right);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  const int64_t x_size{x_cols * x_cols};
  auto x_copy = AllocateScratchMemory<dtype>(x_size);
  auto work_eigvecs_left = AllocateScratchMemory<dtype>(x_size);
  auto work_eigvecs_right = AllocateScratchMemory<dtype>(x_size);

  const auto is_finite = [](ValueType* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size),
                          [](ValueType value) { return std::isfinite(value); });
  };

  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    std::copy_n(x_data, x_size, x_copy.get());
    if (is_finite(x_copy.get(), x_size)) {
      fn(&compute_left_v, &compute_right_v, &x_cols_v, x_copy.get(), &x_cols_v,
         eigvals_real_data, eigvals_imag_data, work_eigvecs_left.get(),
         &x_cols_v, work_eigvecs_right.get(), &x_cols_v, work_data.get(),
         &work_size_v, info_data);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_copy.get(), x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_real_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_imag_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(work_eigvecs_left.get(),
                                          x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(work_eigvecs_right.get(),
                                          x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
      if (info_data[0] == 0) {
        UnpackEigenvectors(x_cols_v, eigvals_imag_data, work_eigvecs_left.get(),
                           eigvecs_left_data);
        UnpackEigenvectors(x_cols_v, eigvals_imag_data,
                           work_eigvecs_right.get(), eigvecs_right_data);
      }
    } else {
      info_data[0] = -4;
    }
    x_data += x_size;
    eigvals_real_data += x_cols;
    eigvals_imag_data += x_cols;
    eigvecs_left_data += x_size;
    eigvecs_right_data += x_size;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals,
    ffi::ResultBuffer<dtype> eigvecs_left,
    ffi::ResultBuffer<dtype> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  const auto* x_data = x.typed_data();
  auto* eigvecs_left_data = eigvecs_left->typed_data();
  auto* eigvecs_right_data = eigvecs_right->typed_data();
  auto* eigvals_data = eigvals->typed_data();
  auto* info_data = info->typed_data();

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  // Prepare LAPACK workspaces.
  int64_t work_size = GetWorkspaceSize(x_cols_v, compute_left, compute_right);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  const int64_t x_size{x_cols * x_cols};
  auto x_copy = AllocateScratchMemory<dtype>(x_size);
  auto rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(2 * x_cols);

  const auto is_finite = [](ValueType* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size), [](const auto& z) {
      return std::isfinite(z.real()) && std::isfinite(z.imag());
    });
  };

  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    std::copy_n(x_data, x_size, x_copy.get());
    if (is_finite(x_copy.get(), x_size)) {
      fn(&compute_left_v, &compute_right_v, &x_cols_v, x_copy.get(), &x_cols_v,
         eigvals_data, eigvecs_left_data, &x_cols_v, eigvecs_right_data,
         &x_cols_v, work_data.get(), &work_size_v, rwork_data.get(), info_data);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_copy.get(), x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvecs_left_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvecs_right_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
    } else {
      info_data[0] = -4;
    }
    x_data += x_size;
    eigvals_data += x_cols;
    eigvecs_left_data += x_size;
    eigvecs_right_data += x_size;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t EigenvalueDecomposition<dtype>::GetWorkspaceSize(
    lapack_int x_cols, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  fn(&compute_left_v, &compute_right_v, &x_cols, nullptr, &x_cols, nullptr,
     nullptr, nullptr, &x_cols, nullptr, &x_cols, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template <ffi::DataType dtype>
int64_t EigenvalueDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_cols, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;
  // NULL rwork crashes, LAPACK unnecessarily writes x_cols into rwork
  RealType rwork[1];
  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  fn(&compute_left_v, &compute_right_v, &x_cols, nullptr, &x_cols, nullptr,
     nullptr, &x_cols, nullptr, &x_cols, &optimal_size, &workspace_query, rwork,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template struct EigenvalueDecomposition<ffi::DataType::F32>;
template struct EigenvalueDecomposition<ffi::DataType::F64>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C64>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C128>;

//== Schur Decomposition ==//

template <ffi::DataType dtype>
ffi::Error SchurDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  if (sort != schur::Sort::kNoSortEigenvalues) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "Ordering eigenvalues on the diagonal is not implemented");
  }

  CopyIfDiffBuffer(x, x_out);

  // TODO(paruzelp): `select` should be passed as an execution context
  bool (*select)(ValueType, ValueType) = nullptr;
  ValueType* x_out_data = x_out->typed_data();
  ValueType* eigvals_real_data = eigvals_real->typed_data();
  ValueType* eigvals_imag_data = eigvals_imag->typed_data();
  ValueType* schur_vectors_data = schur_vectors->typed_data();
  lapack_int* selected_data = selected_eigvals->typed_data();
  lapack_int* info_data = info->typed_data();

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));

  // Prepare LAPACK workspaces.
  std::unique_ptr<bool[]> bwork =
      sort != schur::Sort::kNoSortEigenvalues
          ? AllocateScratchMemory<ffi::DataType::PRED>(x_cols)
          : nullptr;
  auto work_size = GetWorkspaceSize(x_cols, mode, sort);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &sort_v, select, &x_cols_v, x_out_data, &x_cols_v,
       selected_data, eigvals_real_data, eigvals_imag_data, schur_vectors_data,
       &x_cols_v, work_data.get(), &work_size_v, bwork.get(), info_data);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(selected_data, sizeof(lapack_int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_real_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_imag_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(schur_vectors_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));

    x_out_data += x_size;
    eigvals_real_data += x_cols;
    eigvals_imag_data += x_cols;
    schur_vectors_data += x_size;
    ++selected_data;
    ++info_data;
  }

  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error SchurDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<dtype> eigvals,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  if (sort != schur::Sort::kNoSortEigenvalues) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "Ordering eigenvalues on the diagonal is not implemented");
  }

  CopyIfDiffBuffer(x, x_out);

  // TODO(paruzelp): `select` should be passed as an execution context
  bool (*select)(ValueType) = nullptr;
  ValueType* x_out_data = x_out->typed_data();
  ValueType* eigvals_data = eigvals->typed_data();
  ValueType* schur_vectors_data = schur_vectors->typed_data();
  lapack_int* selected_data = selected_eigvals->typed_data();
  lapack_int* info_data = info->typed_data();

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));

  // Prepare LAPACK workspaces.
  std::unique_ptr<bool[]> bwork =
      sort != schur::Sort::kNoSortEigenvalues
          ? AllocateScratchMemory<ffi::DataType::PRED>(x_cols)
          : nullptr;
  auto work_size = GetWorkspaceSize(x_cols, mode, sort);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  auto rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(x_cols);

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &sort_v, select, &x_cols_v, x_out_data, &x_cols_v,
       selected_data, eigvals_data, schur_vectors_data, &x_cols_v,
       work_data.get(), &work_size_v, rwork_data.get(), bwork.get(), info_data);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(schur_vectors_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(selected_data, sizeof(lapack_int));

    x_out_data += x_size;
    eigvals_data += x_cols;
    schur_vectors_data += x_size;
    ++selected_data;
    ++info_data;
  }

  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t SchurDecomposition<dtype>::GetWorkspaceSize(lapack_int x_cols,
                                                    schur::ComputationMode mode,
                                                    schur::Sort sort) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  fn(&mode_v, &sort_v, nullptr, &x_cols, nullptr, &x_cols, nullptr, nullptr,
     nullptr, nullptr, &x_cols, &optimal_size, &workspace_query, nullptr,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template <ffi::DataType dtype>
int64_t SchurDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_cols, schur::ComputationMode mode, schur::Sort sort) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  fn(&mode_v, &sort_v, nullptr, &x_cols, nullptr, &x_cols, nullptr, nullptr,
     nullptr, &x_cols, &optimal_size, &workspace_query, nullptr, nullptr,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template struct SchurDecomposition<ffi::DataType::F32>;
template struct SchurDecomposition<ffi::DataType::F64>;
template struct SchurDecompositionComplex<ffi::DataType::C64>;
template struct SchurDecompositionComplex<ffi::DataType::C128>;

//== Hessenberg Decomposition ==//

template <ffi::DataType dtype>
ffi::Error HessenbergDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, lapack_int low, lapack_int high,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> tau,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));

  CopyIfDiffBuffer(x, x_out);

  ValueType* x_out_data = x_out->typed_data();
  ValueType* tau_data = tau->typed_data();
  lapack_int* info_data = info->typed_data();
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(x_rows));
  // Prepare LAPACK workspaces.
  int64_t work_size = GetWorkspaceSize(x_rows, x_cols, low, high);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  int64_t x_size{x_rows * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_cols_v, &low, &high, x_out_data, &x_leading_dim_v, tau_data,
       work_data.get(), &work_size_v, info_data);
    x_out_data += x_size;
    tau_data += x_cols - 1;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t HessenbergDecomposition<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                         lapack_int x_cols,
                                                         lapack_int low,
                                                         lapack_int high) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;
  fn(&x_cols, &low, &high, nullptr, &x_rows, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct HessenbergDecomposition<ffi::DataType::F32>;
template struct HessenbergDecomposition<ffi::DataType::F64>;
template struct HessenbergDecomposition<ffi::DataType::C64>;
template struct HessenbergDecomposition<ffi::DataType::C128>;

//== Tridiagonal Reduction ==//

template <ffi::DataType dtype>
ffi::Error TridiagonalReduction<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> diagonal,
    ffi::ResultBuffer<ffi::ToReal(dtype)> off_diagonal,
    ffi::ResultBuffer<dtype> tau, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));

  CopyIfDiffBuffer(x, x_out);

  ValueType* x_out_data = x_out->typed_data();
  RealType* diagonal_data = diagonal->typed_data();
  RealType* off_diagonal_data = off_diagonal->typed_data();
  ValueType* tau_data = tau->typed_data();
  lapack_int* info_data = info->typed_data();

  // Prepare LAPACK workspaces.
  const auto work_size = GetWorkspaceSize(x_rows, x_cols);
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<lapack_int>(work_size));
  FFI_ASSIGN_OR_RETURN(auto x_order_v, MaybeCastNoOverflow<lapack_int>(x_cols));

  int64_t x_size = x_rows * x_cols;
  int64_t tau_step = {tau->dimensions().back()};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, diagonal_data,
       off_diagonal_data, tau_data, work_data.get(), &work_size_v, info_data);
    x_out_data += x_size;
    diagonal_data += x_cols;
    off_diagonal_data += x_cols - 1;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t TridiagonalReduction<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                      lapack_int x_cols) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;
  char uplo_v = 'L';
  fn(&uplo_v, &x_cols, nullptr, &x_rows, nullptr, nullptr, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct TridiagonalReduction<ffi::DataType::F32>;
template struct TridiagonalReduction<ffi::DataType::F64>;
template struct TridiagonalReduction<ffi::DataType::C64>;
template struct TridiagonalReduction<ffi::DataType::C128>;

//== General Tridiagonal System Solver ==//

// lapack gtsv

template <ffi::DataType dtype>
ffi::Error TridiagonalSolver<dtype>::Kernel(
    ffi::Buffer<dtype> dl, ffi::Buffer<dtype> d, ffi::Buffer<dtype> du,
    ffi::Buffer<dtype> b, ffi::ResultBuffer<dtype> dl_out,
    ffi::ResultBuffer<dtype> d_out, ffi::ResultBuffer<dtype> du_out,
    ffi::ResultBuffer<dtype> b_out, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, b_rows, b_cols]),
                       SplitBatch2D(b.dimensions()));

  CopyIfDiffBuffer(dl, dl_out);
  CopyIfDiffBuffer(d, d_out);
  CopyIfDiffBuffer(du, du_out);
  CopyIfDiffBuffer(b, b_out);

  auto* dl_out_data = dl_out->typed_data();
  auto* d_out_data = d_out->typed_data();
  auto* du_out_data = du_out->typed_data();
  auto* b_out_data = b_out->typed_data();
  auto* info_data = info->typed_data();

  FFI_ASSIGN_OR_RETURN(auto b_rows_v, MaybeCastNoOverflow<lapack_int>(b_rows));
  FFI_ASSIGN_OR_RETURN(auto b_cols_v, MaybeCastNoOverflow<lapack_int>(b_cols));

  const int64_t b_out_step{b_rows * b_cols};
  const int64_t d_step{b_rows};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&b_rows_v, &b_cols_v, dl_out_data + 1, d_out_data, du_out_data,
       b_out_data, &b_rows_v, info_data);
    b_out_data += b_out_step;
    dl_out_data += d_step;
    d_out_data += d_step;
    du_out_data += d_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct TridiagonalSolver<ffi::DataType::F32>;
template struct TridiagonalSolver<ffi::DataType::F64>;
template struct TridiagonalSolver<ffi::DataType::C64>;
template struct TridiagonalSolver<ffi::DataType::C128>;

// FFI Definition Macros (by DataType)

#define JAX_CPU_DEFINE_TRSM(name, data_type)             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, TriMatrixEquationSolver<data_type>::Kernel,  \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*y*/)     \
          .RemainingArgs()                               \
          .Ret<::xla::ffi::Buffer<data_type>>(/*y_out*/) \
          .Attr<MatrixParams::Side>("side")              \
          .Attr<MatrixParams::UpLo>("uplo")              \
          .Attr<MatrixParams::Transpose>("trans_x")      \
          .Attr<MatrixParams::Diag>("diag"))

#define JAX_CPU_DEFINE_GETRF(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, LuDecomposition<data_type>::Kernel,              \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*ipiv*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEQRF(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, QrFactorization<data_type>::Kernel,          \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/))

#define JAX_CPU_DEFINE_GEQP3(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, PivotingQrFactorization<data_type>::Kernel,          \
      ::xla::ffi::Ffi::Bind()                                    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)             \
          .Arg<::xla::ffi::Buffer<LapackIntDtype>>(/*jpvt*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)         \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*jpvt_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/))

#define JAX_CPU_DEFINE_ORGQR(name, data_type)          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                       \
      name, OrthogonalQr<data_type>::Kernel,           \
      ::xla::ffi::Ffi::Bind()                          \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)   \
          .Arg<::xla::ffi::Buffer<data_type>>(/*tau*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/))

#define JAX_CPU_DEFINE_POTRF(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, CholeskyFactorization<data_type>::Kernel,    \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Attr<MatrixParams::UpLo>("uplo")              \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GESDD(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, SingularValueDecomposition<data_type>::Kernel,   \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*s*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)        \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/) \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESDD_COMPLEX(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SingularValueDecompositionComplex<data_type>::Kernel,        \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>(/*s*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)                    \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)             \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESVD(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, SingularValueDecompositionQR<data_type>::Kernel, \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*s*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)        \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/) \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESVD_COMPLEX(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SingularValueDecompositionQRComplex<data_type>::Kernel,      \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>(/*s*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)                    \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)             \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_SYEVD(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, EigenvalueDecompositionSymmetric<data_type>::Kernel, \
      ::xla::ffi::Ffi::Bind()                                    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)             \
          .Attr<MatrixParams::UpLo>("uplo")                      \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigenvalues*/)   \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)     \
          .Attr<eig::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_HEEVD(name, data_type)                      \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                   \
      name, EigenvalueDecompositionHermitian<data_type>::Kernel,   \
      ::xla::ffi::Ffi::Bind()                                      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)               \
          .Attr<MatrixParams::UpLo>("uplo")                        \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)           \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*eigenvalues*/)                                     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)       \
          .Attr<eig::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GEEV(name, data_type)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                      \
      name, EigenvalueDecomposition<data_type>::Kernel,               \
      ::xla::ffi::Ffi::Bind()                                         \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                  \
          .Attr<eig::ComputationMode>("compute_left")                 \
          .Attr<eig::ComputationMode>("compute_right")                \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_real*/)       \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_imag*/)       \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToComplex(data_type)>>( \
              /*eigvecs_left*/)                                       \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToComplex(data_type)>>( \
              /*eigvecs_right*/)                                      \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEEV_COMPLEX(name, data_type)             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, EigenvalueDecompositionComplex<data_type>::Kernel,   \
      ::xla::ffi::Ffi::Bind()                                    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)             \
          .Attr<eig::ComputationMode>("compute_left")            \
          .Attr<eig::ComputationMode>("compute_right")           \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals*/)       \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvecs_left*/)  \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvecs_right*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEES(name, data_type)                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SchurDecomposition<data_type>::Kernel,                       \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Attr<schur::ComputationMode>("mode")                          \
          .Attr<schur::Sort>("sort")                                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<data_type>>(/*schur_vectors*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_real*/)          \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_imag*/)          \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*selected_eigvals*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEES_COMPLEX(name, data_type)                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SchurDecompositionComplex<data_type>::Kernel,                \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Attr<schur::ComputationMode>("mode")                          \
          .Attr<schur::Sort>("sort")                                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<data_type>>(/*schur_vectors*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals*/)               \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*selected_eigvals*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_SYTRD_HETRD(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                   \
      name, TridiagonalReduction<data_type>::Kernel,               \
      ::xla::ffi::Ffi::Bind()                                      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)               \
          .Attr<MatrixParams::UpLo>("uplo")                        \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)           \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*diagonal*/)                                        \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*off_diagonal*/)                                    \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/)             \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEHRD(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, HessenbergDecomposition<data_type>::Kernel,  \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Attr<lapack_int>("low")                       \
          .Attr<lapack_int>("high")                      \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/)   \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GTSV(name, data_type)              \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                          \
      name, TridiagonalSolver<data_type>::Kernel,         \
      ::xla::ffi::Ffi::Bind()                             \
          .Arg<::xla::ffi::Buffer<data_type>>(/*dl*/)     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*d*/)      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*du*/)     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*b*/)      \
          .Ret<::xla::ffi::Buffer<data_type>>(/*dl_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*d_out*/)  \
          .Ret<::xla::ffi::Buffer<data_type>>(/*du_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*b_out*/)  \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

// FFI Handlers

JAX_CPU_DEFINE_TRSM(lapack_strsm_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_TRSM(lapack_dtrsm_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_TRSM(lapack_ctrsm_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_TRSM(lapack_ztrsm_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GETRF(lapack_sgetrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GETRF(lapack_dgetrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GETRF(lapack_cgetrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GETRF(lapack_zgetrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEQRF(lapack_sgeqrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEQRF(lapack_dgeqrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEQRF(lapack_cgeqrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEQRF(lapack_zgeqrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEQP3(lapack_sgeqp3_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEQP3(lapack_dgeqp3_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEQP3(lapack_cgeqp3_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEQP3(lapack_zgeqp3_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_ORGQR(lapack_sorgqr_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_ORGQR(lapack_dorgqr_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_ORGQR(lapack_cungqr_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_ORGQR(lapack_zungqr_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_POTRF(lapack_spotrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_POTRF(lapack_dpotrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_POTRF(lapack_cpotrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_POTRF(lapack_zpotrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GESDD(lapack_sgesdd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GESDD(lapack_dgesdd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_cgesdd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_zgesdd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GESVD(lapack_sgesvd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GESVD(lapack_dgesvd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GESVD_COMPLEX(lapack_cgesvd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GESVD_COMPLEX(lapack_zgesvd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_SYEVD(lapack_ssyevd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_SYEVD(lapack_dsyevd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_HEEVD(lapack_cheevd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_HEEVD(lapack_zheevd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEEV(lapack_sgeev_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEEV(lapack_dgeev_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEEV_COMPLEX(lapack_cgeev_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEEV_COMPLEX(lapack_zgeev_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_SYTRD_HETRD(lapack_ssytrd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_dsytrd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_chetrd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_zhetrd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEES(lapack_sgees_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEES(lapack_dgees_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEES_COMPLEX(lapack_cgees_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEES_COMPLEX(lapack_zgees_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEHRD(lapack_sgehrd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEHRD(lapack_dgehrd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEHRD(lapack_cgehrd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEHRD(lapack_zgehrd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GTSV(lapack_sgtsv_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GTSV(lapack_dgtsv_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GTSV(lapack_cgtsv_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GTSV(lapack_zgtsv_ffi, ::xla::ffi::DataType::C128);

#undef JAX_CPU_DEFINE_TRSM
#undef JAX_CPU_DEFINE_GETRF
#undef JAX_CPU_DEFINE_GEQRF
#undef JAX_CPU_DEFINE_GEQP3
#undef JAX_CPU_DEFINE_ORGQR
#undef JAX_CPU_DEFINE_POTRF
#undef JAX_CPU_DEFINE_GESDD
#undef JAX_CPU_DEFINE_GESDD_COMPLEX
#undef JAX_CPU_DEFINE_GESVD
#undef JAX_CPU_DEFINE_GESVD_COMPLEX
#undef JAX_CPU_DEFINE_SYEVD
#undef JAX_CPU_DEFINE_HEEVD
#undef JAX_CPU_DEFINE_GEEV
#undef JAX_CPU_DEFINE_GEEV_COMPLEX
#undef JAX_CPU_DEFINE_SYTRD_HETRD
#undef JAX_CPU_DEFINE_GEES
#undef JAX_CPU_DEFINE_GEES_COMPLEX
#undef JAX_CPU_DEFINE_GEHRD
#undef JAX_CPU_DEFINE_GTSV

}  // namespace jax
