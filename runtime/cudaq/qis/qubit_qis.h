/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/MeasureCounts.h"
#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qreg.h"
#include "cudaq/qis/qvector.h"
#include "cudaq/spin_op.h"
#include "host_config.h"
#include <cstring>
#include <functional>
#include <tuple>

#define __qpu__ __attribute__((annotate("quantum")))

// This file describes the API for a default qubit logical instruction
// set for CUDA-Q kernels.

/// For C++17 we can't adhere to the language specification for
/// the operation modifier type. For this case, we drop the modifier
/// template parameter and users have access to a `cNAME` operation for
/// single controlled operations.
#ifdef CUDAQ_USE_STD20
#define CUDAQ_MOD_TEMPLATE template <typename mod = base, typename... Args>
#else
#define CUDAQ_MOD_TEMPLATE template <typename... Args>
#endif

namespace cudaq {

// --------------------------
// Useful C++17 compliant concept checks (note we re-implement
// std::remove_cvref since its a C++20 thing)
template <typename T>
using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using IsQubitType = std::is_same<remove_cvref<T>, cudaq::qubit>;

template <typename T>
using IsQvectorType = std::is_same<remove_cvref<T>, cudaq::qvector<>>;

template <typename T>
using IsQviewType = std::is_same<remove_cvref<T>, cudaq::qview<>>;

template <typename T>
using IsQarrayType = std::is_base_of<cudaq::qarray_base, remove_cvref<T>>;
// --------------------------

namespace details {
template <size_t N, typename Tuple, size_t... Indices>
auto tuple_slice_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  return std::make_tuple(std::get<Indices>(std::forward<Tuple>(tuple))...);
}

template <size_t N, typename... Args>
auto tuple_slice(std::tuple<Args...> &&tuple) {
  return tuple_slice_impl<N>(std::forward<std::tuple<Args...>>(tuple),
                             std::make_index_sequence<N>{});
}

template <size_t N, typename Tuple, size_t... Indices>
auto tuple_slice_last_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  constexpr size_t M = std::tuple_size_v<std::remove_reference_t<Tuple>> - N;
  return std::forward_as_tuple(
      std::get<M + Indices>(std::forward<Tuple>(tuple))...);
}

template <size_t N, typename... Args>
auto tuple_slice_last(std::tuple<Args...> &&tuple) {
  return tuple_slice_last_impl<N>(std::forward<std::tuple<Args...>>(tuple),
                                  std::make_index_sequence<N>{});
}

/// @brief Convert a qubit to its unique id representation
inline QuditInfo qubitToQuditInfo(qubit &q) { return {q.n_levels(), q.id()}; }

/// @brief Map provided qubit arguments to a vector of QuditInfo.
template <typename... QuantumT>
void qubitsToQuditInfos(const std::tuple<QuantumT...> &quantumTuple,
                        std::vector<QuditInfo> &qubits) {
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        qubits.push_back(qubitToQuditInfo(q));
    }
  });
}

/// @brief Search through the qubit arguments and see which ones are negated.
template <typename... QuantumT>
void findQubitNegations(const std::tuple<QuantumT...> &quantumTuple,
                        std::vector<bool> &qubitIsNegated) {
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      qubitIsNegated.push_back(element.is_negative());
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        qubitIsNegated.push_back(q.is_negative());
    }
    return;
  });
}

/// @brief Generic quantum operation applicator function. Supports the
/// following signatures for a generic operation name `OP`
/// `OP(qubit(s))`
/// `OP<ctrl>(qubit..., qubit)`
/// `OP<ctrl>(qubits, qubit)`
/// `OP(scalar..., qubit(s))`
/// `OP<ctrl>(scalar..., qubit..., qubit)`
/// `OP<ctrl>(scalar..., qubits, qubit)`
/// `OP<adj>(qubit)`
/// `OP<adj>(scalar..., qubit)`
/// Control qubits can be negated. Compile errors should be thrown
/// for erroneous signatures.
template <typename mod, std::size_t NumT, std::size_t NumP,
          typename... RotationT, typename... QuantumT,
          std::size_t NumPProvided = sizeof...(RotationT),
          std::enable_if_t<NumP == NumPProvided, std::size_t> = 0>
void applyQuantumOperation(const std::string &gateName,
                           const std::tuple<RotationT...> &paramTuple,
                           const std::tuple<QuantumT...> &quantumTuple) {

  std::vector<double> parameters;
  cudaq::tuple_for_each(paramTuple,
                        [&](auto &&element) { parameters.push_back(element); });

  std::vector<QuditInfo> qubits;
  qubitsToQuditInfos(quantumTuple, qubits);

  std::vector<bool> qubitIsNegated;
  findQubitNegations(quantumTuple, qubitIsNegated);

  assert(qubitIsNegated.size() == qubits.size() && "qubit mismatch");

  // Catch the case where we have multi-target broadcast, we don't allow that
  if (std::is_same_v<mod, base> && NumT > 1 && qubits.size() > NumT)
    throw std::runtime_error(
        "cudaq does not support broadcast for multi-qubit operations.");

  // Operation on correct number of targets, no controls, possible broadcast
  if ((std::is_same_v<mod, base> || std::is_same_v<mod, adj>)&&NumT == 1) {
    for (auto &qubit : qubits)
      getExecutionManager()->apply(gateName, parameters, {}, {qubit},
                                   std::is_same_v<mod, adj>);
    return;
  }

  // Partition out the controls and targets
  std::size_t numControls = qubits.size() - NumT;
  std::vector<QuditInfo> targets(qubits.begin() + numControls, qubits.end()),
      controls(qubits.begin(), qubits.begin() + numControls);

  // Apply X for any negations
  for (std::size_t i = 0; i < controls.size(); i++)
    if (qubitIsNegated[i])
      getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Apply the gate
  getExecutionManager()->apply(gateName, parameters, controls, targets,
                               std::is_same_v<mod, adj>);

  // Reverse any negations
  for (std::size_t i = 0; i < controls.size(); i++)
    if (qubitIsNegated[i])
      getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Reset the negations
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      if (element.is_negative())
        element.negate();
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        if (q.is_negative())
          q.negate();
    }
  });
}

template <typename mod, std::size_t NUMT, std::size_t NUMP, typename... Args>
void genericApplicator(const std::string &gateName, Args &&...args) {
  applyQuantumOperation<mod, NUMT, NUMP>(
      gateName, tuple_slice<NUMP>(std::forward_as_tuple(args...)),
      tuple_slice_last<sizeof...(Args) - NUMP>(std::forward_as_tuple(args...)));
}

} // namespace details

#define CUDAQ_DEFINE_QUANTUM_OPERATION(NAME, NUMT, NUMP)                       \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  CUDAQ_MOD_TEMPLATE                                                           \
  void NAME(Args &&...args) {                                                  \
    details::genericApplicator<mod, NUMT, NUMP>(#NAME,                         \
                                                std::forward<Args>(args)...);  \
  }                                                                            \
  template <typename... Args>                                                  \
  inline void c##NAME(Args &&...args) {                                        \
    static_assert(sizeof...(Args) == NUMT + NUMP + 1,                          \
                  "Invalid number of target qubits provided to " #NAME);       \
    details::genericApplicator<ctrl, NUMT, NUMP>(#NAME,                        \
                                                 std::forward<Args>(args)...); \
  }

CUDAQ_DEFINE_QUANTUM_OPERATION(h, 1, 0)
CUDAQ_DEFINE_QUANTUM_OPERATION(x, 1, 0)
CUDAQ_DEFINE_QUANTUM_OPERATION(y, 1, 0)
CUDAQ_DEFINE_QUANTUM_OPERATION(z, 1, 0)
CUDAQ_DEFINE_QUANTUM_OPERATION(t, 1, 0)
CUDAQ_DEFINE_QUANTUM_OPERATION(s, 1, 0)
CUDAQ_DEFINE_QUANTUM_OPERATION(rx, 1, 1)
CUDAQ_DEFINE_QUANTUM_OPERATION(ry, 1, 1)
CUDAQ_DEFINE_QUANTUM_OPERATION(rz, 1, 1)
CUDAQ_DEFINE_QUANTUM_OPERATION(r1, 1, 1)
CUDAQ_DEFINE_QUANTUM_OPERATION(u3, 1, 3)
CUDAQ_DEFINE_QUANTUM_OPERATION(swap, 2, 0)

// Define common 2 qubit operations.
inline void cnot(qubit &q, qubit &r) {
  details::applyQuantumOperation<ctrl, 1, 0>("x", std::forward_as_tuple(),
                                             std::forward_as_tuple(q, r));
}
inline void ccx(qubit &q, qubit &r, qubit &s) {
  details::applyQuantumOperation<ctrl, 1, 0>("x", std::forward_as_tuple(),
                                             std::forward_as_tuple(q, r, s));
}

// Define common single qubit adjoint operations.
inline void sdg(qubit &q) {
  details::applyQuantumOperation<adj, 1, 0>("s", std::forward_as_tuple(),
                                            std::forward_as_tuple(q));
}
inline void tdg(qubit &q) {
  details::applyQuantumOperation<adj, 1, 0>("t", std::forward_as_tuple(),
                                            std::forward_as_tuple(q));
}

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
void exp_pauli(double theta, QubitRange &&qubits, const char *pauliWord) {
  std::vector<QuditInfo> quditInfos;
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(quditInfos),
                 [](auto &q) { return details::qubitToQuditInfo(q); });
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation, takes a variadic set of
/// qubits, and the number of qubits must be equal to the Pauli word length.
template <typename... QubitArgs>
void exp_pauli(double theta, const char *pauliWord, QubitArgs &...qubits) {

  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{details::qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation with control qubits and a variadic set
/// of qubits. The number of qubits must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QuantumRegister, typename... QubitArgs>
  requires(std::ranges::range<QuantumRegister>)
#else
template <typename QuantumRegister, typename... QubitArgs,
          typename = std::enable_if_t<
              std::is_same_v<std::remove_reference_t<std::remove_cv_t<
                                 decltype(*QuantumRegister().begin())>>,
                             qubit>>>
#endif
void exp_pauli(QuantumRegister &ctrls, double theta, const char *pauliWord,
               QubitArgs &...qubits) {
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return details::qubitToQuditInfo(q); });
  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{details::qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, controls, quditInfos,
                               false, spin_op::from_word(pauliWord));
}

/// @brief Measure an individual qubit, return 0,1 as `bool`
inline measure_result mz(qubit &q) {
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

/// @brief Measure an individual qubit in `x` basis, return 0,1 as `bool`
inline measure_result mx(qubit &q) {
  h(q);
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

// Measure an individual qubit in `y` basis, return 0,1 as `bool`
inline measure_result my(qubit &q) {
  r1(-M_PI_2, q);
  h(q);
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

inline void reset(qubit &q) {
  getExecutionManager()->reset({q.n_levels(), q.id()});
}

// Measure all qubits in the range, return vector of 0,1
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires std::ranges::range<QubitRange>
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
std::vector<measure_result> mz(QubitRange &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.push_back(mz(qq));
  }
  return b;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs);

#if CUDAQ_USE_STD20
template <typename QubitRange, typename... Qs>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange, typename... Qs,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
std::vector<measure_result> mz(QubitRange &qr, Qs &&...qs) {
  std::vector<measure_result> result = mz(qr);
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs) {
  std::vector<measure_result> result = {mz(q)};
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

namespace support {
// Helper to initialize a `vector<bool>` data structure.
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &, char *,
                                             std::size_t);
void __nvqpp_vector_bool_to_initializer_list(void *, const std::vector<bool> &);
}
} // namespace support

// Measure the state in the given spin_op basis.
inline SpinMeasureResult measure(cudaq::spin_op &term) {
  return getExecutionManager()->measure(term);
}

// Cast a measure register to an int64_t.
// This function is classic control code that may run on a QPU.
inline int64_t to_integer(std::vector<measure_result> bits) {
  int64_t ret = 0;
  for (std::size_t i = 0; i < bits.size(); i++) {
    if (bits[i]) {
      ret |= 1UL << i;
    }
  }
  return ret;
}

inline int64_t to_integer(std::string bitString) {
  std::reverse(bitString.begin(), bitString.end());
  return std::stoull(bitString, nullptr, 2);
}

#if CUDAQ_USE_STD20
// This concept tests if `Kernel` is a `Callable` that takes the arguments,
// `Args`, and returns `void`.
template <typename Kernel, typename... Args>
concept isCallableVoidKernel = requires(Kernel &&k, Args &&...args) {
  { k(args...) } -> std::same_as<void>;
};

template <typename T, typename Signature>
concept signature = std::is_convertible_v<T, std::function<Signature>>;

template <typename T>
concept takes_qubit = signature<T, void(qubit &)>;

template <typename T>
concept takes_qvector = signature<T, void(qvector<> &)>;
#endif

// Control the given cudaq kernel on the given control qubit
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel, qubit &control, Args &&...args) {
  std::vector<std::size_t> ctrls{control.id()};
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given register of control qubits
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename QuantumRegister, typename... Args>
  requires std::ranges::range<QuantumRegister> &&
           isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename QuantumRegister, typename... Args,
          typename = std::enable_if_t<
              !std::is_same_v<
                  std::remove_reference_t<std::remove_cv_t<QuantumRegister>>,
                  cudaq::qubit> &&
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel, QuantumRegister &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (std::size_t i = 0; i < ctrl_qubits.size(); i++) {
    ctrls.push_back(ctrl_qubits[i].id());
  }
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given list of references to control
// qubits.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel,
             std::vector<std::reference_wrapper<qubit>> &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (auto &cq : ctrl_qubits) {
    ctrls.push_back(cq.get().id());
  }
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Apply the adjoint of the given cudaq kernel
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void adjoint(QuantumKernel &&kernel, Args &&...args) {
  // static_assert(true, "adj not implemented yet.");
  getExecutionManager()->startAdjointRegion();
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endAdjointRegion();
}

/// Instantiate this type to affect C A C^dag, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_action {
#if CUDAQ_USE_STD20
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
#else
template <
    typename ComputeFunction, typename ActionFunction,
    typename = std::enable_if_t<std::is_invocable_r_v<void, ComputeFunction> &&
                                std::is_invocable_r_v<void, ActionFunction>>>
#endif
void compute_action(ComputeFunction &&c, ActionFunction &&a) {
  c();
  a();
  adjoint(c);
}

/// Instantiate this type to affect C^dag A C, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_dag_action {
#if CUDAQ_USE_STD20
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
#else
template <
    typename ComputeFunction, typename ActionFunction,
    typename = std::enable_if_t<std::is_invocable_r_v<void, ComputeFunction> &&
                                std::is_invocable_r_v<void, ActionFunction>>>
#endif
void compute_dag_action(ComputeFunction &&c, ActionFunction &&a) {
  adjoint(c);
  a();
  c();
}

/// Helper function to extract a slice of a `std::vector<T>` to be used within
/// CUDA-Q kernels.
#if CUDAQ_USE_STD20
template <typename T>
  requires(std::is_arithmetic_v<T>)
#else
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
std::vector<T> slice_vector(std::vector<T> &original, std::size_t start,
                            std::size_t count) {
  std::vector<double> ret(original.begin() + start,
                          original.begin() + start + count);
  return ret;
}

} // namespace cudaq

#define __qop__ __attribute__((annotate("user_custom_quantum_operation")))

#define CUDAQ_EXTEND_OPERATIONS(NAME, NUMT, NUMP, ...)                         \
  namespace cudaq {                                                            \
  struct CONCAT(NAME, _operation) : public ::cudaq::unitary_operation {        \
    std::vector<std::complex<double>>                                          \
    unitary(const std::vector<double> &parameters) const override {            \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  };                                                                           \
  CUDAQ_MOD_TEMPLATE                                                           \
  void NAME(Args &&...args) {                                                  \
    cudaq::getExecutionManager()->registerOperation<CONCAT(NAME, _operation)>( \
        #NAME);                                                                \
    details::genericApplicator<mod, NUMT, NUMP>(#NAME,                         \
                                                std::forward<Args>(args)...);  \
  }                                                                            \
  }                                                                            \
  extern "C" __qop__ void CONCAT(NAME, _generator)(                            \
      const double *params, std::size_t numParams,                             \
      std::complex<double> *output) {                                          \
    std::vector<double> input(params, params + numParams);                     \
    cudaq::CONCAT(NAME, _operation) op;                                        \
    auto tmpOutput = op.unitary(input);                                        \
    for (int i = 0; i < tmpOutput.size(); i++)                                 \
      output[i] = tmpOutput[i];                                                \
    return;                                                                    \
  }
