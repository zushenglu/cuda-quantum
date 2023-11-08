/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/Todo.h"

#include <pybind11/eval.h>
#include <pybind11/stl.h>
#include <variant>

#include "py_observe.h"
#include "utils/OpaqueArguments.h"

#include "cudaq/platform/quantum_platform.h"

#include "common/Logger.h"

namespace cudaq {

enum class PyParType { thread, mpi };

/// @brief Default qpu id value set to 0
constexpr int defaultQpuIdValue = 0;

/// @brief Global cache map of OpaqueArguments
///
/// For asynchronous execution, we need to construct OpaqueArguments
/// outside of the async lambda invocation. If we don't, then we will be
/// using Python types outside of the current GIL context. Bad things happen
/// then.
std::unordered_map<std::size_t, std::unique_ptr<OpaqueArguments>>
    asyncArgsHolder;

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
observe_result pyObserve(kernel_builder<> &kernel, spin_op &spin_operator,
                         py::args args, int shots,
                         std::optional<noise_model> noise) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  auto &platform = cudaq::get_platform();
  if (noise)
    platform.set_noise(&*noise);

  // TODO: would like to handle errors in the case that
  // `kernel.num_qubits() >= spin_operator.num_qubits()`
  kernel.jitCode();
  auto name = kernel.name();

  // Launch the observation task
  auto result = details::runObservation(
                    [&]() mutable {
                      OpaqueArguments argData;
                      packArgs(argData, validatedArgs);
                      kernel.jitAndInvoke(argData.data());
                    },
                    spin_operator, platform, shots, name)
                    .value();
  platform.reset_noise();
  return result;
}

/// @brief Asynchronously run `cudaq::observe` on the provided kernel and
/// spin operator.
async_observe_result pyObserveAsync(kernel_builder<> &kernel,
                                    spin_op &spin_operator, py::args args = {},
                                    std::size_t qpu_id = defaultQpuIdValue,
                                    int shots = defaultShotsValue) {

  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  std::hash<std::string> hasher;

  // Create a unique integer key that combines the kernel name
  // and the validated args.
  std::size_t uniqueHash = hasher(kernel.name()) + hasher(py::str(args));

  // Add the opaque args to the holder and pack the args into it.
  // Note: this pyObserveAsync is executed in a loop (over spin_operator slices)
  // on the main Python thread, while posting functors to other threads
  // (runObservationAsync below). These functors are expected to operate on the
  // same (kernel + params) configuration, hence, making sure that we don't
  // overwrite asyncArgsHolder while running the spin_operator slicing loop
  // (this pyObserveAsync function).
  if (asyncArgsHolder.find(uniqueHash) == asyncArgsHolder.end()) {
    asyncArgsHolder.emplace(uniqueHash, std::make_unique<OpaqueArguments>());
    packArgs(*asyncArgsHolder.at(uniqueHash).get(), validatedArgs);
  }

  // TODO: would like to handle errors in the case that
  // `kernel.num_qubits() >= spin_operator.num_qubits()`
  kernel.jitCode();
  auto name = kernel.name();
  // Get the platform, first check that the given qpu_id is valid
  auto &platform = cudaq::get_platform();

  // Launch the asynchronous execution.
  return details::runObservationAsync(
      [&kernel, uniqueHash]() mutable {
        auto &argData = asyncArgsHolder.at(uniqueHash);
        kernel.jitAndInvoke(argData->data());
      },
      spin_operator, platform, shots, name, qpu_id);
}

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
observe_result pyObservePar(const PyParType &type, kernel_builder<> &kernel,
                            spin_op &spin_operator, py::args args = {},
                            int shots = defaultShotsValue,
                            std::optional<noise_model> noise = std::nullopt) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  auto &platform = cudaq::get_platform();
  if (!platform.supports_task_distribution())
    throw std::runtime_error(
        "The current quantum_platform does not support parallel distribution "
        "of observe() expectation value computations.");

  // FIXME Handle noise modeling with parallel distribution.
  if (noise)
    TODO("Handle Noise Models with python parallel distribution.");

  // TODO: would like to handle errors in the case that
  // `kernel.num_qubits() >= spin_operator.num_qubits()`
  kernel.jitCode();
  auto name = kernel.name();
  auto nQpus = platform.num_qpus();
  if (type == PyParType::thread) {
    // Does this platform expose more than 1 QPU
    // If so, let's distribute the work amongst the QPUs
    if (nQpus == 1)
      printf(
          "[cudaq::observe warning] distributed observe requested but only 1 "
          "QPU available. no speedup expected.\n");
    return details::distributeComputations(
        [&](std::size_t i, spin_op &op) {
          return pyObserveAsync(kernel, op, args, i, shots);
        },
        spin_operator, nQpus);
  }

  if (!mpi::is_initialized())
    throw std::runtime_error("Cannot use mpi multi-node observe() without "
                             "MPI (did you initialize MPI?).");

  // Necessarily has to be MPI
  // Get the rank and the number of ranks
  auto rank = mpi::rank();
  auto nRanks = mpi::num_ranks();

  // Each rank gets a subset of the spin terms
  auto spins = spin_operator.distribute_terms(nRanks);

  // Get this rank's set of spins to compute
  auto localH = spins[rank];

  // Distribute locally, i.e. to the local nodes QPUs
  auto localRankResult = details::distributeComputations(
      [&](std::size_t i, spin_op &op) {
        return pyObserveAsync(kernel, op, args, i, shots);
      },
      localH, nQpus);

  // combine all the data via an all_reduce
  auto exp_val = localRankResult.expectation();
  auto globalExpVal = mpi::all_reduce(exp_val, std::plus<double>());
  return observe_result(globalExpVal, spin_operator);
}

/// @brief Broadcast the observe call over the list-like arguments provided.
std::vector<observe_result>
pyObserveN(kernel_builder<> &kernel, spin_op &op, py::args args = {},
           std::size_t shots = defaultShotsValue,
           std::optional<noise_model> noise = std::nullopt) {

  auto argSet = createArgumentSet(args);
  auto N = argSet.size();
  auto &platform = cudaq::get_platform();
  if (noise)
    platform.set_noise(&*noise);
  kernel.jitCode();
  auto name = kernel.name();
  std::vector<observe_result> results;
  for (std::size_t currentIter = 0; auto &a : argSet) {

    // Ensure the user input is correct.
    auto validatedArgs = validateInputArguments(kernel, a);
    // Launch the observation task
    auto ret = details::runObservation(
                   [&]() mutable {
                     OpaqueArguments argData;
                     packArgs(argData, validatedArgs);
                     kernel.jitAndInvoke(argData.data());
                   },
                   op, platform, shots, name, 0, nullptr, currentIter, N)
                   .value();
    currentIter++;
    results.push_back(ret);
  }

  platform.reset_noise();

  return results;
}

void bindObserve(py::module &mod) {
  auto parallelSubmodule = mod.def_submodule("parallel");

  py::class_<cudaq::parallel::mpi>(
      parallelSubmodule, "mpi",
      "Type indicating that the :func:`observe` function should distribute its "
      "expectation value computations accross available MPI ranks and GPUs for "
      "each term.");
  py::class_<cudaq::parallel::thread>(
      parallelSubmodule, "thread",
      "Type indicating that the :func:`observe` function should distribute its "
      "term "
      "expectation value computations across available GPUs via standard C++ "
      "threads.");

  mod.def(
      "observe",
      [&](py::object kernel,
          std::variant<spin_op, std::vector<spin_op>> &spin_operator,
          py::args args, int shots_count, std::optional<noise_model> noise,
          std::size_t qpuId)
          -> std::variant<observe_result, std::vector<observe_result>> {
        // Do we have enough information to broadcast? And if so
        // is the first arg a list
        auto [disallowBroadCast, firstArgIsList] =
            kernelCallableInputArgAnalysis(kernel);

        // Get the kernel name
        auto kernelName = py::hasattr(kernel, "kernelFunction")
                              ? kernel.attr("kernelFunction")
                                    .attr("__name__")
                                    .cast<std::string>()
                              : kernel.cast<kernel_builder<> &>().name();
        auto &platform = cudaq::get_platform();
        if (noise)
          set_noise(*noise);

        if (!disallowBroadCast && isBroadcastRequest(firstArgIsList, args)) {
          auto argSet = createArgumentSet(args);
          auto N = argSet.size();

          std::vector<observe_result> results;
          for (std::size_t currentIter = 0; auto &a : argSet) {
            // Launch the observation task
            auto ret = details::runObservation(
                           [&]() mutable { kernel(*a); },
                           std::get<0>(spin_operator), platform, shots_count,
                           kernelName, 0, nullptr, currentIter, N)
                           .value();
            currentIter++;
            results.push_back(ret);
          }

          if (noise)
            platform.reset_noise();

          return results;
        }

        // We are not broadcasting, regular observe call
        if (qpuId >= platform.num_qpus())
          throw std::runtime_error("Invalid qpu_id, platform only has " +
                                   std::to_string(platform.num_qpus()) +
                                   " qpus.");

        // We could have been given a list of spin_ops, check here
        spin_op op;
        auto spinVariantIndex = spin_operator.index();
        if (spinVariantIndex == 1) {
          for (auto &o : std::get<std::vector<spin_op>>(spin_operator))
            op += o;
          op -= spin_op();
        } else
          op = std::get<spin_op>(spin_operator);

        // Execute.
        auto result =
            details::runObservation([&]() mutable { kernel(*args); }, op,
                                    platform, shots_count, kernelName, qpuId)
                .value();

        // Reset the noise
        if (noise)
          platform.reset_noise();

        // Return the result if num spin_ops was 1
        if (spinVariantIndex == 0)
          return result;

        // if it was a list of spin_op, process the single result
        // back into a vector of results.
        std::vector<observe_result> results;
        for (auto &o : std::get<std::vector<spin_op>>(spin_operator))
          results.emplace_back(result.exp_val_z(o), o, result.counts(o));
        return results;
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("shots_count") = defaultShotsValue,
      py::arg("noise_model") = py::none(), py::arg("qpu_id") = 0,
      R"#(Compute the expected value of the `spin_operator` with respect to 
the `kernel`. If the input `spin_operator` is a list of `SpinOperator` then compute 
the expected value of every operator in the list and return a list of results.
If the kernel accepts arguments, it will be evaluated 
with respect to `kernel(*arguments)`. Each argument in `arguments` provided
can be a `list` or `ndarray` of arguments of the specified kernel argument
type, and in this case, the `observe` functionality will be broadcasted over
all argument sets and a list of `observe_result` instances will be returned.

Args:
  kernel (:class:`Callable`): The CUDA Quantum kernel function to evaluate the 
    expectation value with respect to.
  spin_operator (:class:`SpinOperator` or `list[SpinOperator]`): The Hermitian spin operator to 
    calculate the expectation of, or a list of such operators.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to add 
    noise to the kernel execution on the simulator. Defaults to an empty 
    noise model. Key-word only.
  qpu_id (Optional[int]): The optional identification for which QPU on 
    the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`ObserveResult` : A data-type containing the expectation value 
    of the `spin_operator` with respect to the `kernel(*arguments)`, 
    or a list of such results in the case of `observe` function broadcasting. If 
    `shots_count` was provided, the :class:`ObserveResult` will also contain a 
    :class:`SampleResult` dictionary.)#");

  mod.def(
      "observe",
      [&](kernel_builder<> &kernel,
          std::variant<spin_op, std::vector<spin_op>> &spin_operator,
          py::args arguments, int shots, std::optional<noise_model> noise,
          std::optional<py::type> execution)
          -> std::variant<observe_result, std::vector<observe_result>,
                          std::vector<std::vector<observe_result>>> {
        // Observe can be a single observe call, a parallel observe call,
        // or a observe broadcast. We'll handle them all here.

        using ObserveApplicator = std::function<std::vector<observe_result>(
            kernel_builder<> &, spin_op &, py::args &, int,
            std::optional<noise_model>)>;

        std::unordered_map<std::string, ObserveApplicator> applicator{
            {"default",
             [](kernel_builder<> &kernel, spin_op &spin_operator,
                py::args arguments, int shots,
                std::optional<noise_model> noise) {
               return std::vector<observe_result>{
                   pyObserve(kernel, spin_operator, arguments, shots, noise)};
             }},
            {"broadcast",
             [](kernel_builder<> &kernel, spin_op &spin_operator,
                py::args arguments, int shots,
                std::optional<noise_model> noise) {
               return pyObserveN(kernel, spin_operator, arguments, shots,
                                 noise);
             }},
            {"thread",
             [](kernel_builder<> &kernel, spin_op &spin_operator,
                py::args arguments, int shots,
                std::optional<noise_model> noise) {
               return std::vector<observe_result>{
                   pyObservePar(PyParType::thread, kernel, spin_operator,
                                arguments, shots, noise)};
             }},
            {"mpi", [](kernel_builder<> &kernel, spin_op &spin_operator,
                       py::args arguments, int shots,
                       std::optional<noise_model> noise) {
               return std::vector<observe_result>{
                   pyObservePar(PyParType::mpi, kernel, spin_operator,
                                arguments, shots, noise)};
             }}};

        std::string applicatorKey = "default";
        if (isBroadcastRequest(kernel, arguments)) {
          applicatorKey = "broadcast";
          // We will always ignore execution={thread/mpi} if
          // a function broadcast has been requested.
          execution = std::nullopt;
        }

        // If thread or mpi parallelization requested
        // let's set ourselves to the correct applicator
        if (execution)
          applicatorKey = py::str(execution.value().attr("__name__"));

        spin_op op;
        auto spinVariantIndex = spin_operator.index();
        if (spinVariantIndex == 1) {
          for (auto &o : std::get<std::vector<spin_op>>(spin_operator))
            op += o;
          op -= spin_op();
        } else {
          op = std::get<spin_op>(spin_operator);
        }

        // Run the observation task
        auto result =
            applicator[applicatorKey](kernel, op, arguments, shots, noise);

        // We can have a vector of observe_results here
        if (result.size() == 1) {
          // If we just have a single result, check that
          // the input spin_operator was just a spin_op,
          // if so return the result
          if (spinVariantIndex == 0)
            return {result[0]};

          // if it was a list of spin_op, process the single result
          // back into a vector of results.
          std::vector<observe_result> results;
          for (auto &o : std::get<std::vector<spin_op>>(spin_operator))
            results.emplace_back(result[0].expectation(o), o,
                                 result[0].counts(o));
          return results;
        }

        // Check if this is a nested broadcast (sweeping both spin_op and
        // params). If so, peel the results: first index is param sweep, second
        // index is spin_op.
        if (spinVariantIndex == 1) {
          std::vector<std::vector<observe_result>> results;
          for (auto &paramSweepResult : result) {
            std::vector<observe_result> spinResults;
            for (auto &o : std::get<std::vector<spin_op>>(spin_operator))
              spinResults.emplace_back(paramSweepResult.expectation(o), o,
                                       paramSweepResult.counts(o));
            results.emplace_back(std::move(spinResults));
          }
          return results;
        }
        // Return the vector of results, this is for observe_n
        return {result};
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("shots_count") = defaultShotsValue,
      py::arg("noise_model") = py::none(), py::arg("execution") = py::none(),
      R"#(Compute the expected value of the `spin_operator` with respect to 
the `kernel`. If the input `spin_operator` is a list of `SpinOperator` then compute 
the expected value of every operator in the list and return a list of results.
If the kernel accepts arguments, it will be evaluated 
with respect to `kernel(*arguments)`. Each argument in `arguments` provided
can be a list or ndarray of arguments of the specified kernel argument
type, and in this case, the `observe` functionality will be broadcasted over
all argument sets and a list of `observe_result` instances will be returned.
If both the input `spin_operator` and `arguments` are broadcast lists, 
a nested list of results over `arguments` then `spin_operator` will be returned.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
    expectation value with respect to.
  spin_operator (:class:`SpinOperator` or `list[SpinOperator]`): The Hermitian spin operator to 
    calculate the expectation of, or a list of such operators.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to add 
    noise to the kernel execution on the simulator. Defaults to an empty 
    noise model.

Returns:
  :class:`ObserveResult`: 
    A data-type containing the expectation value of the `spin_operator` with 
    respect to the `kernel(*arguments)`, or a list of such results in the case 
    of `observe` function broadcasting. If `shots_count` was provided, the 
    :class:`ObserveResult` will also contain a :class:`SampleResult` dictionary.)#");

  /// Expose observe_async, can optionally take the qpu_id to target.
  mod.def(
      "observe_async",
      [&](kernel_builder<> &kernel, spin_op &spin_operator, py::args arguments,
          std::size_t qpu_id, int shots,
          std::optional<noise_model> noise_model) {
        if (!noise_model)
          return pyObserveAsync(kernel, spin_operator, arguments, qpu_id,
                                shots);
        set_noise(*noise_model);
        auto res =
            pyObserveAsync(kernel, spin_operator, arguments, qpu_id, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("qpu_id") = defaultQpuIdValue,
      py::arg("shots_count") = defaultShotsValue,
      py::arg("noise_model") = py::none(),
      R"#(Compute the expected value of the `spin_operator` with respect to 
the `kernel` asynchronously. If the kernel accepts arguments, it will 
be evaluated with respect to `kernel(*arguments)`. When targeting a
quantum platform with more than one QPU, the optional `qpu_id` allows
for control over which QPU to enable. Will return a future whose results
can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
    expectation value with respect to.
  spin_operator (:class:`SpinOperator`): The Hermitian spin operator to 
    calculate the expectation of.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  qpu_id (Optional[int]): The optional identification for which QPU on 
    the platform to target. Defaults to zero. Key-word only.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional 
    :class:`NoiseModel` to add noise to the kernel execution on the simulator.
    Defaults to an empty noise model.

Returns:
  :class:`AsyncObserveResult`: 
  A future containing the result of the call to observe.)#");

  mod.def(
      "observe_n",
      [](kernel_builder<> &self, spin_op &spin_operator, py::args args,
         int shots, std::optional<noise_model> noise_model) {
        PyErr_WarnEx(PyExc_DeprecationWarning,
                     "observe_n() is deprecated, use observe() with the same "
                     "argument-list structure.",
                     1);
        return pyObserveN(self, spin_operator, args, shots, noise_model);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("shots_count") = defaultShotsValue,
      py::arg("noise_model") = py::none(),
      R"#(Broadcast the observe function over the input argument set.
For each argument type in the kernel signature, you must provide a list of 
arguments of that type. This function computes the expected value of the 
given `spin_operator` with respect to the `kernel` at each set of arguments 
provided for the specified number of circuit executions (`shots_count`).

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` 
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel.
    Each argument must be a list of instances of the type specified by 
    the kernel signature. function at. Leave empty if the kernel doesn't 
    accept any arguments.
  shots_count (Optional[int]): The number of kernel executions on the 
      QPU. Defaults to 1000. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to add 
      noise to the kernel execution on the simulator. Defaults to an empty 
      noise model.

Returns:
  :class:`SampleResult`: A dictionary containing the measurement count results 
    for the :class:`Kernel`.)#");
}

} // namespace cudaq
