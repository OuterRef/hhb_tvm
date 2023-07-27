# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, missing-docstring, unused-import
# pylint: disable=import-outside-toplevel
"""
Relay pass transformation infrastructure.
"""
from collections import defaultdict

import functools
import inspect
import itertools
import types
import warnings

import numpy as np
import math
import tvm.ir
from tvm import relay, te
from tvm.runtime import ndarray as _nd
from tvm.tir.expr import ExprOp

# from ..quantize import _forward_op
# from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, rewrite

from . import _ffi_api
from ..backend.utils import mangle_module_name
from .. import function as _function

# from .. import op as _op
TILING_FUNCS = {}

def build_config(opt_level=2, required_pass=None, disabled_pass=None, trace=None):
    """Configure the build behavior by setting config variables. This function
    will be deprecated in TVM v0.7. Instead, we should directly use
    tvm.transform.PassContext.

    Parameters
    ----------
    opt_level: int, optional
        Optimization level. The optimization pass name and level are as the
        following:

        .. code-block:: python

            OPT_PASS_LEVEL = {
                "SimplifyInference": 0,
                "OpFusion": 1,
                "FoldConstant": 2,
                "FoldScaleAxis": 3,
                "AlterOpLayout": 3,
                "CanonicalizeOps": 3,
                "CanonicalizeCast": 3,
                "EliminateCommonSubexpr": 3,
                "CombineParallelConv2D": 4,
                "CombineParallelDense": 4,
                "CombineParallelBatchMatmul": 4,
                "FastMath": 4
            }

    required_pass: set of str, optional
        Optimization passes that are required regardless of optimization level.

    disabled_pass: set of str, optional
        Optimization passes to be disabled during optimization.

    trace: Callable[[IRModule, PassInfo, bool], None]
        A tracing function for debugging or introspection.

    Returns
    -------
    pass_context: PassContext
        The pass context for optimizations.
    """
    warnings.warn(
        "relay.build_config will be deprecated. Please use \
                  tvm.transform.PassContext directly",
        DeprecationWarning,
    )
    return tvm.transform.PassContext(opt_level, required_pass, disabled_pass, trace)


@tvm._ffi.register_object("relay.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relay.Function in a module. A function
    pass class should be created through `function_pass`.
    """


def InferType():
    """Infer the type of an expr.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered type inference pass.
    """
    return _ffi_api.InferType()


def InferTypeLocal(expr):
    """Infer the type of a single expr, reusing type information to do so.

    This populates the checked_type field in expr. We assume existing type information
    in the graph is correct!

    Parameters
    ----------
    expr: relay.Expr
        The expression we want to know the type of

    Returns
    -------
    type: relay.Type
        The type of the expression
    """
    return _ffi_api.InferTypeLocal(expr)


def FoldScaleAxis():
    """Fold the scaling of axis into weights of conv2d/dense. This pass will
    invoke both forward and backward scale folding.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to fold expressions.

    Note
    ----
    Internally, we will call backward_fold_scale_axis before using
    forward_fold_scale_axis as backward folding targets the common conv->bn
    pattern.
    """
    return _ffi_api.FoldScaleAxis()


def BackwardFoldScaleAxis():
    """Backward fold axis scaling into weights of conv2d/dense.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to backward fold expressions.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis as backward folding targets the common
    conv->bn pattern.
    """
    return _ffi_api.BackwardFoldScaleAxis()


def RemoveUnusedFunctions(entry_functions=None):
    """Remove unused global relay functions in a relay module.

    Parameters
    ----------
    entry_functions: list[string]
        The set of entry functions to start from.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to remove unused functions.
    """
    if entry_functions is None:
        entry_functions = ["main"]
    return _ffi_api.RemoveUnusedFunctions(entry_functions)


def ForwardFoldScaleAxis():
    """Fold the scaling of axis into weights of conv2d/dense.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to forward fold expressions.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis, as backward folding targets the
    common conv->bn pattern.
    """
    return _ffi_api.ForwardFoldScaleAxis()


def SimplifyInference():
    """Simplify the data-flow graph for inference phase. An simplified expression
    which is semantically equal to the input expression will be returned.

    Note that batch norms will only be simplified if their result is indexed at
    tuple index 0.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to perform operator simplification.

    """
    return _ffi_api.SimplifyInference()


def FastMath():
    """Converts the expensive non linear functions to their fast but approximate counterparts.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to perform fast math operations.
    """
    return _ffi_api.FastMath()


def CanonicalizeOps():
    """Canonicalize special operators to basic operators.
    This can simplify followed analysis, e.g. expanding bias_add to
    expand_dims and broadcast_add.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass performing the canonicalization.
    """
    return _ffi_api.CanonicalizeOps()


def DeadCodeElimination(inline_once=False, ignore_impurity=False):
    """Remove expressions that do not have any users (dead code).

    Parameters
    ----------
    inline_once: Optional[Bool]
        Whether to inline a binding that is referenced exactly once.
    ignore_impurity: Optional[Bool]
        Whether to ignore possible side-effects in let-bound expressions.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that eliminates the dead code in a Relay program.
    """
    return _ffi_api.DeadCodeElimination(inline_once, ignore_impurity)


def LazyGradientInit():
    """Reduces memory usage of gradient tensors

    Parameters
    ----------

    Returns
    -------
    ret: tvm.transform.Pass
        A pass which delays and/or reduces memory allocation,
        by lazily allocating 0 or one filled tensors.
    """
    return _ffi_api.LazyGradientInit()


def FoldConstantExpr(expr, mod, fold_qnn=False):
    """Fold the constant expressions in a Relay program.
    Parameters
    ----------
    expr: Expr
        The expression to fold
    mod: IRModule
        The module the expr lives in (for global calls)
    fold_qnn: bool
        Whether to fold constants for QNN operations.

    Returns
    -------
    new_expr: Expr
        The expr after Constant Folding
    """
    return _ffi_api.FoldConstantExpr(expr, mod, fold_qnn)


def FoldConstant(fold_qnn=False):
    """Fold the constant expressions in a Relay program.

    Because of backward compatibility reason it skips QNN primitives from folding by default.
    There are some transformation passes like FakeQuantizationToInteger, which requires to keep QNN
    primitives for constant subgraphs. Uncontrolled constant folding of QNN primitives may break
    applicability of FakeQuantizationToInteger. We suggest to use FoldConstant pass with none
    default fold_qnn=True value only when all other QNN sensitive passes were already applied.

    Parameters
    ----------
    fold_qnn: bool
        Whether to fold constants for QNN operations.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for constant folding.
    """
    return _ffi_api.FoldConstant(fold_qnn)


def FuseOps(fuse_opt_level=-1):
    """Fuse operators in an expr to a larger operator according to some rules.

    Parameters
    ----------
    fuse_opt_level : int
        The level of fuse optimization. -1 indicates that the level will be
        inferred from pass context.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for operator fusion.
    """
    return _ffi_api.FuseOps(fuse_opt_level)


def DefuseOps():
    """The inverse operation of FuseOps. It transforms a fused program returned by FuseOps into the
    program before FuseOps. (i.e., x == DefuseOps(FuseOps(x)))

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for operator defusion.
    """
    return _ffi_api.DefuseOps()


def CombineParallelConv2D(min_num_branches=3):
    """Combine multiple conv2d operators into one.

    Parameters
    ----------
    min_num_branches : int
        The minimum number of required parallel branches for performing this
        optimization.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that combines parallel conv2d operators.
    """
    return _ffi_api.CombineParallelConv2D(min_num_branches)


def CombineParallelDense(min_num_branches=3, to_batch=True):
    """Combine multiple dense operators into one. For example:

    .. code-block
                    data
            /              \
        dense (2,2)         dense (2,2)
            |                 |
        elemwise/bcast (2,2)  elemwise/bcast (2,2)

    Would become:

    .. code-block

                data
                |
            batch_matmul+elemwise/bcast (2,2,2)

    or (if to_batch=False)

    .. code-block

                data
                |
            dense+elemwise/bcast (2,2+2)

    Parameters
    ----------
    min_num_branches : int
        The minimum number of required parallel branches for performing this
        optimization.

    to_batch_matmul : bool
        If True, combine parallel dense ops into batch_matmul op.
        If False, combine parallel dense ops into dense op.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that combines parallel dense operators.
    """
    return _ffi_api.CombineParallelDense(min_num_branches, to_batch)


def CombineParallelBatchMatmul(min_num_branches=3):
    """Combine multiple batch matmul operators into one. For example:

    .. code-block
                             data (1, 2, 3)
                         /                  \
        batch_matmul(data, (1, 4, 3))    batch_matmul(data, (1, 5, 3))
            |                                |
        elemwise/bcast (1, 2, 4)         elemwise/bcast (1, 2, 5)

    Would become:

    .. code-block

                data (1, 2, 3)
                |
            batch_matmul(data, (1, 4+5, 3))
                |
            elemwise/bcast (1 ,2, 4+5)

    Parameters
    ----------
    min_num_branches : int
        The minimum number of required parallel branches for performing this
        optimization.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that combines parallel dense operators.
    """
    return _ffi_api.CombineParallelBatchMatmul(min_num_branches)


def BatchingOps():
    """Batching parallel operators into one for Conv2D, Dense and BatchMatmul.

    Returns
    -------
    ret: tvm.transform.Pass
        The sequential pass which apply batching for different operator types.
    """
    return tvm.transform.Sequential(
        [CombineParallelConv2D(), CombineParallelDense(), CombineParallelBatchMatmul()]
    )


def AlterOpLayout():
    """Alternate the layouts of operators or replace primitive operators with
    other expressions.
    This pass can be used for computing convolution in custom layouts or
    other general weight pre-transformation.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that alters the layout of operators.
    """
    return _ffi_api.AlterOpLayout()


class LayoutConfig(object):
    """A structure for customizing the ConvertLayout pass."""

    current = None

    def __init__(self, skip_layers=None):
        self.skip_counter = 0
        self.skip_layers = skip_layers if skip_layers is not None else []

    def check_skip(self):
        skip = self.skip_counter in self.skip_layers
        self.skip_counter += 1
        return skip

    def reset(self):
        self.skip_counter = 0
        self.skip_layers = []

    def __enter__(self):
        self._old_manager = LayoutConfig.current
        LayoutConfig.current = self
        return self

    def __exit__(self, ptype, value, trace):
        LayoutConfig.current = self._old_manager


def ConvertLayout(desired_layouts):
    """Given a dest layout, this pass transforms the expr such that most of the ops input data
    layout is changed to the dest layout. In ideal situation, there are only 2 layout transforms,
    one at the start and one at the end.

    This pass is not a part of relay.build and is expected to be called between framework-relay
    parser and relay.build call. This is very helpful for hardware backends that support/prefer only
    type of data layout.

    RFC - https://discuss.tvm.apache.org/t/layout-conversion-pass/4009

    This pass uses most of the AlterOpLayout and InferCorrectLayout infrastructure. We can define
    new layouts for conv2d ops for now. Most of the other operators try to adapt to their input
    layout using the InferCorrectLayout infrastructure.

    Parameters
    ----------
    desired_layouts : map of op_name to list of layouts
        Specify a mapping of operator names to a list of layouts to convert to, in the order
        defined by the operator. An example for nn.conv2d could be: {"nn.conv2d", ["NHWC", "OHWI]},
        where the first item in the list specifies the data layout and the second specifies the
        kernel layout.

    Returns
    -------
    pass: FunctionPass
      The pass.
    """
    return _ffi_api.ConvertLayout(desired_layouts)


def Legalize(legalize_map_attr_name="FTVMLegalize"):
    """Legalizes an expression with another expression.
    This pass can be used to replace an expr with another expr for target
    dependent optimizations. For example, one expr, though semnatically
    equivalent to the other, can have better performance on a target. This pass
    can be used to legalize the expr in a target-dependent manner.

    Parameters
    ----------
    legalize_map_attr_name : str
        The Op's attr name which corresponds to the legalize rule function.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that rewrites an expr.
    """
    return _ffi_api.Legalize(legalize_map_attr_name)


def MergeComposite(pattern_table):
    """Merge multiple operators into a single composite relay function.

    Parameters
    ----------
    pattern_table : List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, Function]]
        A list of (pattern_name, pattern, check) tuples.
        The order of the patterns in the list will determine the order
        of priority in which they are matched.
        'check' is a function to check whether an extracted pattern matches.
        It can be implemented by pattern writer but if not specified it will
        always return True.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that merges operators into a single composite
        relay function.
    """
    pattern_names = []
    patterns = []
    checks = []
    for tup in pattern_table:
        if len(tup) == 2:
            pattern_name, pattern = tup
            check = lambda extract: True
        elif len(tup) == 3:
            pattern_name, pattern, check = tup

        pattern_names.append(pattern_name)
        patterns.append(pattern)
        checks.append(check)

    return _ffi_api.MergeComposite(pattern_names, patterns, *checks)


def MergeCompilerRegions():
    """Merge together compiler regions.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that merges compiler regions.
    """
    return _ffi_api.MergeCompilerRegions()


def ToANormalForm():
    """Turn Graph Normal Form expression into A Normal Form Expression.
    The scope of the root expression is the global scope.
    The scope of any non root expression is the least common ancestor of all it's scope.
    Values are ordered by post-DFS order in each scope.

    Returns
    -------
    ret : Union[tvm.transform.Pass, tvm.relay.Expr]
        The registered pass that transforms an expression into A Normal Form.
    """
    return _ffi_api.ToANormalForm()


def ToANormalFormExpr(e):
    """ToANormalForm, but on expression level.

    Parameters
    ----------
    e : Expr
        The graph expression.

    Returns
    -------
    ret : Expr
        The transformed expresion.
    """
    return _ffi_api.ToANormalFormExpr(e)


def ToBasicBlockNormalForm():
    """Turn an expression to Basic Block Normal Form.
    We define a block as a group of expressions implied by the scope structure.
    Each graph node can only belong to a single block.
    For any value that is being used in multiple blocks, it has to be referred
    by a Var which is defined in a block, whose scope is the least common ancestor
    of blocks this value is used.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that transforms an expression into Basic Block Normal Form.
    """
    return _ffi_api.ToBasicBlockNormalForm()


def ToCPS(expr, mod=None):
    """
    Turn expression into continuation passing style(CPS).

    Every intermediate compute will be passed to a continuation.

    Returns
    -------
    result: tvm.transform.Pass
        The registered pass that transforms an expression into CPS.
    """
    return _ffi_api.to_cps(expr, mod)


def EtaExpand(expand_constructor=False, expand_global_var=False):
    """Add abstraction over a constructor or global variable bound to a function

    Parameters
    ----------
    expand_constructor: bool
        Whether to expand constructors.

    expand_global_var: bool
        Whether to expand global variables.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that eta expands an expression.
    """
    return _ffi_api.EtaExpand(expand_constructor, expand_global_var)


def ToGraphNormalForm():
    """Turn a Relay program in A Normal Form into Graph Normal Form

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that transforms an expression into Graph Normal Form.
    """
    return _ffi_api.ToGraphNormalForm()


def EliminateCommonSubexpr(fskip=None):
    """Eliminate common subexpressions.

    Parameters
    ----------
    fskip: Callable
        The callback function that decides whether an expression should be
        skipped.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that eliminates common subexpressions.
    """
    return _ffi_api.EliminateCommonSubexpr(fskip)


def PartialEvaluate():
    """Evaluate the static fragment of the code.

    Note
    ----
    This transformation could be either `Module -> Module` or `Expr -> Expr`.
    It will directly transform the input expression to a new one if the target
    expression is provided. Otherwise, it will rely on the pass manager to
    carry out transformation.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that performs partial evaluation on an expression.
    """
    return _ffi_api.PartialEvaluate()


def CanonicalizeCast():
    """
    Canonicalize cast expressions to make operator fusion more efficient.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that canonicalizes cast expression.
    """
    return _ffi_api.CanonicalizeCast()


def LambdaLift():
    """
    Lift the closure to global function.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that lifts the lambda function.
    """
    return _ffi_api.LambdaLift()


def PartitionGraph(mod_name="default", bind_constants=True):
    """Partition a Relay program into regions that can be executed on different
    backends.

    Parameters
    ----------
    mod_name : string
        Controls the prefix of the name of each partitioned subraph.
        If `mod_name` is None, then `tvmgen_` prefix is used.
        Otherwise, `tvmgen_mod_name_` prefix is used.

    bind_constants: bool
        Whether or not to bind constants in partitioned subgraphs. Note that the codegen needs
        to maintain the bound constants; Otherwise the constants will be maintained by
        the metadata module. So it is recommended for C-source based codegens to
        set bind_constants=False to avoid embedding large constants in a C source file.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that partitions the Relay program.
    """
    mod_name = mangle_module_name(mod_name)
    return _ffi_api.PartitionGraph(mod_name, bind_constants)


def AnnotateTarget(targets, include_non_call_ops=True):
    """Annotate ops in an experession with a provied compiler/target and then
    use it for codegen.

    Parameters
    ----------
    targets : str or List[str]
        The list of target compilers used for codegen.
    include_non_call_ops : boolean
        If True then non-call ops also will be annotated with targets
        If False then non-call ops will not be processed

    Returns
    -------
    ret : tvm.transform.Pass
        The annotated pass that wrapps ops with subgraph_start and
        subgraph_end.
    """
    if isinstance(targets, str):
        targets = [targets]
    return _ffi_api.AnnotateTarget(
        [tvm.runtime.container.String(t) for t in targets], include_non_call_ops
    )


def DynamicToStatic():
    """If possible, convert tvm.relay.dynamic* ops to static versions

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for dynamic->static conversion.
    """
    return _ffi_api.DynamicToStatic()


def Inline():
    """Perform inlining on the given Relay IR module. The global functions that
    are marked as `inline` should be always inlined. A cost model will be
    needed in the future to decide if it is profitable to inline the function.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that performs inlining for a Relay IR module.
    """
    return _ffi_api.Inline()


def gradient(expr, mod=None, mode="higher_order"):
    """
    Transform the input function,
    returning a function that calculate the original result,
    paired with gradient of the input.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, which is a Function or a GlobalVar.

    mod : Optional[tvm.IRModule]

    mode : Optional[String]
        The mode of the automatic differentiation algorithm.
        'first_order' only works on first order code, but will not produce
        reference nor closure.
        'higher_order' works on all code using reference and closure.

    Returns
    -------
    expr : tvm.relay.Expr
      The transformed expression.
    """
    if mode == "first_order":
        warnings.warn(
            "using transform.gradient for first-order AD is deprecated, please use the"
            "FirstOrderGradient module pass",
            DeprecationWarning,
        )
        if mod is not None:
            raise RuntimeError(
                "to run first-order AD on a module, please use the FirstOrderGradient module pass."
            )
        return FirstOrderGradient()(tvm.IRModule.from_expr(expr))["main"]
    if mode == "higher_order":
        return _ffi_api.gradient(expr, mod)
    raise Exception("unknown mode")


def FirstOrderGradient():
    """
    Transforms all global functions in the module to return the original result, paired with the
    gradients of the inputs. This pass transforms each global function independently and does not
    support interprocedural AD. Additionally, this pass does not support any control-flow or
    references, and should only be used on pure data-flow graphs.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered FirstOrderGradient pass.
    """
    return _ffi_api.FirstOrderGradient()


def Defunctionalization(func, mod):
    """
    Performs defunctionalization on func,
    transforming func from a higher-order program to a first-order program.

    At each call site, the function is cloned and type parameters are substituted in.
    Function arguments are encoded as datatypes
    and additional apply functions are used for application.

    Parameters
    ----------
    func : tvm.relay.Function
        The input function, which should not be polymorphic or be higher-order.
        This is because all types must be known and we can't encode function arguments
        to the program itself.

    mod : tvm.IRModule
        The IRModule containing function and type definitions,
        which is also mutated during this pass.

    Returns
    -------
    expr : tvm.relay.Function
      The output function.
    """
    return _ffi_api.Defunctionalization(func, mod)


def to_cps(func, mod=None):
    """
    Turn expression into CPS expression.

    Every intermediate compute will be passed to a continuation.

    Parameters
    ----------
    func: tvm.relay.Function
        The input function.

    mod: Optional[tvm.IRModule]
        The global module.

    Returns
    -------
    result: tvm.relay.Function
      The output function.
    """
    use_mod = mod if mod is not None else tvm.ir.IRModule()
    return _ffi_api.to_cps(func, use_mod)


def un_cps(func):
    """
    Turn an cps function into a Function without the continuation argument.

    Note that this will not give the exact same interface as before cps:
      If the input/output is higher order, they will still be in cps form.

    Parameters
    ----------
    func: tvm.relay.Function
        The input function

    Returns
    -------
    result: tvm.relay.Function
        The output function
    """
    return _ffi_api.un_cps(func)


def _wrap_class_function_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""

    class PyFunctionPass(FunctionPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in cass pass_cls creation failed.fg
            self.handle = None
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_function(func, mod, ctx)

            self.__init_handle_by_constructor__(_ffi_api.MakeFunctionPass, _pass_func, pass_info)
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyFunctionPass.__init__, pass_cls.__init__)
    PyFunctionPass.__name__ = pass_cls.__name__
    PyFunctionPass.__doc__ = pass_cls.__doc__
    PyFunctionPass.__module__ = pass_cls.__module__
    return PyFunctionPass


def function_pass(pass_func=None, opt_level=None, name=None, required=None):
    """Decorate a function pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created function pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Function, Module, PassContext) -> Function]]
        The transformation function or class.

    opt_level : int
        The optimization level of this module pass.

    name : Optional[str]
        The name of the function pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the module pass is dependent on.

    Returns
    -------
    create_function_pass : Union[Callable, FunctionPass]

        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new FunctionPass will be returned when we decorate a pass function.
        A new FunctionPass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a function pass class.

    .. code-block:: python

        @relay.transform.function_pass(opt_level=1)
        class TestReplaceFunc:
            def __init__(self, new_func):
                self.new_func = new_func

            def transform_function(self, func, mod, ctx):
                # just for demo purposes
                # transform func to new_func
                return self.new_func

        x = relay.var("x", shape=(10, 20))
        f1 = relay.Function([x], x)
        f2 = relay.Function([x], relay.log(x))
        # fpass is now a special pass that replaces every
        # function to f1
        fpass = TestReplaceFunc(f1)
        # now every function in input_mod is replaced by f1
        res_mod = fpass(input_mod)


    The following code creates a function pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relay.transform.function_pass(opt_level=2)
        def transform(func, mod, ctx):
            # my transformations here.
            return func

        function_pass = transform
        assert isinstance(function_pass, transform.FunctionPass)
        assert function_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = function_pass(m)
        # Now constant folding should have been applied to every function in
        # the provided module m. And the updated module will be returned.
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the function pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_function_pass(pass_arg):
        """Internal function that creates a function pass"""
        fname = name if name else pass_arg.__name__
        info = tvm.transform.PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_api.MakeFunctionPass(pass_arg, info)

    if pass_func:
        return create_function_pass(pass_func)
    return create_function_pass


@function_pass(opt_level=1)
class ChangeBatch:
    """
    Change the batch size.

    Parameters
    ----------
    data: Dict[relay.Var, int]
      A dictionary of all the params to change.
      The keys are all params, and the values are which dimension hold the batch.

    batch_size: int
      The batch size to change to.

    Returns
    -------
    pass: FunctionPass
      The pass.
    """

    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size

    def transform_function(self, func, mod, ctx):
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        change_batch = self

        class ChangeBatchMutator(tvm.relay.ExprMutator):
            def visit_var(self, var):
                if var in change_batch.data:
                    ty = var.type_annotation
                    new_shape = list(ty.shape)
                    new_shape[change_batch.data[var]] = change_batch.batch_size
                    return relay.Var(var.name_hint, relay.TensorType(new_shape, ty.dtype))
                return var

        return ChangeBatchMutator().visit(func)


def DenseToSparse(weight_name, weight_shape):
    """
    Rewrite qualified ```nn.dense operation``` to ```nn.sparse_dense```
    This pass is used in ```data_dep_optimization.bsr_dense```
    Parameters of this pass is generated by ```analysis.sparse_dense.process_params```

    Parameters
    ----------
    weight_name: Array[String]
      Names of weights which qualified sparse contrains

    weight_shape: Array[Array[IntImm]]
      Weights shape in BSR format.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered DenseToSparse pass.
    """
    return _ffi_api.DenseToSparse(weight_name, weight_shape)


def Conv2dToSparse(weight_name, weight_shape, layout, kernel_size):
    """
    Rewrite qualified ```nn.conv2d operation``` to ```nn.sparse_conv2d```

    Parameters
    ----------
    weight_name: Array[String]
      Names of weights which qualified sparse contrains

    weight_shape: Array[Array[IntImm]]
      Weights shape in BSR format.

    layout : str
        layout of data

    Returns
    -------
    ret : tvm.transform.Pass
        The registered DenseToSparse pass.
    """
    return _ffi_api.Conv2dToSparse(weight_name, weight_shape, layout, kernel_size)


def Conv2dToSparse2(layout, kernel_size, blocksize, sparsity_threshold):
    """
    Rewrite freezed ```nn.conv2d``` operation to ```nn.sparse_conv2d```

    Parameters
    ----------
    layout : str
        layout of data

    kernel_size : int
        kernel size of conv2d

    Returns
    -------
    ret : tvm.transform.Pass
        The registered DenseToSparse pass.
    """
    return _ffi_api.Conv2dToSparse2(layout, kernel_size, *blocksize, sparsity_threshold)


def SimplifyFCTranspose(target_weight_name):
    """
    Rewrite ```y = nn.dense(x, transpose(w, [1, 0]))``` to ```y = nn.dense(x, wt)```
    This pass is used in ```data_dep_optimization.simplify_fc_transpose```

    Parameters
    ----------
    weight_name: Array[String]
      Names of weights which qualified ```y = nn.dense(x, transpose(w, [1, 0]))```
      This parameter is generated by ```analysis.search_fc_transpose``` function

    Returns
    -------
    ret : tvm.transform.Pass
        The registered SimplifyFCTranspose pass.
    """
    return _ffi_api.SimplifyFCTranspose(target_weight_name)


def SimplifyExpr():
    """
    Simplify the Relay expression, including merging consecutive reshapes.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered SimplifyExpr pass.
    """
    return _ffi_api.SimplifyExpr()


@function_pass(opt_level=1)
class SpaceToBatch2AtrousConv:
    """Convert space_to_batch + conv + batch_to_space into dilation/atrous conv"""

    def transform_function(self, func, mod, ctx):
        from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, rewrite

        def conv2python(data):
            res = list()
            for d in data:
                if isinstance(d, tvm.ir.container.Array):
                    d_res = list()
                    for dd in d:
                        d_res.append(int(dd))
                    res.append(d_res)
                elif isinstance(d, tvm.tir.expr.IntImm):
                    res.append(int(d))
            return res

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                self.inp = wildcard()
                self.w = wildcard()
                self.b = wildcard()
                self.is_transpose1 = is_op("transpose")(self.inp)
                self.is_space_to_batch_nd = is_op("nn.space_to_batch_nd")(self.is_transpose1)
                self.is_transpose2 = is_op("transpose")(self.is_space_to_batch_nd)
                self.is_conv2d = is_op("nn.conv2d")(self.is_transpose2, self.w)
                self.is_bias_add = is_op("nn.bias_add")(self.is_conv2d, self.b)
                self.is_transpose3 = is_op("transpose")(self.is_bias_add)
                self.is_batch_to_space_nd = is_op("nn.batch_to_space_nd")(self.is_transpose3)
                self.is_transpose4 = is_op("transpose")(self.is_batch_to_space_nd)

                self.pattern = self.is_transpose4

            def callback(self, pre, post, node_map):
                paddings = conv2python(node_map[self.is_space_to_batch_nd][0].attrs.paddings)
                block_shape = conv2python(node_map[self.is_space_to_batch_nd][0].attrs.block_shape)
                crops = conv2python(node_map[self.is_batch_to_space_nd][0].attrs.crops)

                groups = int(node_map[self.is_conv2d][0].attrs.groups)
                channels = int(node_map[self.is_conv2d][0].attrs.channels)
                kernel_size = conv2python(node_map[self.is_conv2d][0].attrs.kernel_size)

                new_paddings = np.subtract(paddings, crops)
                new_paddings = new_paddings.transpose().flatten().tolist()

                new_node = relay.op.nn.conv2d(
                    node_map[self.inp][0],
                    node_map[self.w][0],
                    padding=new_paddings,
                    dilation=block_shape,
                    groups=groups,
                    channels=channels,
                    kernel_size=kernel_size,
                )

                new_node = relay.op.nn.bias_add(new_node, node_map[self.b][0])
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=3)
class AddPreprocessNode:
    """Insert the preprocess node(mul/add) into original module."""

    def __init__(self, mean_value=None, scale_value=None, layout="NCHW"):
        self.mean_value = mean_value
        self.scale_value = scale_value
        self.layout = layout

        def _convert_data(data, layout):
            if not data:
                return data
            new_data = np.array(data, np.float32)
            if new_data.ndim == 1:
                if layout == "NCHW":
                    new_data = np.expand_dims(new_data, (1, 2))
                elif layout == "NHWC":
                    new_data = np.expand_dims(new_data, (0, 1))
                else:
                    raise ValueError("Unsupport for layout: %s in AddPreprocessNode." % layout)
            return new_data

        self.mean_value = _convert_data(self.mean_value, self.layout)
        self.scale_value = _convert_data(self.scale_value, self.layout)

    def transform_function(self, func, mod, ctx):
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        preprocess_params = self

        class InnerMutator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                pre_call = op_args[0]
                if isinstance(pre_call, relay.expr.Var):
                    out = pre_call
                    if (
                        preprocess_params.mean_value is not None
                        and preprocess_params.scale_value is not None
                    ):
                        new_scale = preprocess_params.scale_value
                        new_mean = preprocess_params.scale_value * preprocess_params.mean_value

                        out = relay.op.multiply(out, relay.expr.const(new_scale, "float32"))
                        # pylint: disable=invalid-unary-operand-type
                        out = relay.op.add(out, relay.expr.const(-new_mean, "float32"))
                    elif preprocess_params.mean_value is not None:
                        # pylint: disable=invalid-unary-operand-type
                        out = relay.op.add(
                            out, relay.expr.const(-preprocess_params.mean_value, "float32")
                        )
                    elif preprocess_params.scale_value is not None:
                        out = relay.op.multiply(
                            out, relay.expr.const(preprocess_params.scale_value, "float32")
                        )

                    op_args[0] = out
                return relay.expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)

        return InnerMutator().visit(func)


def PlanDevices(default_device):
    """
    Uses existing "on_device" and "device_copy" calls to infer the virtual device on which
    every Relay sub-expression should run and the result stored. Captures the result of that
    analysis using new "on_device" and "device_copy" calls. Sub-expressions which are
    not otherwise constrained are assigned to the default primitive virtual device describe by
    config. However data and computations which must be hosted on a CPU (such as shapes and
    shape functions) use the host virtual device of the config.

    Parameters
    ----------
    config : tvm.CompilationConfig
        The compilation configuration, specifying available targets and default devices.

    Returns
    -------
    ret : tvm.transforms.Pass
        The pass.
    """
    return _ffi_api.PlanDevices(config)


def ManifestLifetimes():
    """
    Manifest the lifetimes of variables after allocations have been manifested, by inserting kill
    operations once variables become dead.
    """
    return _ffi_api.ManifestLifetimes()


def FoldExplicitPadding():
    """
    FoldExplicitPadding finds explict padding before an op that can support
    implicit padding and fuses them.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered ImplicitPadding pass.
    """
    return _ffi_api.FoldExplicitPadding()


def AnnotateSpans():
    """
    Annotate a program with span information by first generating its textual
    representation and then parsing it back into a Relay AST annotated with
    span information.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered AnnotateSpans pass.
    """
    return _ffi_api.AnnotateSpans()


def FakeQuantizationToInteger(hard_fail=False, use_qat=False):
    # pylint: disable=anomalous-backslash-in-string
    """
    Find regions of the graph of the form

    .. code-block:: text

        x    w
        |    |
        dq   dq
         \\   /
          op1
           |
          op2
           |
           q

    where ``q == qnn.quantize`` and ``dq = qnn.dequantize``
    and rewrite them into integer versions of ``op1`` and ``op2``

    Rules for rewriting indivdual ops are in fake_quantization_to_integer.py

    Parameters
    ----------
    hard_fail : boolean
        How do deal with errors during graph rewriting.
        If true, raise an error.
        If false, skip rewriting the subgraph.

    use_qat : boolean
        To perform an additional QAT pass - convert enabled operations with dequantized inputs.
        Example: in the graph above op2 is not registered with the FakeQuantizationToInteger
        attribute, op1 operation can still be converted. Converted pattern below:

        .. code-block:: text

            x    w
            |    |
            \\   /
              op1
              |
              dq
              |
              op2
              |
              q

    Returns
    -------
    ret : tvm.transform.Pass
        The registered FakeQuantizationToInteger pass.
    """
    return _ffi_api.FakeQuantizationToInteger(hard_fail, use_qat)


def FlattenAtrousConv():
    # pylint: disable=anomalous-backslash-in-string
    """
    The purpose of this pass is to find a sequence of space_to_batch_nd-conv2d-batch_to_space_nd
    operations:

    .. code-block:: text

      x     w
      |     |
      s2b   |
       \\   /
        conv2d
         |
         b2s

    and convert them into subgraphs with a convolution with the modified "dilation" and
    recalculated "padding" parameters.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered FlattenAtrousConv pass.
    """
    return _ffi_api.FlattenAtrousConv()


def ToMixedPrecision(mixed_precision_type="float16", missing_op_mode=1):
    """
    Automatic mixed precision rewriter. Rewrite an FP32 relay graph into a version
    where as many operations as possible are in the target mixed_precision_type.

    Parameters
    ----------
    mixed_precision_type: str
      The target datatype to transform operations in the graph to use.

    missing_op_mode: int
      Determines how to handle ops not registered with FTVMMixedPrecisionConversionType
        0: Does not allow any missing ops. Will throw errors when encountering any.
        1: Allow missing ops but emit warnings.
        2: Allow missing ops and silently ignore them.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass.
    """
    if missing_op_mode < 0 or missing_op_mode > 2:
        raise ValueError("Missing op mode is either 0, 1, or 2")
    return _ffi_api.ToMixedPrecision(mixed_precision_type, missing_op_mode)


def SplitArgs(max_function_args):
    """Split function with huge number of arguments to smaller pieces.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for constant folding.
    """
    return _ffi_api.SplitArgs(max_function_args)


def OutlineCompilerFunctionsWithExistingGlobalSymbols(compiler_filter=""):
    """Outlines all literal functions in direct call positions which have a "Compiler"
    attribute.

    The outlined functions are bound to unique global vars according to their existing
    "global_symbol" attribute. At most one function with the same global symbol is outlined.

    If compiler_filter is non-empty only functions with that as their attribute value are
    outlined.

    This pass may be useful for external codegen using the "RelayToTIR" custom pass mechanism
    to prepare the IRModule before custom lowering.

    Parameters
    ----------
    compiler_filter : String
        If non-empty, the 'compiler' attribute to filter on.

    Returns
    -------
    ret : tvm.transform.Pass
        The pass.
    """
    return _ffi_api.OutlineCompilerFunctionsWithExistingGlobalSymbols(compiler_filter)


def MarkCompilerFunctionsAsExtern(compiler_filter=""):
    """Marks all global functions which have a "Compiler" attribute matching
    compiler_filter as 'extern'.

    The function's attributes are replaced with a single "Extern" attribute, and
    all calls to the function are switched to use the 'call_lowered' calling convention.

    If compiler_filter is non-empty only functions with that as their attribute value are
    outlined.

    This pass may be useful for external codegen using the "RelayToTIR" custom pass mechanism to
    cleanup the IRModule after custom lowering.

    Parameters
    ----------
    compiler_filter : String
        If non-empty, the 'compiler' attribute to filter on.

    Returns
    -------
    ret : tvm.transform.Pass
        The pass.
    """
    return _ffi_api.MarkCompilerFunctionsAsExtern(compiler_filter)


def InlineCompilerFunctionsBoundTo(global_vars):
    """Inlines all global functions bound to a global var in global_vars.

    Both the global "Compiler" attributed function, and any calls to "Composite" functions it its
    body are inlined.

    This pass may be useful for external codegen which needs to undo partitioning based on
    properties of the entire partition.

    Parameters
    ----------
    global_vars : Array[tvm.relay.GlobalVar]
        The global vars of all 'Compiler' functions to inline.

    Returns
    -------
    ret : tvm.transform.Pass
        The pass.
    """
    return _ffi_api.InlineCompilerFunctionsBoundTo(global_vars)

# TILING_FUNCS = {}
# def tiling_func_register(func_name):
#     """Register func in TILING_FUNCS"""

#     def decorator(func):
#         TILING_FUNCS[func_name] = func.__name__

#         def wrapper(self, call, op_args):
            
#             return func(self, op_args, call.attrs)

#         return wrapper

#     return decorator

def Deconv2dToConv2d(mod):
    class Deconv2dToConv2dClass(relay.ExprMutator):
        """Convert qnn model to relay"""

        def __init__(self):
            super(Deconv2dToConv2dClass, self).__init__()
        
        def visit_call(self, call):
            
            op_args = [self.visit(arg) for arg in call.args]
            
            pre_call = op_args[0]

            if call.op.name == "nn.conv2d_transpose":
                if not call.attrs.data_layout == "NCHW" or not call.attrs.kernel_layout == "OIHW" :
                    return relay.expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)

                ori_weight = op_args[1].data.asnumpy()
                b_shape = ori_weight.shape
                in_shape = relay.frontend.common.infer_shape(pre_call)
                # We only support batch size 1
                out_shape = relay.frontend.common.infer_shape(call)
                assert(out_shape[0] == 1)
                stride = call.attrs.strides
                
                # kernel output channels
                original_filter_num = int(b_shape[0])
                # kernel input channels
                original_map_num = int(b_shape[1])
                # kernel input size
                original_row_num = int(b_shape[2])
                original_col_num = int(b_shape[3])
                
                stride_h = int(stride[0])
                stride_w = int(stride[1])

                # partition kernel size
                
                spilt_row_num = int(original_row_num / stride_h) if (original_row_num % stride_h == 0) else int(original_row_num / stride_h + 1) 
                spilt_col_num = int(original_col_num / stride_w) if (original_col_num % stride_w == 0) else int(original_col_num / stride_w + 1) 

                spilt_filter_num = stride_h ** stride_w  # partition number

                trans_w_array = np.zeros((spilt_filter_num, original_filter_num, original_map_num, spilt_row_num, spilt_col_num),
                                        dtype=ori_weight.dtype)

                n_pad = ((0, 0), (0, 0), (spilt_col_num * stride_w - b_shape[3], 0), (spilt_row_num * stride_h - b_shape[2], 0))
                w_array = np.pad(ori_weight, n_pad, 'constant', constant_values=0)
                # transform
                for cur_spilt_num in range(spilt_filter_num):
                    for cur_row_num in range(spilt_row_num):
                        for cur_col_num in range(spilt_col_num):
                            trans_w_array[cur_spilt_num,
                                        :, :, spilt_row_num - cur_row_num - 1,
                                        spilt_col_num - cur_col_num - 1] = w_array[
                                :, :, stride_h * cur_row_num + cur_spilt_num // stride_h,
                                stride_w * cur_col_num + cur_spilt_num % stride_w]
                
                trans_w_array = trans_w_array.transpose(1,0,2,3,4)
                trans_w_array = trans_w_array.reshape(spilt_filter_num * original_filter_num, original_map_num, spilt_row_num, spilt_col_num)
                
                transpose_weight = relay.const(trans_w_array)
                
                padding = (int(spilt_row_num / 2), int(spilt_col_num / 2))
                #print (padding)

                out = relay.op.nn.conv2d(
                    op_args[0],
                    transpose_weight,
                    kernel_size=(spilt_row_num, spilt_col_num),
                    padding=padding,
                )
                
                new_shape = [original_filter_num, stride_h, stride_w, in_shape[2], in_shape[3]]
                out_shape = [1, original_filter_num, in_shape[2] * stride_h, in_shape[3]* stride_w]

                data = relay.op.transform.reshape(out, new_shape)
                # The data will be transposed to
                # [b, oc, h, upscale_factor, w, upscale_factor]
                # for further reshape
                axes = [0, 3, 1, 4, 2]
                data = relay.op.transform.transpose(data, axes)
                data = relay.op.transform.reshape(data, out_shape)
                return data
                

            return relay.expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)
            
    mod["main"] = Deconv2dToConv2dClass().visit(mod["main"])

    return mod

def LayerGroupTiling(mod, sram_size, element_size):
    class LookUpClass(relay.ExprVisitor):
        """Traverses the graph to determine consumers. The result is maintained in `consumers_list`.

        Attributes
        ----------
        consumers_list : Dict[tvm.relay.expr.Call, List[int]]
            Mapping from NPU operation to list of boolean values that represent
            whether or not each consumer is an NPU operation.
        
        """
        def __init__(self):
            self.consumers_list = defaultdict(list)
            super().__init__()

        def visit_call(self, call: relay.Call):
            args = []
            # Expand tuples
            for arg in call.args:
                if isinstance(arg, relay.Tuple):
                    args.extend(arg.fields)
                else:
                    args.append(arg)

            for arg in args:
                if isinstance(arg, relay.Call) :
                    self.consumers_list[arg].append(1)
                    
            super().visit_call(call)
            
    class LayerGroupTilingClass(relay.ExprMutator):
        """Convert qnn model to relay"""

        def __init__(self, consumers_list, sram_size, element_size):
            self._consumers_list = consumers_list
            self._sram_size = sram_size
            self._element_size = element_size
            self.target = [
                "nn.bias_add",
                "nn.conv2d",
                "nn.relu",
                # "reshape",
            ]
            self.complex_ops = [
                "nn.conv2d",
            ]
            
            super(LayerGroupTilingClass, self).__init__()
        
        def visit_call(self, call):
            # print("*******************************************************************************")
            call_list = []
            temp_list = []
            can_fuse = True
            first_op = False
            fuse_num = 0
            temp_num = 0
            cmplx_num = 0
            tile_num = 0
            last_tile_dims = [2]  # out_shape tile dim(s) idx
            tile_dims = last_tile_dims
            tile_pos = []       # [[[0, N], [0, C], [h-start, h-end], [0, W]], tile1, tile2, ...]
            current_call = call
            current_arg = call.args
            pre_call = current_arg[0]

            if not current_call.op.name in self.target:
                op_args = [self.visit(arg) for arg in current_call.args]
                return relay.expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)

            first_op = isinstance(pre_call, relay.Var)

            temp_num = temp_num + 1
            temp_list.append(current_call)
            
            out_shape = relay.frontend.common.infer_shape(current_call)
            tile_dims_iter_list = []
            for tile_dim in tile_dims:
                tile_dims_iter_list.append(range(out_shape[tile_dim]))
            # Cartesian product for multiple tiling dims
            for cartes_prod in itertools.product(*tile_dims_iter_list):
                tile_idx_range = []
                cartes_idx = 0
                for i in range(len(out_shape)):
                    if i in tile_dims or i - len(out_shape) in tile_dims:
                        tile_idx_range.append([cartes_prod[cartes_idx], cartes_prod[cartes_idx]+1])
                        cartes_idx += 1
                    else:
                        tile_idx_range.append([0, out_shape[i]])
                tile_pos.append(tile_idx_range)
            can_fuse, tile_dims, tile_pos = self.TryToTilingGroup(current_call, tile_dims, tile_pos)
            
            if can_fuse and current_call.op.name in self.complex_ops:
                cmplx_num = cmplx_num + 1
                call_list.extend(temp_list)
                fuse_num = fuse_num + temp_num
                temp_list = []
                temp_num = 0
            
            if can_fuse and not first_op:
                if not pre_call.op.name in self.target:
                    call_list.extend(temp_list)
                    fuse_num = fuse_num + temp_num
                    temp_list = []
                    temp_num = 0
                    
            #TODO 从下向上遍历计算图，在支持h维度切分的情况下，第一个算子与之后每一个算子必须在target列表中（为了避开多输入算子，后续可以使用输入类型检查进一步细化），同时后续算子的输出只能有一个consumer（第一个算子的输出可以有多个consumer）
            while (not first_op and can_fuse and cmplx_num < 3 and pre_call.op.name in self.target 
                    and current_call.op.name in self.target and len(self._consumers_list[pre_call]) == 1) :

                current_call = pre_call
                temp_num = temp_num + 1
                temp_list.append(current_call)
                current_arg = current_call.args
                pre_call = current_arg[0]

                can_fuse, tile_dims, tile_pos = self.TryToTilingGroup(current_call, tile_dims, tile_pos)  # backward

                if current_call.op.name in self.complex_ops and can_fuse:
                    cmplx_num = cmplx_num + 1
                    call_list.extend(temp_list)
                    fuse_num = fuse_num + temp_num
                    temp_num = 0
                    temp_list = []
                    if tile_dims == [1]:
                        break
                    
                first_op = isinstance(pre_call, relay.Var)
                if first_op:
                    break
                
                if not pre_call.op.name in self.target and can_fuse:
                    call_list.extend(temp_list)
                    fuse_num = fuse_num + temp_num
                    temp_list = []
                    temp_num = 0

            if not can_fuse:
                if cmplx_num == 0:
                    tile_dims = [1]
                    call_list.extend(temp_list)
                    fuse_num = fuse_num + temp_num
                    temp_num = 0
                    temp_list = []
                    if call_list[fuse_num - 1].op.name in self.complex_ops:
                        cmplx_num = cmplx_num + 1    

            if first_op:
                call_list.extend(temp_list)
                fuse_num = fuse_num + temp_num
                temp_num = 0
                temp_list = []

            current_call = call_list[fuse_num - 1]
            op_args = [self.visit(arg) for arg in current_call.args]
            can_fuse = False

            # TODO (bug): oc维度缺少能否完成最小tiling的检查，可能导致死循环
            # determine tile_num
            while not can_fuse :
                tile_dims = last_tile_dims
                tile_num = tile_num + 1
                tile_pos = []
                pos_sum = 0
                for i in range(tile_num):
                    tile_len = (out_shape[tile_dims[0]] + i) // tile_num
                    tile_idx_range = []
                    for j in range(len(out_shape)):
                        if j in tile_dims or j - len(out_shape) in tile_dims:
                            tile_idx_range.append([pos_sum, pos_sum + tile_len])
                        else:
                            tile_idx_range.append([0, out_shape[j]])
                    tile_pos.append(tile_idx_range)
                    pos_sum = pos_sum + tile_len
                
                for op in call_list:
                    can_fuse, tile_dims, tile_pos = self.TryToTilingGroup(op, tile_dims, tile_pos)  # backward

                    if not can_fuse:
                        break
                if tile_num > out_shape[tile_dims[0]]:
                    raise Exception("cannot be tiled along oc")
            
            # 对各层进行tiling
            group_out = []
            first_input_tile_list = []
            first_input_tile_list = self.TileInput(call_list[fuse_num - 1], op_args[0], tile_num, tile_dims, tile_pos)
            group_out.append(first_input_tile_list)
            
            for i in range(fuse_num):
                input_tile_list = group_out[i]
                current_call = call_list[fuse_num - 1 - i]
                output_tile_list, tile_dims, tile_pos = self.GetOutputTile(current_call, tile_dims, input_tile_list, tile_pos)  # forward
                group_out.append(output_tile_list)

            if tile_num == 1:
                new_call =  group_out[fuse_num][0]
            else:
                new_call =  relay.op.concatenate(group_out[fuse_num], (len(out_shape)+tile_dims[0])%len(out_shape))
            
            return new_call
        
        # 从输出tile形状推断输入tile形状，目前只支持单维度拆分
        def TryToTilingGroup(self, call, output_tile_dims, output_tile_pos):
            can_fuse = False
            input_tile_pos = []
            input_tile_dims = []
            out_shape = relay.frontend.common.infer_shape(call)
            max_tile_size = 0
            
            if call.op.name == "nn.relu":
                input_tile_pos = output_tile_pos
                input_tile_dims = output_tile_dims
                for pos in output_tile_pos:
                    tile_elem_cnt = 1
                    for i in range(len(out_shape)):
                        tile_elem_cnt *= pos[i][1] - pos[i][0]
                    required_size = tile_elem_cnt * 2 * self._element_size
                    max_tile_size = required_size if required_size > max_tile_size else max_tile_size

            if call.op.name == "reshape":
                in_shape = relay.frontend.common.infer_shape(call.args[0])
                in_shape_prod = functools.reduce(lambda x, y: x * y, in_shape)
                lhs_prod = functools.reduce(lambda x, y: x * y, out_shape[:output_tile_dims[0]])

                # tiling dim cannot be reshaped
                reshape_can_tile = False
                tile_elem_cnt = 1
                for idx, dim in enumerate(in_shape):
                    if dim == out_shape[output_tile_dims[0]] and tile_elem_cnt == lhs_prod:
                        reshape_can_tile = True
                        ih_idx = idx
                        input_tile_dims = [idx]
                        break
                    else:
                        tile_elem_cnt *= dim
                if not reshape_can_tile:
                    raise Exception(f"cannot tile on {output_tile_dims}")
                    # return False, output_tile_dims, input_tile_pos

                for pos in output_tile_pos:
                    tile_len = pos[output_tile_dims[0]][1] - pos[output_tile_dims[0]][0]
                    required_size = in_shape_prod / out_shape[output_tile_dims[0]] * tile_len * 2 * element_size
                    max_tile_size = required_size if required_size > max_tile_size else max_tile_size
                    tile_idx_range = []
                    for idx, in_dim in enumerate(in_shape):
                        if idx == ih_idx:
                            tile_idx_range.append(pos[output_tile_dims[0]])
                        else:
                            tile_idx_range.append([0, in_dim])
                    input_tile_pos.append(tile_idx_range)

            if call.op.name == "nn.conv2d":
                attrs = call.attrs
                padding = attrs["padding"]
                strides = attrs["strides"]
                dilation = attrs["dilation"]
                groups = attrs["groups"]
                weight = call.args[1].data.asnumpy()
                w_shape = weight.shape
                w_h = w_shape[2]
                in_shape = relay.frontend.common.infer_shape(call.args[0])
                input_tile_dims = output_tile_dims
                input_tile_pos = []
                
                assert len(output_tile_dims) == 1, "only single dimension tiling supported"
                assert(dilation[1] == 1 and dilation[0] == 1)
                
                is_depthwise = False
                is_group = False
                if groups > 1:
                    if groups == w_shape[0] == in_shape[1]:
                        is_depthwise = True
                    else :
                        is_group = True
                        assert(not is_group)
                
                for pos in output_tile_pos:
                    # if tile_dim == "h":
                    if output_tile_dims == [2]:
                        # if is_depthwise:
                        #     return False, input_tile_pos_list
                        start = pos[2][0] * strides[1] - padding[0] if (pos[2][0] * strides[1] - padding[0]) > 0 else 0
                        end = (pos[2][1] - 1) * strides[1] + w_h - padding[0] if ((pos[2][1] - 1) * strides[1] + w_h - padding[0]) < in_shape[2] else in_shape[2]
                        # tile_idx_range = [start, end]
                        tile_idx_range = []
                        for idx, dim in enumerate(in_shape):
                            if idx == output_tile_dims[0]:
                                tile_idx_range.append([start, end])
                            else:
                                tile_idx_range.append([0, dim])
                        input_tile_pos.append(tile_idx_range)
                        
                        out_size = out_shape[0] * out_shape[1] * out_shape[3] * (pos[2][1] - pos[2][0]) * self._element_size
                        in_size = in_shape[0] * in_shape[1] * in_shape[3] * (end - start) * self._element_size
                        w_size = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3] * self._element_size
                        
                        required_size = out_size + in_size + w_size
                        max_tile_size = required_size if required_size > max_tile_size else max_tile_size
                        # print("********************")
                        # print(output_tile_pos)
                        # print(end)
                        # print(start)
                        # print(required_size)
                        # print(out_size)
                        # print(in_size)
                        # print(w_size)
                        # print(w_shape)
                    # if tile_dim == "oc":
                    if output_tile_dims == [1]:
                        if is_depthwise:
                            out_size = out_shape[0] * (pos[1][1] - pos[1][0]) * out_shape[2] * out_shape[3] * self._element_size
                            in_size = in_shape[0] * (pos[1][1] - pos[1][0]) * in_shape[2] * in_shape[3] * self._element_size
                        else:
                            out_size = out_shape[0] * (pos[1][1] - pos[1][0]) * out_shape[2] * out_shape[3] * self._element_size
                            in_size = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] * self._element_size
                        w_size = (pos[1][1] - pos[1][0]) * w_shape[1] * w_shape[2] * w_shape[3] * self._element_size
                        required_size = out_size + in_size + w_size
                        max_tile_size = required_size if required_size > max_tile_size else max_tile_size
                        tile_idx_range = []
                        for idx, dim in enumerate(in_shape):
                            if idx == output_tile_dims[0]:
                                tile_idx_range.append(pos[idx])
                            else:
                                tile_idx_range.append([0, dim])
                        input_tile_pos.append(tile_idx_range)
                        
                        # print("********************")
                        # print(output_tile_pos)
                        # print(required_size)
                        # print(out_size)
                        # print(in_size)
                        # print(w_size)
                        # print(w_shape)
            
            if call.op.name == "nn.bias_add":
                bias = call.args[1].data.asnumpy()
                b_shape = bias.shape
                assert(len(b_shape) == 1)
                input_tile_pos = output_tile_pos
                input_tile_dims = output_tile_dims
                for pos in output_tile_pos:
                    tile_elem_cnt = 1
                    for i in range(len(out_shape)):
                        tile_elem_cnt *= pos[i][1] - pos[i][0]
                    if 1 in output_tile_dims:  # oc
                        b_size = (pos[1][1] - pos[1][0]) * self._element_size
                    else:
                        b_size = b_shape[0] * self._element_size
                    required_size = tile_elem_cnt * 2 * self._element_size + b_size
                    max_tile_size = required_size if required_size > max_tile_size else max_tile_size
                
            if max_tile_size <= self._sram_size:
                can_fuse = True
            
            return can_fuse, input_tile_dims, input_tile_pos
        
        def TileInput(self, first_call, input, tile_num, tile_dims, tile_pos):
            assert len(tile_dims) == 1, "only single dimension tiling supported"
            input_tile_list = []
            if tile_num == 1:
                input_tile_list.append(input)
                return input_tile_list
            
            if first_call.op.name == "nn.relu":
                for i in range(tile_num):
                    current_tile_size = tile_pos[i][tile_dims[0]]
                    slice = relay.op.strided_slice(data = input, begin = [current_tile_size[0]], end = [current_tile_size[1]], axes = [tile_dims[0]])
                    input_tile_list.append(slice)
            
            if first_call.op.name == "reshape":
                for i in range(tile_num):
                    current_tile_size = tile_pos[i][tile_dims[0]]
                    axe = (tile_dims[0] + len(tile_pos[i])) % len(tile_pos[i])
                    slice = relay.op.strided_slice(data = input, begin = [current_tile_size[0]], end = [current_tile_size[1]], axes = [axe])
                    input_tile_list.append(slice)

            if first_call.op.name == "nn.conv2d":
                attrs = tiling_get_attrs(first_call.attrs)
                groups = attrs["groups"]
                weight = first_call.args[1].data.asnumpy()
                w_shape = weight.shape
                in_shape = relay.frontend.common.infer_shape(first_call.args[0])
                
                is_depthwise = False
                is_group = False
                if groups > 1:
                    if groups == w_shape[0] == in_shape[1]:
                        is_depthwise = True
                    else :
                        is_group = True
                        assert(not is_group)
                
                # if tile_dim == "oc" :
                if tile_dims == [1]:
                    if is_depthwise:
                        for i in range(tile_num):
                            current_tile_size = tile_pos[i][tile_dims[0]]
                            slice = relay.op.strided_slice(data = input, begin = [current_tile_size[0]], end = [current_tile_size[1]], axes = [tile_dims[0]])
                            input_tile_list.append(slice)
                    else:
                        for i in range(tile_num):
                            input_tile_list.append(input)
                # elif tile_dim == "h":
                elif tile_dims == [2]:
                    for i in range(tile_num):
                        current_tile_size = tile_pos[i][tile_dims[0]]
                        slice = relay.op.strided_slice(data = input, begin = [current_tile_size[0]], end = [current_tile_size[1]], axes = [tile_dims[0]])
                        input_tile_list.append(slice)    
                
            if first_call.op.name == "nn.bias_add": 
                for i in range(tile_num):
                    current_tile_size = tile_pos[i][tile_dims[0]]
                    slice = relay.op.strided_slice(data = input, begin = [current_tile_size[0]], end = [current_tile_size[1]], axes = [tile_dims[0]])
                    input_tile_list.append(slice)
                    
            return input_tile_list
        
        # 由前一个算子的call_list，向下传递
        def GetOutputTile(self, call, input_tile_dims, input_tile_list, input_tile_pos):
            output_tile_list = []
            output_tile_dims = []
            output_tile_pos = []
            
            if call.op.name == "nn.relu":
                output_tile_dims = input_tile_dims
                output_tile_pos = input_tile_pos
                for tile_input in input_tile_list:
                    tile_output = relay.expr.Call(call.op, [tile_input], call.attrs, call.type_args, call.span)
                    output_tile_list.append(tile_output)
            
            if call.op.name == "reshape":
                attrs = tiling_get_attrs(call.attrs)
                out_shape = attrs["newshape"]
                in_shape = relay.frontend.common.infer_shape(call.args[0])
                lhs_prod = functools.reduce(lambda x, y: x * y, in_shape[:input_tile_dims[0]])
                tile_elem_cnt = 1
                for idx, dim in enumerate(out_shape):
                    if dim == in_shape[input_tile_dims[0]] and tile_elem_cnt == lhs_prod:
                        output_tile_dims = [idx]
                        break
                    else:
                        tile_elem_cnt *= dim
                new_shape = out_shape
                for idx, tile_input in enumerate(input_tile_list):
                    new_shape[output_tile_dims[0]] = input_tile_pos[idx][input_tile_dims[0]][1] - input_tile_pos[idx][input_tile_dims[0]][0]
                    tile_output = relay.reshape(data=tile_input, newshape=new_shape)
                    output_tile_list.append(tile_output)
                    
                    tile_idx_range = []
                    for out_idx, out_dim in enumerate(out_shape):
                        if out_idx == output_tile_dims[0]:
                            tile_idx_range.append(input_tile_pos[idx][input_tile_dims[0]])
                        else:
                            tile_idx_range.append([0, out_dim])
                    output_tile_pos.append(tile_idx_range)
            
            if call.op.name == "nn.conv2d":
                output_tile_dims = input_tile_dims
                output_tile_pos = []
                attrs = tiling_get_attrs(call.attrs)
                # Array<IndexExpr> strides;
                # Array<IndexExpr> padding;
                # Array<IndexExpr> dilation;
                # int groups;
                # IndexExpr channels;
                # Array<IndexExpr> kernel_size;
                # tvm::String data_layout;
                # tvm::String kernel_layout;
                # tvm::String out_layout;
                # tvm::String auto_scheduler_rewritten_layout;   // The layout after auto-scheduler's layout rewrite
                # Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights
                # DataType out_dtype;
                padding = attrs["padding"]
                strides = attrs["strides"]
                dilation = attrs["dilation"]
                groups = attrs["groups"]
                channels = attrs["channels"]
                kernel_size = attrs["kernel_size"]
                data_layout = attrs["data_layout"]
                kernel_layout = attrs["kernel_layout"]
                out_layout = attrs["out_layout"]
                out_dtype = attrs["out_dtype"]
                assert(strides[1] >= padding[0] and strides[1] >= padding[2])
                weight = call.args[1].data.asnumpy()
                w_shape = weight.shape
                
                in_shape = relay.frontend.common.infer_shape(call.args[0])
                out_shape = relay.frontend.common.infer_shape(call)
                
                is_depthwise = False
                is_group = False
                if groups > 1:
                    if groups == w_shape[0] == in_shape[1]:
                        is_depthwise = True
                    else :
                        is_group = True
                        assert(not is_group)
                
                tile_index = 0
                pos_sum = 0
                for tile_input in input_tile_list:
                    # if tile_dim == "h":
                    if input_tile_dims == [2]:
                        s_w_expr = relay.const(weight)
                        if tile_index == 0 and not tile_index == len(input_tile_list) - 1:
                            new_padding = [padding[0], padding[1], 0, padding[3]]
                        elif tile_index == len(input_tile_list) - 1 and not tile_index == 0 :
                            new_padding = [0, padding[1], padding[2], padding[3]]
                        elif tile_index == len(input_tile_list) - 1 and tile_index == 0 :
                            new_padding = [padding[0], padding[1], padding[2], padding[3]]
                        else:
                            new_padding = [0, padding[1], 0, padding[3]]
                        
                        tile_output = relay.op.nn.conv2d(
                            tile_input, 
                            s_w_expr, 
                            padding = new_padding,
                            strides = strides,
                            dilation = dilation,
                            groups = groups,
                            channels = channels,
                            kernel_size = kernel_size,
                            data_layout = data_layout,
                            kernel_layout = kernel_layout,
                            out_layout = out_layout,
                            out_dtype = out_dtype
                        )
                        output_tile_list.append(tile_output)

                        oh = ((input_tile_pos[tile_index][2][1] - input_tile_pos[tile_index][2][0]) + (new_padding[0] + new_padding[2]) - kernel_size[1]) // strides[1] + 1
                        tile_idx_range = []
                        for idx, dim in enumerate(out_shape):
                            if idx == input_tile_dims[0]:
                                tile_idx_range.append([pos_sum, oh + pos_sum])
                            else:
                                tile_idx_range.append([0, dim])
                        output_tile_pos.append(tile_idx_range)
                        pos_sum = oh + pos_sum

                    # if tile_dim == "oc":
                    if input_tile_dims == [1]:
                        # inshape = relay.frontend.common.infer_shape(tile_input)
                        # print("inshape")
                        # print(inshape)
                        
                        channel_pos = input_tile_pos[tile_index][1]
                        s_weight = weight[channel_pos[0] : channel_pos[1], :, :, :]
                        s_weight_shape = s_weight.shape
                        s_w_expr = relay.const(s_weight)
                        
                        channels = channel_pos[1] - channel_pos[0]
                        if is_depthwise:
                            groups = channels
                            
                        tile_output = relay.op.nn.conv2d(
                            tile_input, 
                            s_w_expr, 
                            padding = padding,
                            strides = strides,
                            dilation = dilation,
                            groups = groups,
                            channels = channels,
                            kernel_size = kernel_size,
                            data_layout = data_layout,
                            kernel_layout = kernel_layout,
                            out_layout = out_layout,
                            out_dtype = out_dtype
                        )
                        output_tile_list.append(tile_output)

                        tile_idx_range = []
                        for idx, dim in enumerate(out_shape):
                            if idx == input_tile_dims[0]:
                                tile_idx_range.append(channel_pos)
                            else:
                                tile_idx_range.append([0, dim])
                        output_tile_pos.append(tile_idx_range)
                        
                    tile_index = tile_index + 1
                    
            if call.op.name == "nn.bias_add":
                output_tile_dims = input_tile_dims
                output_tile_pos = input_tile_pos
                attrs = call.attrs
                bias = call.args[1].data.asnumpy()
                tile_index = 0
                for tile_input in input_tile_list:
                    inshape = relay.frontend.common.infer_shape(tile_input)
                    # if tile_dim == "h":
                    if input_tile_dims == [2]:
                        # b_shape = bias.shape
                        # s_bias = bias.reshape(1, b_shape[0], 1, 1)
                        # s_bias = np.broadcast_to(s_bias, (1, b_shape[0], inshape[2], inshape[3]))
                        # s_b_expr = relay.const(s_bias)
                        # tile_output = relay.op.add(tile_input, s_b_expr)
                        
                        tile_output = relay.op.nn.bias_add(tile_input, call.args[1])
                        # tile_output = relay.expr.Call(call.op, [tile_input, call.args[1]], call.attrs, call.type_args, call.span)
                        output_tile_list.append(tile_output)
                    # if tile_dim == "oc":
                    if input_tile_dims == [1]:
                        # channel_pos = tile_pos[tile_index]
                        # s_bias = bias[channel_pos[0] : channel_pos[1]]
                        # b_shape = s_bias.shape
                        # s_bias = s_bias.reshape(1, b_shape[0], 1, 1)
                        # s_bias = np.broadcast_to(s_bias, (1, b_shape[0], inshape[2], inshape[3]))
                        # s_b_expr = relay.const(s_bias)
                        # tile_output = relay.op.add(tile_input, s_b_expr)
                        
                        
                        channel_pos = input_tile_pos[tile_index][1]
                        s_bias = bias[channel_pos[0] : channel_pos[1]]
                        s_b_expr = relay.const(s_bias)
                        tile_output = relay.op.nn.bias_add(tile_input, s_b_expr)
                        
                        output_tile_list.append(tile_output)
                        
                    tile_index = tile_index + 1
                
            return output_tile_list, output_tile_dims, output_tile_pos
    
    lookup = LookUpClass()
    lookup.visit(mod["main"])
    
    mod["main"] = LayerGroupTilingClass(lookup.consumers_list, sram_size, element_size).visit(mod["main"])

    return mod



def tiling_get_attrs(attrs):
    ret = {}
    for i in dir(attrs):
    
        ret[i] = getattr(attrs, i)
        if isinstance(ret[i], tvm.ir.container.Array):
            ret[i] = tiling_get_array_value(ret[i])

    return ret

def tiling_get_array_value(data):
    out = []
    for x in data:
        if isinstance(x, tvm.ir.container.Array):
            out.append(tiling_get_array_value(x))
        else:
            out.append(x.value)
    return out