#=
expression_utils:
- Julia version: 1.9.3
- Author: dang
- Date: 2024-01-11
=#
using FLux
struct OperatorNode
    operator::Int
    operator_str::String
    arity::Int
    parent::Union{Nothing, OperatorNode}
    left_child::Union{Nothing, OperatorNode}
    right_child::Union{Nothing, OperatorNode}
end

function OperatorNode(operator, operators, arity, parent=nothing)
    operator_str = operators.operator_list[operator]
    new(operator, operator_str, arity, parent)
end

function add_child!(node::OperatorNode, child::OperatorNode)
    if isnothing(node.left_child)
        node.left_child = child
    elseif isnothing(node.right_child)
        node.right_child = child
    else
        error("Both children have been created.")
    end
end

function set_parent(node::OperatorNode, parent_node)
    node.parent = parent_node
end

function remaining_children(node::OperatorNode)
    if node.arity == 0
        return false
    elseif node.arity == 1 && !isnothing(node.left_child)
        return false
    elseif node.arity == 2 && !isnothing(node.left_child) && !isnothing(node.right_child)
        return false
    end
    return true
end

function recursive_print(node::OperatorNode)
    if node.arity == 2
        left_print = recursive_print(node.left_child)
        right_print = recursive_print(node.right_child)
        return "($left_print) $(node.operator_str) ($right_print)"
    elseif node.arity == 1
        child_str = recursive_print(node.left_child)
        return "$(node.operator_str)($child_str)"
    else
        return node.operator_str
    end
end

function torch_print(node::OperatorNode, operators)
    if node.arity == 2
        left_str = torch_print(node.left_child, operators)
        right_str = torch_print(node.right_child, operators)
        return "$(operators.func_dict[node.operator_str])(($left_str), ($right_str))"
    elseif node.arity == 1
        child_str = torch_print(node.left_child, operators)
        return "$(operators.func_dict[node.operator_str])(($child_str))"
    else
        return node.operator_str
    end
end

function torch_print_helper(node::OperatorNode, operators)
    if node.arity == 2
        left_print = torch_print_helper(node.left_child, operators)
        right_print = torch_print_helper(node.right_child, operators)
        return "$(operators.func_dict[node.operator_str])(($left_print), ($right_print))"
    elseif node.arity == 1
        child_str = torch_print_helper(node.left_child, operators)
        return "$(operators.func_dict[node.operator_str])(($child_str))"
    else
        if node.operator_str == 'c'
            return "@"
        elseif startswith(node.operator_str, "var_")
            var_index = operators.var_dict[node.operator_str]
            return "x[:, $var_index]"
        else
            return "($(node.operator_str))"
        end
    end
end

function construct_tree(operators, sequence, length)
    root = OperatorNode(sequence[1], operators, operators.arity_i(sequence[1]))
    past_node = root
    for operator in sequence[2:length]
        curr_node = OperatorNode(operator, operators, operators.arity_i(operator), past_node)
        add_child!(past_node, curr_node)
        past_node = curr_node
        while !remaining_children(past_node)
            past_node = past_node.parent
            if isnothing(past_node)
                break
            end
        end
    end
    return root
end

struct Expression
    sequence::Vector{Int}
    root::OperatorNode
    num_constants::Int
    c::Vector{Float64}
    expression::String

    function Expression(operators, sequence::Vector{Int}, length::Int)
        new_sequence = sequence[1:length]
        root = construct_tree(operators, new_sequence, length)
        num_constants = count(==(operators.operator_list_index('c')), new_sequence)
        c = num_constants > 0 ? rand(num_constants) : Float64[]
        expression = torch_print(root, operators)
        new(new_sequence, root, num_constants, c, expression)
    end
end

function forward(expr::Expression, x)
    # 将表达式字符串转换为可执行的 Julia 代码
    expr_code = Meta.parse(replace(expr.expression, "self.c" => "expr.c"))
    eval(expr_code)
end

function get_constants(expr::Expression)
    expr.c
end

function Base.show(io::IO, expr::Expression)
    c_expression = recursive_print(expr.root)
    constant_dict = ["c$i" => round(expr.c[i], digits=4) for i in 1:expr.num_constants]
    for (holder, learned_val) in constant_dict
        c_expression = replace(c_expression, holder => learned_val)
    end
    print(io, c_expression)
end

function optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, optimizer_type)
    expressions_with_constants = filter(e -> e.num_constants > 0, expressions)

    if isempty(expressions_with_constants)
        return
    end

    exp_ens = ExpressionEnsemble(expressions_with_constants)
    y_constants_ens = repeat(y_constants, 1, length(expressions_with_constants))

    optimizer = if optimizer_type == "lbfgs"
        Flux.Optimise.LBFGS()
    elseif optimizer_type == "adam"
        ADAM(inner_lr)
    else
        RMSProp(inner_lr)
    end

    loss_fn(x, y) = Flux.mse(exp_ens(x), y)

    for _ in 1:inner_num_epochs
        if optimizer_type == "lbfgs"
            Flux.train!(loss_fn, params(exp_ens), [(X_constants, y_constants_ens)], optimizer)
        else
            gs = gradient(() -> loss_fn(X_constants, y_constants_ens), params(exp_ens))
            Flux.Optimise.update!(optimizer, params(exp_ens), gs)
        end
    end
end

struct ExpressionEnsemble
    models::Vector
end

Flux.@functor ExpressionEnsemble

function (ee::ExpressionEnsemble)(x)
    [model(x) for model in ee.models]
end

struct ExpressionEnsemble
    models::Vector
end

Flux.@functor ExpressionEnsemble

function (ee::ExpressionEnsemble)(x)
    results = [model(x) for model in ee.models]
    return hcat(results...)
end

