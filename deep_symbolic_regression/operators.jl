#=
operators:
- Julia version: 1.9.3
- Author: dang
- Date: 2024-01-11
=#
struct Operators
    operator_list::Vector{String}
    constant_operators::Vector{String}
    nonvar_operators::Vector{String}
    var_operators::Vector{String}
    arity_dict::Dict{String, Int}
    zero_arity_mask::Vector{Int}
    nonzero_arity_mask::Vector{Int}
    variable_mask::Vector{Int}
    nonvariable_mask::Vector{Int}
    arity_two::Vector{Int}
    arity_one::Vector{Int}
    arity_zero::Vector{Int}
    variable_tensor::Vector{Int}
    func_dict::Dict{String, Function}
    var_dict::Dict{String, Int}

    Operators(operator_list::Vector{String}) = new(
        operator_list,
        [x for x in operator_list if isnumeric(replace(x, '.', '', count=1))],
        [x for x in operator_list if startswith(x, "var_")],
        [x for x in operator_list if !isnumeric(replace(x, '.', '', count=1)) && !startswith(x, "var_")],
        make_arity_dict(),
        make_mask(operator_list, 0),
        make_mask(operator_list, 1),
        [x in var_operators for x in operator_list],
        [!(x in var_operators) for x in operator_list],
        findall(x -> arity_dict[x] == 2, operator_list),
        findall(x -> arity_dict[x] == 1, operator_list),
        findall(x -> arity_dict[x] == 0, operator_list),
        findall(x -> x in var_operators, operator_list),
        make_func_dict(),
        make_var_dict(var_operators)
    )
end

function make_arity_dict()
    Dict(
        "*" => 2, "+" => 2, "-" => 2, "/" => 2, "^" => 2,
        "cos" => 1, "sin" => 1, "tan" => 1, "exp" => 1, "ln" => 1,
        "sqrt" => 1, "square" => 1, "c" => 0
    )
end

function make_mask(operator_list, arity)
    [arity_dict[op] == arity ? 1 : 0 for op in operator_list]
end

function make_func_dict()
    Dict(
        "*" => (*)(x, y),
        "+" => (+)(x, y),
        "-" => (-)(x, y),
        "/" => (/)(x, y),
        "^" => (^)(x, y),
        "cos" => cos,
        "sin" => sin,
        "tan" => tan,
        "exp" => exp,
        "ln" => log,
        "sqrt" => sqrt,
        "square" => x -> x^2
    )
end

function make_var_dict(var_operators)
    Dict([op => i for (i, op) in enumerate(var_operators)])
end