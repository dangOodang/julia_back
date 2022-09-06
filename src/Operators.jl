module OperatorsModule

using SpecialFunctions: SpecialFunctions
import SpecialFunctions: erf, erfc
import Base: @deprecate
#TODO - actually add these operators to the module!

function gamma(x::T)::T where {T<:Real}
    if x <= T(0) && abs(x % 1) < T(1e-6)
        T(1//100000000)
    else
        SpecialFunctions.gamma(x)
    end
end
gamma(x) = SpecialFunctions.gamma(x)

atanh_clip(x) = atanh(mod(x + 1, 2) - 1)

# Implicitly defined:
#binary: mod
#unary: exp, abs, log1p, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, erf, erfc, gamma, relu, round, floor, ceil, round, sign.

# Use some fast operators from https://github.com/JuliaLang/julia/blob/81597635c4ad1e8c2e1c5753fda4ec0e7397543f/base/fastmath.jl
# Define allowed operators. Any julia operator can also be used.
function plus(x::T, y::T)::T where {T<:Real}
    return x + y #Do not change the name of this operator.
end
function sub(x::T, y::T)::T where {T<:Real}
    return x - y #Do not change the name of this operator.
end
function mult(x::T, y::T)::T where {T<:Real}
    return x * y #Do not change the name of this operator.
end
function square(x::T)::T where {T<:Real}
    return x * x
end
function cube(x::T)::T where {T<:Real}
    return x^3
end
function pow_abs(x::T, y::T)::T where {T<:Real}
    return abs(x)^y
end
function div(x::T, y::T)::T where {T<:Real}
    return x / y
end
function log_nan(x::T)::T where {T<:Real}
    x <= T(0) && return T(NaN)
    return log(x)
end
function log2_nan(x::T)::T where {T<:Real}
    x <= T(0) && return T(NaN)
    return log2(x)
end
function log10_nan(x::T)::T where {T<:Real}
    x <= T(0) && return T(NaN)
    return log10(x)
end
function log1p_nan(x::T)::T where {T<:Real}
    x <= T(-1) && return T(NaN)
    return log1p(x)
end
function acosh_abs(x::T)::T where {T<:Real}
    return acosh(abs(x) + convert(T, 1))
end

# Generics:
square(x) = x * x
cube(x) = x * x * x
plus(x, y) = x + y
sub(x, y) = x - y
mult(x, y) = x * y
pow_abs(x, y) = abs(x)^y
div(x, y) = x / y
log_nan(x) = log(x)
log2_nan(x) = log2(x)
log10_nan(x) = log10(x)
log1p_nan(x) = log1p(x)
acosh_abs(x) = acosh(abs(x) + 1)

function sqrt_abs(x::T)::T where {T}
    return sqrt(abs(x))
end
function neg(x::T)::T where {T}
    return -x
end

function greater(x::T, y::T)::T where {T}
    return convert(T, (x > y))
end
function greater(x, y)
    return (x > y)
end
function relu(x::T)::T where {T}
    return convert(T, (x > 0)) * x
end

function logical_or(x::T, y::T)::T where {T}
    return convert(T, (x > convert(T, 0) || y > convert(T, 0)))
end

# (Just use multiplication normally)
function logical_and(x::T, y::T)::T where {T}
    return convert(T, (x > convert(T, 0) && y > convert(T, 0)))
end

# Deprecated operations:
@deprecate pow pow_abs

end
