module OptionsStructModule

using Optim: Optim
using StatsBase: StatsBase
import Random: AbstractRNG
import LossFunctions: SupervisedLoss

mutable struct MutationWeightings
    mutate_constant::Float64
    mutate_operator::Float64
    add_node::Float64
    insert_node::Float64
    delete_node::Float64
    simplify::Float64
    randomize::Float64
    do_nothing::Float64
end

const mutations = fieldnames(MutationWeightings)

"""
    MutationWeightings(;kws...)

This defines how often different mutations occur. These weightings
will be normalized to sum to 1.0 after initialization.
# Arguments
- `mutate_constant::Float64`: How often to mutate a constant.
- `mutate_operator::Float64`: How often to mutate an operator.
- `add_node::Float64`: How often to append a node to the tree.
- `insert_node::Float64`: How often to insert a node into the tree.
- `delete_node::Float64`: How often to delete a node from the tree.
- `simplify::Float64`: How often to simplify the tree.
- `randomize::Float64`: How often to create a random tree.
- `do_nothing::Float64`: How often to do nothing.
"""
function MutationWeightings(;
    mutate_constant=0.048,
    mutate_operator=0.47,
    add_node=0.79,
    insert_node=5.1,
    delete_node=1.7,
    simplify=0.0020,
    randomize=0.00023,
    do_nothing=0.21,
)
    return MutationWeightings(
        mutate_constant,
        mutate_operator,
        add_node,
        insert_node,
        delete_node,
        simplify,
        randomize,
        do_nothing,
    )
end

"""Convert MutationWeightings to a vector."""
@generated function Base.convert(
    ::Type{V}, weightings::MutationWeightings
) where {V<:AbstractVector}
    fields = [:(weightings.$(mut)) for mut in mutations]
    return :(V([$(fields...)]))
end

"""Sample a mutation, given the weightings."""
function Base.rand(rng::AbstractRNG, weightings::MutationWeightings)::Symbol
    weights = convert(Vector, weightings)
    return mutations[StatsBase.sample(rng, 1:length(mutations), StatsBase.Weights(weights))]
end

"""This struct defines how complexity is calculated."""
struct ComplexityMapping{T<:Real}
    use::Bool  # Whether we use custom complexity, or just use 1 for everythign.
    binop_complexities::Vector{T}  # Complexity of each binary operator.
    unaop_complexities::Vector{T}  # Complexity of each unary operator.
    variable_complexity::T  # Complexity of using a variable.
    constant_complexity::T  # Complexity of using a constant.
end

Base.eltype(::ComplexityMapping{T}) where {T} = T

function ComplexityMapping(use::Bool)
    return ComplexityMapping{Int}(use, zeros(Int, 0), zeros(Int, 0), 1, 1)
end

"""Promote type when defining complexity mapping."""
function ComplexityMapping(;
    binop_complexities::Vector{T1},
    unaop_complexities::Vector{T2},
    variable_complexity::T3,
    constant_complexity::T4,
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    promoted_T = promote_type(T1, T2, T3, T4)
    return ComplexityMapping{promoted_T}(
        true,
        binop_complexities,
        unaop_complexities,
        variable_complexity,
        constant_complexity,
    )
end

struct Options{A,B,dA,dB,C<:Union{SupervisedLoss,Function},D}
    binops::A
    unaops::B
    diff_binops::dA
    diff_unaops::dB
    bin_constraints::Vector{Tuple{Int,Int}}
    una_constraints::Vector{Int}
    complexity_mapping::ComplexityMapping{D}
    ns::Int
    parsimony::Float32
    alpha::Float32
    maxsize::Int
    maxdepth::Int
    fast_cycle::Bool
    migration::Bool
    hofMigration::Bool
    fractionReplacedHof::Float32
    shouldOptimizeConstants::Bool
    hofFile::String
    npopulations::Int
    perturbationFactor::Float32
    annealing::Bool
    batching::Bool
    batchSize::Int
    mutation_weights::MutationWeightings
    crossoverProbability::Float32
    warmupMaxsizeBy::Float32
    useFrequency::Bool
    useFrequencyInTournament::Bool
    npop::Int
    ncyclesperiteration::Int
    fractionReplaced::Float32
    topn::Int
    verbosity::Int
    probNegate::Float32
    nuna::Int
    nbin::Int
    seed::Union{Int,Nothing}
    loss::C
    progress::Bool
    terminal_width::Union{Int,Nothing}
    optimizer_algorithm::String
    optimize_probability::Float32
    optimizer_nrestarts::Int
    optimizer_options::Optim.Options
    recorder::Bool
    recorder_file::String
    probPickFirst::Float32
    earlyStopCondition::Union{Function,Nothing}
    stateReturn::Bool
    timeout_in_seconds::Union{Float64,Nothing}
    max_evals::Union{Int,Nothing}
    skip_mutation_failures::Bool
    enable_autodiff::Bool
    nested_constraints::Union{Vector{Tuple{Int,Int,Vector{Tuple{Int,Int,Int}}}},Nothing}
    deterministic::Bool
end

function Base.print(io::IO, options::Options)
    return print(
        io,
        """Options(
# Operators:
    binops=$(options.binops), unaops=$(options.unaops),
# Loss:
    loss=$(options.loss),
# Complexity Management:
    maxsize=$(options.maxsize), maxdepth=$(options.maxdepth), bin_constraints=$(options.bin_constraints), una_constraints=$(options.una_constraints), useFrequency=$(options.useFrequency), useFrequencyInTournament=$(options.useFrequencyInTournament), parsimony=$(options.parsimony), warmupMaxsizeBy=$(options.warmupMaxsizeBy), 
# Search Size:
    npopulations=$(options.npopulations), ncyclesperiteration=$(options.ncyclesperiteration), npop=$(options.npop), 
# Migration:
    migration=$(options.migration), hofMigration=$(options.hofMigration), fractionReplaced=$(options.fractionReplaced), fractionReplacedHof=$(options.fractionReplacedHof),
# Tournaments:
    probPickFirst=$(options.probPickFirst), ns=$(options.ns), topn=$(options.topn), 
# Constant tuning:
    perturbationFactor=$(options.perturbationFactor), probNegate=$(options.probNegate), shouldOptimizeConstants=$(options.shouldOptimizeConstants), optimizer_algorithm=$(options.optimizer_algorithm), optimize_probability=$(options.optimize_probability), optimizer_nrestarts=$(options.optimizer_nrestarts), optimizer_iterations=$(options.optimizer_options.iterations),
# Mutations:
    mutationWeights=$(options.mutationWeights), crossoverProbability=$(options.crossoverProbability), skip_mutation_failures=$(options.skip_mutation_failures)
# Annealing:
    annealing=$(options.annealing), alpha=$(options.alpha), 
# Speed Tweaks:
    batching=$(options.batching), batchSize=$(options.batchSize), fast_cycle=$(options.fast_cycle), 
# Logistics:
    hofFile=$(options.hofFile), verbosity=$(options.verbosity), seed=$(options.seed), progress=$(options.progress),
# Early Exit:
    earlyStopCondition=$(options.earlyStopCondition), timeout_in_seconds=$(options.timeout_in_seconds),
)""",
    )
end
Base.show(io::IO, options::Options) = Base.print(io, options)

end
