#=
rnn:
- Julia version: 1.6
- Author: dang
- Date: 2024-01-10
=#
using Flux
using Flux: LSTM, RNN, GRU

struct DSRRNN{T<:Union{LSTM, RNN, GRU}}
    rnn::T
    projection_layer::Dense
    init_input::Flux.TrackedArray
    init_hidden::Flux.TrackedArray
    init_hidden_lstm::Union{Flux.TrackedArray, Nothing}
    input_size::Int
    hidden_size::Int
    output_size::Int
    min_length::Int
    max_length::Int
    num_layers::Int
    dropout::Float64
end

function DSRRNN(operators, hidden_size; rnn_type=:rnn, num_layers=1, dropout=0.0, min_length=2, max_length=15, use_gpu=false)
    input_size = 2 * length(operators)
    output_size = length(operators)

    init_input = param(randn(1, input_size))
    init_hidden = param(randn(num_layers, hidden_size))
    init_hidden_lstm = rnn_type == :lstm ? param(randn(num_layers, hidden_size)) : nothing

    rnn_layer = rnn_type == :rnn ? RNN(input_size, hidden_size, num_layers; dropout=dropout) :
                rnn_type == :lstm ? LSTM(input_size, hidden_size, num_layers; dropout=dropout) :
                                    GRU(input_size, hidden_size, num_layers; dropout=dropout)

    projection_layer = Dense(hidden_size, output_size)

    # 将模型移动到 GPU，如果需要
    if use_gpu
        init_input = gpu(init_input)
        init_hidden = gpu(init_hidden)
        init_hidden_lstm = init_hidden_lstm !== nothing ? gpu(init_hidden_lstm) : nothing
        rnn_layer = gpu(rnn_layer)
        projection_layer = gpu(projection_layer)
    end

    return DSRRNN(rnn_layer, projection_layer, init_input, init_hidden, init_hidden_lstm, input_size, hidden_size, output_size, min_length, max_length, num_layers, dropout)
end

function (m::DSRRNN)(x)
    hidden = m.init_hidden
    if m.rnn isa LSTM && m.init_hidden_lstm !== nothing
        output, hidden_state = m.rnn(x, hidden, m.init_hidden_lstm)
    else
        output, hidden_state = m.rnn(x, hidden)
    end
    output = m.projection_layer(output)
    return Flux.softmax(output, dims=1), hidden_state
end

function sample_sequence(model, n, operators; min_length=2, max_length=15)
    # 初始化序列和统计量
    sequences = zeros(Int, n, 0)
    entropies = zeros(Float32, n, 0)
    log_probs = zeros(Float32, n, 0)
    sequence_mask = ones(Bool, n, 1)

    # 准备输入和隐藏状态
    input_tensor = repeat(model.init_input, outer=(1, n))
    hidden_tensor = repeat(model.init_hidden, outer=(1, n, 1))
    hidden_lstm = model.rnn isa LSTM ? repeat(model.init_hidden_lstm, outer=(1, n, 1)) : nothing

    counters = ones(Int, n)
    lengths = zeros(Int, n)

    # 生成序列的循环
    while any(sequence_mask[:, end])
        # 根据 RNN 类型选择适当的 forward 函数
        if model.rnn isa LSTM
            output, hidden_state = model.rnn(input_tensor, hidden_tensor, hidden_lstm)
            hidden_tensor, hidden_lstm = hidden_state
        else
            output, hidden_tensor = model.rnn(input_tensor, hidden_tensor)
        end

        # 应用约束和归一化
        # 这里需要自定义 apply_constraints 函数
        output = apply_constraints(output, counters, lengths, sequences, operators)
        output = softmax(output, dims=1)

        # 从分类分布中采样
        dist = Categorical.(eachcol(output))
        token = [rand(dist[i]) for i in 1:n]

        # 更新序列和统计量
        sequences = hcat(sequences, hcat(token...))
        lengths .+= 1

        # 更新计数器和序列掩码
        # 这里需要自定义 update_counters 函数
        counters, sequence_mask = update_counters(token, counters, sequence_mask, operators)

        # 计算下一个输入张量
        # 这里需要自定义 get_next_input 函数
        input_tensor = get_next_input(sequences, lengths, operators)
    end

    # 计算总熵和总概率
    entropies = sum(entropies .* sequence_mask[:, 1:end-1], dims=2)
    log_probs = sum(log_probs .* sequence_mask[:, 1:end-1], dims=2)
    sequence_lengths = sum(sequence_mask, dims=2)

    return sequences, sequence_lengths, entropies, log_probs
end

function forward(model::DSRRNN, input, hidden, hidden_lstm=nothing)
    # 输入维度的调整
    input = reshape(input, size(input)..., 1)

    if model.rnn isa LSTM
        output, (hn, cn) = model.rnn(input, (hidden_lstm, hidden))
        hidden_state = (hn, cn)
    elseif model.rnn isa GRU
        output, hn = model.rnn(input, hidden)
        hidden_state = hn
    else # RNN
        output, hn = model.rnn(input, hidden)
        hidden_state = hn
    end

    # 对输出进行处理
    output = reshape(output, :, size(output, 3))
    output = model.projection_layer(output)
    output = softmax(output, dims=1)

    return output, hidden_state
end

function apply_constraints(output, counters, lengths, sequences, operators, min_length, max_length)
    # 添加小的 epsilon，以保证可以选择任何东西
    epsilon = ones(size(output)) * 1e-20
    output .+= epsilon

    # 确保满足最小长度
    min_boolean_mask = (counters .+ lengths .>= min_length)
    min_length_mask = max.(operators.nonzero_arity_mask, min_boolean_mask)
    output = min.(output, min_length_mask)

    # 确保不超过最大长度
    max_boolean_mask = (counters .+ lengths .<= max_length - 2)
    max_length_mask = max.(operators.zero_arity_mask, max_boolean_mask)
    output = min.(output, max_length_mask)

    # 确保所有表达式都有一个变量
    nonvar_zeroarity_mask = .!(operators.zero_arity_mask .& operators.nonvariable_mask)
    if lengths[1] == 0
        output = min.(output, nonvar_zeroarity_mask)
    else
        nonvar_zeroarity_mask = repeat(nonvar_zeroarity_mask, outer=(size(counters, 1), 1))
        counter_mask = (counters .== 1)
        contains_novar_mask = .!(any(sequences .== operators.variable_tensor, dims=2))
        last_token_and_no_var_mask = .!(counter_mask .& contains_novar_mask)
        nonvar_zeroarity_mask = max.(nonvar_zeroarity_mask, last_token_and_no_var_mask)
        output = min.(output, nonvar_zeroarity_mask)
    end

    return output
end

function get_next_input(parent_sibling, operators)
    # 对 -1 值的处理
    parent = abs.(parent_sibling[:, 1])
    sibling = abs.(parent_sibling[:, 2])

    # 生成一热编码张量
    parent_onehot = Flux.onehotbatch(parent, 1:length(operators))
    sibling_onehot = Flux.onehotbatch(sibling, 1:length(operators))

    # 使用掩码将 -1 值对应的位置零化
    parent_mask = (parent_sibling[:, 1] .!= -1)
    sibling_mask = (parent_sibling[:, 2] .!= -1)
    parent_onehot .*= parent_mask
    sibling_onehot .*= sibling_mask

    # 合并父节点和兄弟节点的一热编码
    input_tensor = hcat(parent_onehot, sibling_onehot)

    return input_tensor
end