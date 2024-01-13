#=
train:
- Julia version: 1.9.3
- Author: dang
- Date: 2024-01-11
=#
using Flux
using Statistics
using LinearAlgebra

function train(
    X_constants, y_constants, X_rnn, y_rnn, operator_list,
    min_length=2, max_length=12, rnn_type="lstm", num_layers=1, dropout=0.0, lr=0.0005,
    optimizer_type="adam", inner_optimizer_type="rmsprop", inner_lr = 0.1, inner_num_epochs=15,
    entropy_coefficient=0.005, risk_factor=0.95, initial_batch_size=2000, scale_initial_risk=True,
    batch_size=500, num_batches=200, hidden_size=500, use_gpu=true, live_print=true, summary_print=true
)

    epoch_best_rewards = []
    epoch_best_expressions = []

    # 初始化 RNN 和优化器
    operators = Operators(operator_list)
    dsr_rnn = DSRRNN(operators, hidden_size, min_length, max_length, rnn_type, num_layers, dropout)
    optimizer = optimizer_type == "adam" ? ADAM(lr) : RMSProp(lr)

    best_expression = nothing
    best_performance = -Inf

    start = time()
    sequences, sequence_lengths, log_probabilities, entropies = sample_sequence(dsr_rnn, initial_batch_size)

    for i in 1:num_batches
        # 将序列转换为可以评估的表达式
        expressions = [Expression(operators, sequences[j], sequence_lengths[j]) for j in 1:size(sequences, 1)]

        # 优化常数
        optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer_type)

        # 测试表达式
        rewards = [benchmark(expression, X_rnn, y_rnn) for expression in expressions]

        # 更新最佳表达式
        epoch_best_index = argmax(rewards)
        epoch_best_expression = expressions[epoch_best_index]
        push!(epoch_best_rewards, rewards[epoch_best_index])
        push!(epoch_best_expressions, epoch_best_expression)

        if rewards[epoch_best_index] > best_performance
            best_performance = rewards[epoch_best_index]
            best_expression = epoch_best_expression
        end

        # 提前停止条件
        if best_performance >= 0.98
            live_print && println("~ Early Stopping Met ~\nBest Expression: $best_expression")
            break
        end

        # 计算风险阈值
        threshold = i == 1 && scale_initial_risk ?
            quantile(rewards, 1 - (1 - risk_factor) / (initial_batch_size / batch_size)) :
            quantile(rewards, risk_factor)

        indices_to_keep = findall(reward -> reward > threshold, rewards)

        # 选择奖励、概率和熵的子集
        rewards = rewards[indices_to_keep]
        log_probabilities = log_probabilities[indices_to_keep]
        entropies = entropies[indices_to_keep]

        # 计算风险寻求和熵梯度
        risk_seeking_grad = sum((rewards .- threshold) .* log_probabilities) / length(rewards)
        entropy_grad = entropy_coefficient * sum(entropies) / length(rewards)

        # 计算损失并反向传播
        loss = -lr * (risk_seeking_grad + entropy_grad)
        Flux.back!(loss)
        update!(optimizer, params(dsr_rnn), Flux.gradients())

        # 打印每个 Epoch 的总结
        live_print && println("Epoch: $i\nEntropy Loss: $entropy_grad\nRisk-Seeking Loss: $risk_seeking_grad\nTotal Loss: $loss\nBest Performance (Overall): $best_performance\nBest Performance (Epoch): $(rewards[epoch_best_index])\nBest Expression (Overall): $best_expression\nBest Expression (Epoch): $epoch_best_expression")

        # 为下一个批次采样
        sequences, sequence_lengths, log_probabilities, entropies = sample_sequence(dsr_rnn, batch_size)
    end

    summary_print && println("\nTime Elapsed: $(time() - start)\nEpochs Required: $i\nBest Performance: $best_performance\nBest Expression: $best_expression")

    return epoch_best_rewards, epoch_best_expressions, best_performance, best_expression
end

function benchmark(expression, X_rnn, y_rnn)
    y_pred = expression(X_rnn)
    return reward_nrmse(y_pred, y_rnn)
end

function reward_nrmse(y_pred, y_rnn)
    mse_loss = Flux.mse(y_pred, y_rnn)
    rmse_val = sqrt(mse_loss) # 计算 RMSE
    rmse_val = rmse_val * std(y_rnn) # 使用目标的标准差进行归一化
    rmse_val = isnan(rmse_val) ? 1e10 : min(rmse_val, 1e10) # 修正 NaN 并剪辑
    return 1 / (1 + rmse_val) # Squash
end