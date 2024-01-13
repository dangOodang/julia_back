#=
dsr_main:
- Julia version: 1.9.3
- Author: dang
- Date: 2024-01-11
=#
# 先引入必要的Julia库，这些库可能需要安装
# 如果还没有安装，可以通过Julia的包管理器进行安装
using Flux  # 对应于Python中的PyTorch
using Random  # 用于随机数生成
using Plots  # 对应于matplotlib.pyplot
include("train.jl")
# Julia中函数定义的方式
function get_data()
    X = range(-1, stop=1.1, step=0.1)
    y = X.^3 + X.^2 + X
    X = reshape(X, length(X), 1)

    # 在Julia中，对数组的操作通常是原地的（in-place），例如shuffle!
    comb = collect(zip(X, y))
    shuffle!(comb)
    X, y = unzip(comb)

    # 计算用于训练的比例
    training_proportion = 0.2
    div = Int(floor(training_proportion * length(X)))
    X_constants, X_rnn = X[1:div], X[div+1:end]
    y_constants, y_rnn = y[1:div], y[div+1:end]

    return X_constants, X_rnn, y_constants, y_rnn
end

# 主函数
function main()
    # 加载数据
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # 执行回归任务
    results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin', 'var_x'],
        min_length = 2,
        max_length = 15,
        type = 'lstm',
        num_layers = 2,
        hidden_size = 250,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'rmsprop',
        inner_lr = 0.1,
        inner_num_epochs = 25,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 2000,
        scale_initial_risk = True,
        batch_size = 500,
        num_batches = 500,
        use_gpu = False,
        live_print = True,
        summary_print = True
    )

    # 解包结果
    epoch_best_rewards, epoch_best_expressions, best_reward, best_expression = results

    # 绘图
    plot(1:length(epoch_best_rewards), epoch_best_rewards, xlabel="Epoch", ylabel="Reward", title="Reward over Time")
end

main()