#'Deep' Policy Gradient
using Flux
using Distributions
using Gym
using Random
using Plots

env = GymEnv("LunarLander-v2")

struct Agent
    policy_model::Chain
end

function Agent()
    policy_model = Chain(
        Dense(8,32, relu),
        Dense(32,4, relu),
        softmax
    )
    return Agent(policy_model)
end

Random.seed!(123)
Gym.seed!(env, 123)
agent = Agent()
opt = ADAM(0.0001)
batch_size = 5000
returns = []

#This does not max out the episode reward but shows how the agent improves over time
for i=1:250
    total = 0
    state = reset!(env)

    #render(env)

    state_batch = []
    action_batch = []
    weight_batch = []
    return_batch = []
    length_batch = []
    reward_batch = []
    mask_batch = []

    while true
        push!(state_batch, state)
        action_probs = Float64.(Tracker.data(agent.policy_model(state)))

        #avoid erroneous softmax output
        action_probs = round.(action_probs; sigdigits = 2)
        if sum(action_probs) != 1.
            diff = 1. - sum(action_probs)
            action_probs[argmax(action_probs)[1]] += diff
        end

        #Multinomial returns one-hot encoded output
        action_mask = rand(Multinomial(1, action_probs), 1)
        action_mask = reshape(action_mask,4)
        action = (argmax(action_mask))[1]

        state, reward, done, _ = step!(env, action-1)

        push!(action_batch, action)
        push!(mask_batch, action_mask)
        push!(reward_batch, reward)

        if done

            episode_return = sum(reward_batch)
            episode_length = size(reward_batch)[1]
            push!(return_batch, episode_return)
            push!(length_batch, episode_length)

            #use episode return to weight the policy gradient further down
            push!(weight_batch, repeat([episode_return], episode_length))

            state, done, reward_batch = reset!(env), false, []

            if size(state_batch)[1] > batch_size
                break
            end

        end
    end

    #flatten out the weights from nested array to simple array
    weight_batch_flat = collect(Iterators.flatten(weight_batch))

    #calculate inputs to the pseudo-loss for policy gradient calculation
    weighted_logs = [weight_batch_flat[i]*log(sum(mask_batch[i] .* agent.policy_model(state_batch[i]))) for i in 1:(size(state_batch)[1])]
    loss(x) = -mean(x)

    #Flux does the rest - differentation for policy gradient and weight updates
    Flux.train!(loss, Flux.params(agent.policy_model), weighted_logs, opt)

    push!(returns, mean(return_batch))
    println(mean(return_batch))
end

plot(returns)
