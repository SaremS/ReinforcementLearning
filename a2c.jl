#A2C
using Flux
using Distributions
using Gym
using Random
using Plots

env = GymEnv("CartPole-v1")

struct Agent
    input_model::Chain
    actor_head::Chain
    critic_head::Chain
end


function Agent()
    input_model = Chain(
        Dense(4, 32, relu)
    )

    actor_head = Chain(
        Dense(32, 2),
        softmax
    )

    critic_head = Chain(
        Dense(32, 1)
    )

    return Agent(input_model, actor_head, critic_head)
end



Random.seed!(123)
Gym.seed!(env, 123)
agent = Agent()
opt = ADAM(0.0001)
batch_size = 2000
returns = []

γ = 0.9999

for i=1:100
    total = 0
    state = reset!(env)

    #render(env)

    state_batch = []
    action_batch = []
    return_batch = []
    length_batch = []
    reward_batch = []
    mask_batch = []
    value_batch = []
    bellman_batch = []

    while true
        push!(state_batch, state)
        action_probs = Float64.(Tracker.data(agent.actor_head(agent.input_model(state))))

        #avoid erroneous softmax output
        action_probs = round.(action_probs; sigdigits = 2)
        if sum(action_probs) != 1.
            diff = 1. - sum(action_probs)
            action_probs[argmax(action_probs)[1]] += diff
        end

        #Multinomial returns one-hot encoded output
        action_mask = rand(Multinomial(1, action_probs), 1)
        action_mask = reshape(action_mask,2)
        action = (argmax(action_mask))[1]

        state_value = agent.critic_head(agent.input_model(state))[1]

        state, reward, done, _ = step!(env, action-1)

        push!(action_batch, action)
        push!(mask_batch, action_mask)
        push!(reward_batch, reward)
        push!(value_batch, state_value)

        if done

            episode_return = sum(reward_batch) #500 at max
            episode_length = size(reward_batch)[1]
            push!(return_batch, episode_return)
            push!(length_batch, episode_length)


            #Bellman updates
            bellman_updates = zeros(size(reward_batch)[1])
            R = 0.
            for t in reverse(1:(size(reward_batch)[1]))
                R = reward_batch[t] + γ * R
                bellman_updates[t] = R
            end

            #normalizing
            bellman_updates = (bellman_updates .- mean(bellman_updates)) ./ std(bellman_updates)
            push!(bellman_batch, bellman_updates)

            state, done, reward_batch = reset!(env), false, []

            if size(state_batch)[1] > batch_size
                break
            end

        end
    end

    #calculate batch advantages
    value_batch_flat = collect(Iterators.flatten(value_batch))
    bellman_batch_flat = collect(Iterators.flatten(bellman_batch))

    advantages = bellman_batch_flat .- value_batch_flat


    #calculate inputs to the pseudo-loss for policy gradient calculation
    weighted_logs = [Tracker.data(advantages[z])*log(sum(mask_batch[z] .* agent.actor_head(agent.input_model(state_batch[z])))) for z in 1:(size(state_batch)[1])]

    #combined actor&critic loss
    loss(logs, adv) = -mean(logs) + 0.5*mean(adv.^2)

    #Train one step
    Flux.train!(loss, Flux.params(agent.input_model, agent.actor_head, agent.critic_head), zip(weighted_logs, advantages), opt)

    push!(returns, mean(return_batch))
    println(mean(return_batch))
end

plot(returns)
