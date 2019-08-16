#DQN
using Flux
using Distributions
using Gym
using Random
using Plots

env = GymEnv("CartPole-v1")
action_space_size = env.action_space.n

struct Agent
    dqn_model::Chain
end

function Agent()
    dqn_model = Chain(
        Dense(4,5, relu),
        Dense(5,5, relu),
        Dense(5, action_space_size),
    )
    return Agent(dqn_model)
end

Random.seed!(321)
Gym.seed!(env, 321)

action_agent = Agent()
value_agent = Agent()

opt = ADAM(0.01)
replay_size = 200000
batch_size = 200

γ = 0.999

global e = 1.
global e_min = 0.01
global e_decay = 0.995

probabilities = repeat([1. / action_space_size], action_space_size)

returns = []
state_batch = []
action_batch = []
reward_batch = []
next_state_batch = []

for i=1:1500

    state = reset!(env)
    episode_rewards = []

    while true
        push!(state_batch, state)

        #render(env)

        if rand() < max(e_min, min(e, 1. - log10(e_decay)))
            action = argmax(rand(Multinomial(1,probabilities)))
            push!(action_batch, action)
        else
            action = argmax(action_agent.dqn_model(state))
            push!(action_batch, action)
        end

        next_state, reward, done, _ = step!(env, action-1)

        push!(reward_batch, reward)
        push!(episode_rewards, reward)
        push!(next_state_batch, next_state)

        if done
            push!(done_batch, 1)
            state, done= reset!(env), false
            break
        else
            push!(done_batch, 0)
            state = next_state
        end
    end

    size_rewards = size(reward_batch)[1]
    out_size = max(0,size_rewards - batch_size)

    remove_sample = sort(Distributions.sample(collect(1:size_rewards), out_size, replace = false))
    deleteat!(action_batch, remove_sample)
    deleteat!(reward_batch, remove_sample)
    deleteat!(state_batch, remove_sample)
    deleteat!(next_state_batch, remove_sample)
    deleteat!(done_batch, remove_sample)


    action_batch_flat = collect(Iterators.flatten(action_batch))
    reward_batch_flat = collect(Iterators.flatten(reward_batch))

    sample_size = min(batch_size, size(action_batch_flat)[1])
    sampled = Distributions.sample(collect(1:size(state_batch)[1]), sample_size)

    action_sample = action_batch_flat[sampled]
    reward_sample = reward_batch_flat[sampled]
    state_sample = state_batch[sampled]
    next_state_sample = next_state_batch[sampled]

    state_outputs = action_agent.dqn_model.(state_sample)
    qvalues_state = [state_outputs[action_sample[i]] for i in size(action_sample)[1]]

    next_state_outputs = value_agent.dqn_model.(next_state_sample)
    qvalues_next_state = maximum.(next_state_outputs)

    bellman_values = Tracker.data(qvalues_next_state.*γ .+ reward_sample)

    Flux.train!(Flux.mse, Flux.params(action_agent.dqn_model), zip(qvalues_state, bellman_values), opt)

    global e *= e > e_min ? e_decay : 1.

    if i%100==0
        Flux.loadparams!(value_agent.dqn_model, Flux.params(action_agent.dqn_model))
    end
    push!(returns, sum(episode_rewards))
    println(sum(episode_rewards))
end

plot(returns)
