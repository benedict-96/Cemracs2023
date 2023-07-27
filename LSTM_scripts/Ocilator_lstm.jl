using Pkg
cd("/Users/zeyuan/Documents/GitHub/GeometricMachineLearning.jl")
Pkg.activate(".")

using Lux
using JLD2
using MLUtils: DataLoader,splitobs
using Random
using ProgressMeter
using Optimisers
using Zygote
using Plots

filename="/Users/zeyuan/Documents/GitHub/Cemracs2023/LSTM_scripts/Ocilator_9Samples_1000steps_2707.jld2"

p1list = load(filename,"p1list")
p2list = load(filename,"p2list")
q1list = load(filename,"q1list")
q2list = load(filename,"q2list")

p1list = hcat(p1list...)'
p2list = hcat(p2list...)'
q1list = hcat(q1list...)'
q2list = hcat(q2list...)'

data = cat(p1list,p2list,q1list,q2list,dims=3)
perm = [3,2,1]
data = permutedims(data,perm)

sequence_len = 100
train_input = data[:,1:sequence_len,:]
train_target = data[:,21:sequence_len+20,:]
(x_train,y_train),(x_val,y_val) = splitobs((train_input, train_target); at=8/9, shuffle=false)

train_loader = DataLoader((x_train,y_train),batchsize=1,shuffle = false)
val_loader = DataLoader((x_val,y_val),batchsize=1,shuffle = false)

input_dims = output_dims = 4
model = Recurrence(LSTMCell(input_dims => output_dims),return_sequence = true)
rng = Random.default_rng()
Random.seed!(rng, 0)
ps,st=Lux.setup(Random.default_rng(),model)

function compute_loss(x, y, model, ps, st)
    # @show size(y)
    y_pred, st = model(x, ps, st)
    seq_len = size(y,2)
    batchsize = size(y,3)
    # @show batchsize
    # @show size(y_pred)
    error = sum(sum(abs.(y[:,i,:] - y_pred[i]) for i in 1:seq_len)/seq_len)/batchsize
    # @show error
    # @show size(error)
    return error, y_pred, st
end


#Define a Optimisers
opt = Optimisers.ADAM(0.1)
st_opt = Optimisers.setup(opt, ps)

#Start the Training Process
epochs = 500
err_ls = []  
@showprogress for epoch in 1:epochs
    err = 0  
    for (x,y) in train_loader
        # @show size(x)
        # y, st = model(x, ps, st)
        gs = Zygote.gradient(p -> compute_loss(x,y,model,p,st)[1],ps)[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
        # @show y
        # @show size(y)
        err += compute_loss(x,y,model,ps,st)[1]
        # push!(err_ls,err)
    end
    push!(err_ls,err/length(train_loader))
end

err_ls
plot(err_ls)

test_len = 20
output = []
for (x,y) in val_loader
    for _ in range(1,5)
        # @show size(x)
        # @show size(y)
        y_pred, st = model(x, ps, st)
        # @show size(y_pred)
        # @show size(y_pred[end-test_len:end])
        for i in 1:test_len
            x = cat(x, y_pred[end-test_len+i], dims=2)
        end
        # @show size(x)
        # x = x[:, 2:end, :]
        # @show x
        # break
    end
    @show size(x)
    output = x
    # err = sum(abs2,y_pred .- y)
    # @show err
    break
end


#Find the truth to compare 
output = output[:,:,1]
truth = data[:,1:200,9]
# See whether the right sample to compare
output[:,99:106]
truth[:,99:106]

plot(truth[1,90:110],truth[2,90:110],label="Truth")
plot!(output[1,90:110],output[2,90:110],label="Prediction")


fig = plot(1,xlim = (-4, 4),ylim = (-0.5, 0.5))
@gif for x=range(90,110)
    push!(fig, 1, (truth[1,x],truth[2,x]))
end

fig1 = plot(1,xlim = (-4, 4),ylim = (-0.5, 0.5))
@gif for x=range(90,110)
    push!(fig1, 1, (output[1,x],output[2,x]))
end



