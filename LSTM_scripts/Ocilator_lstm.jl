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
using LinearAlgebra:norm

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


sequence_len = 8
train_input = data[:,1:sequence_len,:]
train_target = data[:,2:sequence_len+1,:]
for i in 1:1001-sequence_len-1
    train_input = cat(train_input,data[:,i+1:i+sequence_len,:],dims=3)
    train_target = cat(train_target,data[:,i+2:i+sequence_len+1,:],dims=3)
end
# 4*8*8937


# input is (4, 100, 81),i.e. the first 9*100 steps for each sample 
# target is 20- 120,120-220,...820-920 for each sample 

(x_train,y_train),(x_val,y_val) = splitobs((train_input, train_target); at=991/993, shuffle=false)
batchsize = 9
train_loader = DataLoader((x_train,y_train),batchsize=batchsize,shuffle = false)
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
    # @show size(y_pred[1])
    # error = sum(sum(abs.(y[:,i,:] - y_pred[i]) for i in 1:seq_len)/seq_len)/batchsize
    error = sum(sum((abs.(y[:,i,:] - y_pred[i])).^2 for i in 1:seq_len))/(seq_len^(0.5) * batchsize)
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
        # break
    end
    push!(err_ls,err/length(train_loader))
    # break 
end

err_ls
plot(err_ls)


test_len = 8
output = []
for (x,y) in val_loader
    for _ in range(1,2)
        @show size(x)
        @show size(y)
        y_pred, st = model(x, ps, st)
        @show size(y_pred[end][:,1])
        # @show size(y_pred[end-test_len:end])
        # for i in 1:test_len
        x = cat(x, y_pred[end][:,1], dims=2)
        @show size(x)
        # end
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
truth = data[:,992:1001,1]

# See whether the right sample to compare
# output[:,99:106]
# truth[:,99:106]

plot(truth[1,:],truth[2,:],label="Truth")
plot!(output[1,:],output[2,:],label="Prediction")


fig = plot(1,xlim = (-0.2, 0.5),ylim = (-0.3, 0.3))
@gif for x=range(1,10)
    push!(fig, 1, (truth[1,x],truth[2,x]))
end

fig1 = plot(1,xlim = (-0.2, 0.5),ylim = (-0.3, 0.3))
@gif for x=range(1,10)
    push!(fig1, 1, (output[1,x],output[2,x]))
end



# p,q vs t in val set 
plot(range(1,200),truth[1,1:200],label="Truth",)
plot!(range(1,200),output[1,1:200],label="Prediction")
xlims!(90,200)
vline!([100],label = "t=100",color=:black,linestyle=:dash,linewidth=1)
vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Val P1")

plot(range(1,200),truth[2,1:200],label="Truth")
plot!(range(1,200),output[2,1:200],label="Prediction")
xlims!(90,200)
vline!([100],label = "t=100",color=:black,linestyle=:dash,linewidth=1)
vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Val P2")

plot(range(1,200),truth[3,1:200],label="Truth")
plot!(range(1,200),output[3,1:200],label="Prediction")
xlims!(90,200)
vline!([100],label = "t=100",color=:black,linestyle=:dash,linewidth=1)
vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Val P3")

plot(range(1,200),truth[4,1:200],label="Truth")
plot!(range(1,200),output[4,1:200],label="Prediction")
xlims!(90,200)
vline!([100],label = "t=100",color=:black,linestyle=:dash,linewidth=1)
vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Val P4")



# See what about the training set 
plot_train_loader = DataLoader((x_train,y_train),batchsize=1,shuffle = false)

for (x,y) in plot_train_loader
    for _ in range(1,7)
        @show size(x)
        @show size(y)
        y_pred, st = model(x, ps, st)
        @show size(y_pred[end][:,1])
        # @show size(y_pred[end-test_len:end])
        # for i in 1:test_len
        x = cat(x, y_pred[end][:,1], dims=2)
        @show size(x)
        # end
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
truth = data[:,1:15,1]
# See whether the right sample to compare
output[:,99:106]
truth[:,99:106]

pre_seq_len = size(output,2)

plot(truth[1,:],truth[2,:],label="Truth")
plot!(output[1,:],output[2,:],label="Prediction")

# Momenta
fig = plot(1,xlim = (-0.2, 0.2),ylim = (-0.3, 0.3))
@gif for x=range(90,150)
    push!(fig, 1, (truth[1,x],truth[2,x]))
end

fig1 = plot(1,xlim = (-0.2, 0.2),ylim = (-0.3, 0.3))
@gif for x=range(90,150)
    push!(fig1, 1, (output[1,x],output[2,x]))
end

# Position 
fig2 = plot(1,xlim = (-0.2, 0.2),ylim = (-0.5, 0.5))
@gif for x=range(90,150)
    push!(fig2, 1, (truth[3,x],truth[4,x]))
end

fig3 = plot(1,xlim = (-0.2, 0.2),ylim = (-0.5, 0.5))
@gif for x=range(90,150)
    push!(fig3, 1, (output[3,x],output[4,x]))
end



# p,q vs t in training set 
plot(range(1,pre_seq_len),truth[1,1:pre_seq_len],label="Truth",)
plot!(range(1,pre_seq_len),output[1,1:pre_seq_len],label="Prediction")
# xlims!(90,200)
vline!([8],label = "t=8",color=:black,linestyle=:dash,linewidth=1)
# vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Train P1")



plot(range(1,pre_seq_len),truth[2,1:pre_seq_len],label="Truth")
plot!(range(1,pre_seq_len),output[2,1:pre_seq_len],label="Prediction")
# xlims!(90,200)
vline!([8],label = "t=8",color=:black,linestyle=:dash,linewidth=1)
# vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Train P2")



plot(range(1,pre_seq_len),truth[3,1:pre_seq_len],label="Truth")
plot!(range(1,pre_seq_len),output[3,1:pre_seq_len],label="Prediction")
# xlims!(90,200)
vline!([8],label = "t=8",color=:black,linestyle=:dash,linewidth=1)
# vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Train P3")



plot(range(1,pre_seq_len),truth[4,1:pre_seq_len],label="Truth")
plot!(range(1,pre_seq_len),output[4,1:pre_seq_len],label="Prediction")
# xlims!(90,200)
vline!([8],label = "t=8",color=:black,linestyle=:dash,linewidth=1)
# vline!([120],label = "t=120",color=:black,linestyle=:dash,linewidth=2)
title!("Train P4")

# For training set ,it could not predict well