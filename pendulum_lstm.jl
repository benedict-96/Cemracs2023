using Pkg
cd("/Users/zeyuan/Documents/GitHub/GeometricMachineLearning.jl")
Pkg.activate(".")

using Lux
using JLD2
using MLUtils: DataLoader,splitobs
using Random
using ProgressMeter

# filename="pendulum_100samples_10001steps_qlist_plist_2607.jld2"
# filename="pendulum_3000samples_200steps_qlist_plist_2607.jld2"
filename="pendulum_3000PureSamples_200steps_qlist_plist_2707.jld2"

plist = load(filename,"plist")
qlist = load(filename,"qlist")

plist = hcat(plist...)'
qlist = hcat(qlist...)'

#100 samples, 10001 steps
#try to use (p,q)[1:20] to predict (p,q)[21]
#the input size of LSTMcell : in_dims * batchsize 
sequence_len = 100
batchsize = 5

data = cat(plist,qlist,dims=3)
perm = [3,2,1]
data = permutedims(data,perm)

# #delete some "bad" data
# #specifically for filename="pendulum_100samples_10001steps_qlist_plist_2607.jld2"
# data = data[:, :, [1:9; 11:end]]
# data = data[:, :, [1:11; 13:end]]
# data = data[:, :, [1:32; 34:end]]
# data = data[:, :, [1:30; 32:end]]
# data = data[:, :, [1:31; 33:end]]
# data = data[:, :, [1:42; 44:end]]
# data = data[:, :, [1:57; 59:end]]
# data = data[:, :, [1:74; 76:end]]
# data = data[:, :, [1:75; 77:end]]
# data = data[:, :, [1:83; 85:end]]

#delete the end element for beauty 😆
# data = data[:, :, 1:end-1]


fig = plot()
for i = 1:20
    plot!(fig,data[1,1:100,i],data[2,1:100,i])
end
display(fig)

#Prepare the data and split into 2 parts
train_input = data[:,1:sequence_len,:]
train_target = data[:,21:sequence_len+20,:]
(x_train,y_train),(x_val,y_val) = splitobs((train_input, train_target); at=29/30, shuffle=false)

@show size(x_train)  #(in_dims,sequence_len,train_samples) = (2,20,90)
@show size(y_train)  #(in_dims,train_samples) = (2,90)

#Define a loader, for each sample from the loader : x=(in_dims,sequence_len,batchsize,),y = (out_dims,batchsize) => (2×20×5 Array{Float64, 3}, 2×5 Matrix{Float64},)
train_loader = DataLoader((x_train,y_train),batchsize=50,shuffle = false)
val_loader = DataLoader((x_val,y_val),batchsize=1,shuffle = false)

#Define the Model 
input_dims = output_dims = 2
model = Recurrence(LSTMCell(input_dims => output_dims),return_sequence = true)
rng = Random.default_rng()
Random.seed!(rng, 0)
ps,st=Lux.setup(Random.default_rng(),model)

#Define the Loss function 
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


# To see the model performance on 100:200, since 1:100 are input data 
test_len = 20
output = []
train_loader = DataLoader((x_train,y_train),batchsize=1,shuffle = false)
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
truth = data[:,:,2901]
# See whether the right sample to compare
output[:,97:106]
truth[:,97:106]

#Plot the result
using Plots
plot(truth[1,90:200],truth[2,90:200],label="Truth")
plot!(output[1,90:200],output[2,90:200],label="Prediction")
plot!(output[1,90:100+test_len]-truth[1,90:100+test_len],output[2,90:100+test_len]-truth[2,90:100+test_len],label="Difference")


#Plot Gif, but only one line for now, either truth or prediction 
# p = plot(1,xlim = (-1, 1),ylim = (-1, 1.5))
fig = plot(1,xlim = (-1, 1),ylim = (-1, 1.5))
@gif for x=range(90,200)
    # push!(p, 1, (truth[1,x],truth[2,x]))
    push!(fig, 1, (output[1,x],output[2,x]))
end




# Conclusion
#   - LSTM Could not train well when has some noise data 
#   - Just soso when train without noise data 
#   - Discuss to use same standard. i.e. loss function