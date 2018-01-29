using MXNet

mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup model
model = mx.FeedForward(mlp, context=mx.gpu())

# optimization algorithm
optimizer = mx.SGD(lr=0.1, momentum=0.9)

# # Save checkpoints
# cb_save = mx.do_checkpoint( "mnist", frequency = 20, save_epoch_0 = true )

# fit parameters
# mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider, callbacks = [cb_save] )
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider )

print( "Start evaluating..." )

probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = Array[]
for batch in eval_provider
    push!(labels, copy(mx.get(eval_provider, batch, :softmax_label)))
end
labels = cat(1, labels...)

# Now we use compute the accuracy
correct = 0
for i = 1:length(labels)
    # labels are 0...9
    if indmax(probs[:,i]) == labels[i]+1
        correct += 1
    end
end
accuracy = 100correct/length(labels)
println(mx.format("Accuracy on eval set: {1:.2f}%", accuracy))

# Save model
mx.save( "mnist_arch.json", model.arch )
mx.save( "mnist_arg.params", model.arg_params )
mx.save( "mnist_aux.params", model.aux_params )
#=
Load Model as
    model = mx.FeedForward( mx.load("mnist_arch.json",mx.SymbolicNode), context = mx.gpu() )
    model.arg_params    = mx.load( "mnist_arg_params.json", mx.NDArray )
    model.aux_params = Dict{Symbol,mx.NDArray}()
=#

#=
Load Checkpoint as
    arch, arg_params, aux_params = mx.load_checkpoint("first", 100)    # arch is the network structure, arg_params contains the weights and biases
    mdl2 = mx.FeedForward(arch, context = mx.cpu())                    # Only populates the arch and ctx fields
    mdl2.arg_params = arg_params # Populate the arg_params fields
    mdl2.aux_params = aux_params
=#

#=
# MNIST training
using MXNet

mlp = @mx.chain mx.Variable(:data)	=>
	mx.FullyConnected( name = :fc1, num_hidden = 128 ) 	=>
	mx.Activation( name = :relu1, act_type = :relu )	=>
	mx.FullyConnected( name = :fc2, num_hidden = 64 )	=>
	mx.Activation( name = :relu2, act_type = :relu )	=>
	mx.FullyConnected( name = :fc3, num_hidden = 10 )	=>
	mx.SoftmaxOutput( name = :softmax )
	
# Data provider
batch_size 	= 100
include( Pkg.dir("MXNet","examples","mnist","mnist-data.jl") )	
train_provider, eval_provider = get_mnist_providers( batch_size )

# Setup model
model = mx.FeedForward( mlp, context = mx.cpu() )

# Optimization Algorithm
optimizer = mx.SGD( lr = 0.1, momentum = 0.9 )

# fit 
mx.fit( model, optimizer, train_provider, n_epoch, eval_data = eval_provider )
=#
