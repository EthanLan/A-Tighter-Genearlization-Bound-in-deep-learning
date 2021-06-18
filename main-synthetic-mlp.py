import numpy as np
import argparse

from synthetic_mlp_bounds import IncohBoundSimulator, CmiBoundSimulator

batch_size = 512                 # Total number of training images in a single batch
num_epochs = 1001                # Total number of training steps


parser = argparse.ArgumentParser(description='Tensorflow Mutual Information Bound')
parser.add_argument('--runs',default=20, type=int, help='the number of times to training DNNs')
parser.add_argument('--epochs', default=1001, type=int, help='the number of training epochs for each run')
parser.add_argument('--train_size',default=512, type=int, help='the number of all training samples')
parser.add_argument('--var',default=8, type=float, help='the variance of added Gaussian noise')
parser.add_argument('--nfilters',default=8, type=int, help='the number neurons in the first hidden layer')
parser.add_argument('--neurons',default=6, type=int, help='the number neurons in the second hidden layer')
parser.add_argument('--lr',default=0.003, type=float, help='the learning rate')

args = parser.parse_args()
print(args)


num_runs = args.runs
num_iteration = args.epochs
num_traindp = args.train_size

sq_gen_inoch = np.zeros((num_runs,num_iteration,4))
sq_gen_cmi = np.zeros((num_runs,num_iteration,4))
    
print(num_iteration)
print(num_runs)

for rep in range(num_runs):
    #_, _, _, sq_gen_grad_sample = GradBoundSimulator().train()
    #sq_gen_grad[rep, :] = sq_gen_grad_sample
    print('The %dth runs'%(rep))

    loss, train_acc, test_acc, _, _, cmi_bound = CmiBoundSimulator(epochs=args.epochs,train_size=args.train_size,var=args.var,nfilters=args.nfilters,n_neurons=args.neurons,lr=args.lr).train()
    sq_gen_cmi[rep,:,:] = np.concatenate((loss.reshape((-1,1)), train_acc.reshape((-1,1)), test_acc.reshape((-1,1)),cmi_bound.reshape((-1,1))),axis=1)

    loss, train_acc, test_acc, incoh_bound = IncohBoundSimulator(epochs=args.epochs,train_size=args.train_size,var=args.var,nfilters=args.nfilters,n_neurons=args.neurons,lr=args.lr).train()
    sq_gen_inoch[rep,:,:] = np.concatenate((loss.reshape((-1,1)), train_acc.reshape((-1,1)), test_acc.reshape((-1,1)),incoh_bound.reshape((-1,1))),axis=1)
        
    np.savez('MLP_Synthetic_Bound_%d62_var%d.npz'%(args.nfilters,args.var),sq_gen_cmi=sq_gen_cmi,sq_gen_inoch=sq_gen_inoch)
    
