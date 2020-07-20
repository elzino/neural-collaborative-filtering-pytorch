from data_utils import Dataloader, get_train_instances
from eval import evaluate_model
from model import NeuMF
from train import train
from time import time


import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description='Neural Collaborative Filtering')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=32,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,64,32,16]',  # [64,32,16,8] [64,64,32,16]
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--pretrain_path', nargs='?', default='',
                        help='Specify the pretrain model file path. If empty, no pretrain will be used')

    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    verbose = args.verbose
    pretrain_path = args.pretrain_path

    topK = 10
    evaluation_threads = 1
    print("NeuMF arguments: %s " %(args))
    model_out_file = 'ModelWeights/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, layers, time())

    # Loading data
    t1 = time()
    data_loader = Dataloader(args.path + args.dataset)
    trainMatrix, testRatings, testNegatives = data_loader.trainMatrix, data_loader.testRatings, data_loader.testNegatives
    num_users, num_items = trainMatrix.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, trainMatrix.nnz, len(testRatings)))

    # Build model
    model = NeuMF(num_users, num_items, mf_dim, layers)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Load pretrain model
    if pretrain_path:
        model.load_state_dict(torch.load(pretrain_path))

    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, 1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        torch.save(model.state_dict(), model_out_file)

    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(trainMatrix, num_negatives)
        
        # Training
        avg_loss = train(model, optimizer, criterion, user_input, item_input, labels, batch_size)
        t2 = time()
        
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, avg_loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    torch.save(model.state_dict(), model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))

    if args.out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))

if __name__ == '__main__':
    main()