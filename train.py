import torch
import numpy as np
import torch.nn.functional as F
from time import time

def train(model, optimizer, criterion, user_input, item_input, labels, batch_size):
    model.train()
    print('train() start')

    data_len = len(user_input)

    rand_index = np.random.permutation(data_len)
    user_input = torch.tensor(user_input)
    item_input = torch.tensor(item_input)
    labels = torch.tensor(labels)

    total_loss = 0
    step_loss = 0
    t1 = time()
    num_step = data_len // batch_size

    for i in range(num_step):
        optimizer.zero_grad()

        user_batch = user_input[rand_index[i * batch_size : (i+1) * batch_size]]
        item_batch = item_input[rand_index[i * batch_size : (i+1) * batch_size]]
        label_batch = labels[rand_index[i * batch_size : (i+1) * batch_size]]

        prediction = model(user_batch, item_batch)
        loss = criterion(prediction.flatten(), label_batch.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        step_loss += loss
        if (i % 100 == 0):
            print('num step : %d // average loss : %.4f // total average loss : %.4f // step time : %.3f'
                    % (i, step_loss / 100, total_loss / (i+1), time()-t1))
            t1 = time()
            step_loss = 0

    return total_loss / num_step

