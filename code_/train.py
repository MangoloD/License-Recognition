import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from net import MainNet
from sample_data import Sampling
import matplotlib.pyplot as plt

if __name__ == '__main__':
    BATCH = 10
    EPOCH = 100
    save_path = r'../params/seq2seq.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)

    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Params!")

    train_data = Sampling(root="G:/DL_Data/Plate/train_plate", is_train=True)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True, num_workers=4)
    valid_data = Sampling(root="G:/DL_Data/Plate/validate_plate", is_train=False)
    valid_loader = data.DataLoader(dataset=valid_data, batch_size=BATCH, shuffle=False, num_workers=4)

    train_acc_list = []
    valid_acc_list = []
    for epoch in range(EPOCH):
        print("epoch--{}".format(epoch))
        net.train()
        true_num = 0
        for i, (x, y) in enumerate(train_loader):
            batch_x = x.to(device)
            batch_y = y.float().to(device)

            output = net(batch_x)
            loss = loss_func(output, batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            label_y = torch.argmax(y, 2).detach().numpy()
            out_y = torch.argmax(output, 2).cpu().detach().numpy()
            true_num += np.sum(out_y == label_y)
            if i % 50 == 0:
                print("loss:{:.6f}".format(loss.item()))
                print("label_y:\t", label_y[0])
                print("out_y:\t\t", out_y[0])

        train_acc = true_num / (len(train_data) * 7)
        print("train_acc:{:.2f}%".format(train_acc * 100))
        train_acc_list.append(train_acc)

        net.eval()
        true_num = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                batch_x = x.to(device)
                batch_y = y.float().to(device)
                output = net(batch_x)
                label_y = torch.argmax(y, 2).detach().numpy()
                out_y = torch.argmax(output, 2).cpu().detach().numpy()
                true_num += np.sum(out_y == label_y)
            valid_acc = true_num / (len(valid_data) * 7)
            print("valid_acc:{:.2f}%".format(valid_acc * 100))
            valid_acc_list.append(valid_acc)

        plt.clf()
        plt.plot(train_acc_list, label='train_acc')
        plt.plot(valid_acc_list, label='valid_acc')
        plt.title('accuracy')
        plt.legend()
        plt.savefig('../graph/acc_{}'.format(epoch + 1))

        torch.save(net.state_dict(), save_path)
