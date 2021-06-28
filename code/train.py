import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from net import MainNet
from sample_data import Sampling
from utils import label_to_str


def train(train_data_path, validate_data_path, params_path, index=0, max_acc=0.2):
    batch_size = 10
    epoch_num = 100
    start_num = 0
    avg_num = 5
    params = f"{params_path}/seq2seq_{index}.pth"

    train_data = Sampling(train_data_path)
    validate_data = Sampling(validate_data_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)

    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = nn.MSELoss()

    if os.path.exists(params):
        net.load_state_dict(torch.load(params))
        print("Load Params Successful...")
    else:
        print("You don't have any Params...")

    evaluate_acc = []
    for epoch in range(epoch_num):
        print(f"\nEpoch:{epoch + 1}")
        net.train()
        train_loss = []
        train_acc = []
        for image, label in train_loader:
            data_image = image.to(device)
            data_label = label.float().to(device)
            output = net(data_image)
            # output = output.permute(1, 0, 2)
            # target_length = torch.randint(1, 7, (batch_size,))
            loss = loss_fn(output, data_label)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tra_output = torch.argmax(output, dim=2).detach().cpu().numpy()
            tra_label = torch.argmax(data_label, dim=2).detach().cpu().numpy()

            accuracy = np.sum(tra_output == tra_label, dtype=np.float32) / (batch_size * 7)
            train_acc.append(accuracy)
        # print(train_loss, train_acc)
        print(f"tra_loss:{np.mean(train_loss)}\t tra_acc:{np.mean(train_acc)}")
        # exit()

        net.eval()
        validate_loss = []
        validate_acc = []
        for i, (image, label) in enumerate(validate_loader):
            data_image = image.to(device)
            data_label = label.float().to(device)

            output = net(data_image)
            loss = loss_fn(output, data_label)

            validate_loss.append(loss.item())

            val_output = torch.argmax(output, 2).detach().cpu().numpy()
            val_label = torch.argmax(data_label, 2).detach().cpu().numpy()
            accuracy = np.sum(val_output == val_label, dtype=np.float32) / (batch_size * 7)
            validate_acc.append(accuracy)

            if (i + 1) % 10 == 0:
                print("label_y:", label_to_str(val_label[0]))
                print("out_y:", label_to_str(val_output[0]))
        print(f"val_loss:{np.mean(validate_loss)}\t val_acc:{np.mean(validate_acc)}")

        evaluate_acc.append(np.mean(validate_acc))

        # 模型保存
        if np.mean(validate_acc) > max_acc:
            max_acc = np.mean(validate_acc)
            torch.save(net.state_dict(), f"{params_path}/{index + epoch}.pth")
            print("The params is saved successfully...")

        # 过拟合判断
        acc_length = len(evaluate_acc)
        if (acc_length - start_num) % 10 == 0:
            start_r2 = evaluate_acc[start_num:acc_length - avg_num]
            end_r2 = evaluate_acc[acc_length - avg_num:acc_length]
            if np.mean(start_r2) > np.mean(end_r2):
                exit()
            start_num += avg_num


if __name__ == '__main__':
    train_path = "G:/DL_Data/Plate/train_plate"
    validate_path = "G:/DL_Data/Plate/validate_plate"
    pth_path = "../params"
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    train(train_path, validate_path, pth_path)
