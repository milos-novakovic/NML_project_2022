import numpy as np
import matplotlib.pyplot as plt


def my_plot(l1: list,
            l2: list,
            title: str,
            xlabel: str,
            ylabel: str,
            name1: str,
            name2: str,
            units: str = ""
            ):
    l1_np = np.array(l1)
    l2_np = np.array(l2)
    plt.plot(l1_np)
    plt.plot(l2_np)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([name1, name2])
    plt.grid(b=True, axis='both')
    plt.minorticks_on()
    print(f"Avarage value of {name1} = {np.mean(l1_np):.2f} {units}")
    print(f"Avarage value of {name2} = {np.mean(l2_np):.2f} {units}")
    plt.show()


file_name1 = "node_sampler_training_evaluation_test"
file_name2 = "random_walk_sampler_training_evaluation_test"


with open(file_name1 + ".txt") as f:
    lines1 = f.readlines()

with open(file_name2 + ".txt") as f:
    lines2 = f.readlines()

non_desirable_words = ['train_sampling.py:121: UserWarning:', '/usr/local/lib/python3.7/',
                       'warnings.warn', 'torch.nn.utils.clip_grad_norm', 'tcmalloc: large alloc']  # ,'',]

ns_loss_per_epoch = []
ns_time_sec_per_epoch = []
ns_acc_mini_per_epoch = []
ns_acc_macro_per_epoch = []

rw_loss_per_epoch = []
rw_time_sec_per_epoch = []
rw_acc_mini_per_epoch = []
rw_acc_macro_per_epoch = []


lines1_updated = []
for line in lines1:
    add_word = True
    for non_desirable_word in non_desirable_words:
        if non_desirable_word in line:
            add_word = False

    if add_word:
        lines1_updated.append(line)

        if 'epoch:' in line:
            ns_loss_per_epoch.append(float(line.split(' ')[-1]))
        elif 'Finished training epoch' in line:
            ns_time_sec_per_epoch.append(float(line.split(' ')[-2]))
        elif 'Val' in line and not '#Val' in line:
            ns_acc_mini_per_epoch.append(float(line.split(' ')[2][:-1]))
            ns_acc_macro_per_epoch.append(float(line.split(' ')[5][:-1]))
        elif "training using time" in line:
            print(
                f"Total Training Time (for Node sampler) = {float(line.split(' ')[-1][:-1]) : .2f} sec.")
        elif "Test F1-mic" in line:
            print(
                f"Test F1-micro accuracy (for Node sampler) = {float(line.split(' ')[2][:-1])}")
            print(
                f"Test F1-macro accuracy (for Node sampler) = {float(line.split(' ')[5][:-1])}")


lines2_updated = []
for line in lines2:
    add_word = True
    for non_desirable_word in non_desirable_words:
        if non_desirable_word in line:
            add_word = False

    if add_word:
        lines2_updated.append(line)
        if 'epoch:' in line:
            rw_loss_per_epoch.append(float(line.split(' ')[-1]))
        elif 'Finished training epoch' in line:
            rw_time_sec_per_epoch.append(float(line.split(' ')[-2]))
        elif 'Val' in line and not '#Val' in line:
            rw_acc_mini_per_epoch.append(float(line.split(' ')[2][:-1]))
            rw_acc_macro_per_epoch.append(float(line.split(' ')[5][:-1]))
        elif "training using time" in line:
            print(
                f"Total Training Time (for Randowm walk sampler) = {float(line.split(' ')[-1][:-1]) : .2f} sec.")
        elif "Test F1-mic" in line:
            print(
                f"Test F1-micro accuracy (for Randowm walk sampler) = {float(line.split(' ')[2][:-1])}")
            print(
                f"Test F1-macro accuracy (for Randowm walk sampler) = {float(line.split(' ')[5][:-1])}")

with open(file_name1 + "_cleared.txt", 'w') as f:
    f.write(''.join(lines1_updated))

with open(file_name2 + "_cleared.txt", 'w') as f:
    f.write(''.join(lines2_updated))


my_plot(l1=ns_loss_per_epoch, l2=rw_loss_per_epoch, title="Training Losses",
        xlabel="Epoch",
        ylabel="Loss",
        name1="Node Sampler",
        name2="Random Walk Sampler",
        units=""
        )

my_plot(l1=ns_acc_mini_per_epoch, l2=rw_acc_mini_per_epoch, title="Training Accuracy Mini",
        xlabel="Epoch",
        ylabel="Accuracy Mini",
        name1="Node Sampler",
        name2="Random Walk Sampler",
        units=""
        )

my_plot(l1=ns_acc_macro_per_epoch, l2=rw_acc_macro_per_epoch, title="Training Accuracy Macro",
        xlabel="Epoch",
        ylabel="Accuracy Macro",
        name1="Node Sampler",
        name2="Random Walk Sampler",
        units=""
        )

my_plot(l1=ns_time_sec_per_epoch, l2=rw_time_sec_per_epoch, title="Training time",
        xlabel="Epoch",
        ylabel="Training per epoch [seconds]",
        name1="Node Sampler",
        name2="Random Walk Sampler",
        units="sec"
        )


print('end')
