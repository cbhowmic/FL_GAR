import numpy as np
from matplotlib import pyplot as plt

plt.style.use("bmh")
# plt.style.use("seaborn-white")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 155
plt.rcParams['axes.facecolor'] = 'white'


def plot_subfigures(path0):
    attack_str = (attack_type + "_attacked" + str(num_attacked)) if attack else ""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 5))

    # key = "average"
    # path = path0 % (dataset, iid, network, num_worker, attack_str)
    # train_loss = np.load("%s/train_loss_%s.npy" % (path, key))[0]
    # print(train_loss)
    # steps = np.arange(0, np.shape(train_loss)[0])
    # plt.plot(steps, train_loss)
    # plt.show()

    all_keys = ["average", "median", "GM", "medoid"]
    # all_keys = []
    for rule in ['TM', 'krum', 'multi_krum', 'bulyan', 'multi_bulyan']:
        for trim in trim_size:
            rule_para = "%s_f%s" %(rule, trim)
            all_keys.append(rule_para)

    for rule in ['SM']:
        for temp in temperature:
            rule_para = "%s_T%s" %(rule, temp)
            all_keys.append(rule_para)

    for rule in ['TSM']:
        for trim in trim_size:
            for temp in temperature:
                rule_para = "%s_f%s_T%s" %(rule, trim, temp)
                all_keys.append(rule_para)

    print(all_keys)

    try:
        path = path0 % (dataset, iid, network, num_worker, attack_str)
        print('path:', path)

        train_loss, train_accuracy, test_loss, test_accuracy = {}, {}, {}, {}
        # for key in ["average", "median", "GM", "krum", "bulyan", "multi_krum", "multi_bulyan", "medoid", "SM", "TSM"]:
        for key in all_keys:
            print('key', key)
            # train_loss[key] = np.load("%s/train_loss_%s.npy" % (path, key))[0]
            # print(train_loss[key])
            try:
                train_loss[key] = np.load("%s/train_loss_%s.npy" % (path, key))[0]
                # print(train_loss[key])
                train_accuracy[key] = np.load("%s/train_acc_%s.npy" % (path, key))[0]
                test_loss[key] = np.load("%s/test_loss_%s.npy" % (path, key))[0]
                test_accuracy[key] = np.load("%s/test_acc_%s.npy" % (path, key))[0]
            except:
                print('failure')
                pass

        line_style = {"average": "-", "median": "-", "GM": "-", "krum": "-", "bulyan": "-", "multi_krum": "-", "multi_bulyan": "-",
                      "medoid": "--", "SM": "--", "TSM": "--"}
        # for key in ["average", "median", "GM", "krum", "bulyan", "multi_krum", "multi_bulyan", "medoid", "SM", "TSM"]:
        for key in all_keys:

            try:
                ax[0,0].plot(train_loss[key], linestyle=line_style[key], label=key)
                ax[0,1].plot(train_accuracy[key], linestyle=line_style[key])
                ax[1,0].plot(test_loss[key], linestyle=line_style[key])
                ax[1,1].plot(test_accuracy[key], linestyle=line_style[key])
                ax[0,0].legend()

            except:
                pass



            # plt.legend()

        # ax[0,0].xlabel(r'Communication round', fontsize=15)
        # ax[0,1].xlabel(r'Communication round', fontsize=15)
        # ax[1,0].xlabel(r'Communication round', fontsize=15)
        # ax[1,1].xlabel(r'Communication round', fontsize=15)
        # ax[0,0].ylabel(r'Training loss', fontsize=15)
        # ax[0,1].ylabel(r'Training accuracy', fontsize=15)
        # ax[1,0].ylabel(r'Testing loss', fontsize=15)
        # ax[1,1].ylabel(r'Testing accuracy', fontsize=15)
        plt.tight_layout()

    except:
        pass
    # ax[0, 0].xlabel(r'Communication round', fontsize=15)
    # ax[0, 1].xlabel(r'Communication round', fontsize=15)
    # ax[1, 0].xlabel(r'Communication round', fontsize=15)
    # ax[1, 1].xlabel(r'Communication round', fontsize=15)
    # ax[0, 0].ylabel(r'Training loss', fontsize=15)
    # ax[0, 1].ylabel(r'Training accuracy', fontsize=15)
    # ax[1, 0].ylabel(r'Testing loss', fontsize=15)
    # ax[1, 1].ylabel(r'Testing accuracy', fontsize=15)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataset = 'Spambase'
    network = 'MLP'
    num_worker = 20
    iid = 'iid'
    attack = False
    attack_type = "label_flipping"
    num_attacked = 2
    trim_size = [0,2,5,9]
    temperature = [10]


    path = "/home/bhowmic/PycharmProjects/FL_spambase/results/data_%s_%s_%s_%sclients_%s"
    plot_subfigures(path)