import torch
import os
import numpy as np

def flatten_list(message, byzantineSize):
    wList = [torch.cat([p.flatten() for p in parameters]) for parameters in message]
    wList.extend([torch.zeros_like(wList[0]) for _ in range(byzantineSize)])
    wList = torch.stack(wList)
    return wList


def unflatten_vector(vector, model):
    paraGroup = []
    cum = 0
    for p in model.parameters():
        newP = vector[cum:cum+p.numel()]
        paraGroup.append(newP.view_as(p))
        cum += p.numel()
    return paraGroup



def save_model(args, agg_rule, model):
    print('Saving the trained model...')
    attack_str = (args.attack_type + "_attacked" + str(args.num_attacked)) if args.attack else ""
    try:
        os.makedirs("models")
    except OSError:
        print("Creation of the directory failed")
    if agg_rule in ['TM', 'krum', 'multi_krum', 'bulyan', 'multi_bulyan']:
        torch.save(model.state_dict(), "models/Model_%s_%s_%s_%sclients_%s_%s_f%s.pt" % (args.dataset, args.iid, args.network, args.clients,
                                                                                     attack_str, agg_rule,  args.trim_size))
    elif agg_rule == 'SM':
        torch.save(model.state_dict(), "models/Model_%s_%s_%s_%sclients_%s_%s_T%s.pt" % (args.dataset, args.iid, args.network, args.clients,
                                                                     attack_str, agg_rule, args.temperature))
    elif agg_rule == 'TSM':
        trim_method = 'our'
        torch.save(model.state_dict(), "models/Model_%s_%s_%s_%sclients_%s_%s%s_f%s_T%s.pt" % (args.dataset, args.iid, args.network, args.clients,
                                                attack_str, agg_rule, trim_method, args.trim_size, args.temperature))
    else:
        torch.save(model.state_dict(), "models/Model_%s_%s_%s_%sclients_%s_%s.pt" % (args.dataset, args.iid, args.network, args.clients,
                                                               attack_str, agg_rule))



def save_data(args, agg_rule, train_loss, train_accuracy, test_loss, test_accuracy):
    print('Saving the data...')
    attack_str = (args.attack_type + "_attacked" + str(args.num_attacked)) if args.attack else ""
    try:
        os.makedirs("results/data_%s_%s_%s_%sclients_%s" % (args.dataset, args.iid, args.network, args.clients, attack_str))
    except OSError:
        print("Creation of the directory failed")
    if agg_rule in ['TM', 'krum', 'multi_krum', 'bulyan', 'multi_bulyan']:
        np.save('results/data_%s_%s_%s_%sclients_%s/train_loss_%s_f%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                              agg_rule, args.trim_size), train_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/train_acc_%s_f%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                             agg_rule, args.trim_size), train_accuracy)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_loss_%s_f%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                             agg_rule, args.trim_size), test_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_acc_%s_f%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                            agg_rule, args.trim_size), test_accuracy)
    elif agg_rule == 'SM':
        np.save('results/data_%s_%s_%s_%sclients_%s/train_loss_%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                            agg_rule, args.temperature), train_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/train_acc_%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                agg_rule, args.temperature), train_accuracy)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_loss_%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                agg_rule, args.temperature), test_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_acc_%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                agg_rule, args.temperature), test_accuracy)
    elif agg_rule == 'TSM':
        trim_method = 'our'
        np.save('results/data_%s_%s_%s_%sclients_%s/train_loss_%s%s_f%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                            agg_rule,  trim_method, args.trim_size, args.temperature), train_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/train_acc_%s%s_f%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                             agg_rule, trim_method, args.trim_size, args.temperature), train_accuracy)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_loss_%s%s_f%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                             agg_rule, trim_method, args.trim_size, args.temperature), test_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_acc_%s%s_f%s_T%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                             agg_rule, trim_method, args.trim_size, args.temperature), test_accuracy)
    else:
        np.save('results/data_%s_%s_%s_%sclients_%s/train_loss_%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                           agg_rule), train_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/train_acc_%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                         agg_rule), train_accuracy)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_loss_%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                       agg_rule), test_loss)
        np.save('results/data_%s_%s_%s_%sclients_%s/test_acc_%s.npy' % (args.dataset, args.iid, args.network, args.clients, attack_str,
                                                                                         agg_rule), test_accuracy)
    # else:
    #     try:
    #         os.makedirs("data_att/%s_%d" % (args.att_type, args.num_attacked))
    #     except OSError:
    #         print("Creation of the directory %s failed")
    #     np.save('data_att/%s_%d/test_acc_%s.npy' % (args.att_type, args.num_attacked, args.aggregate), test_accuracy,
    #             allow_pickle=False)
    #     np.save('data_att/%s_%d/train_loss_%s.npy' % (args.att_type, args.num_attacked, args.aggregate), train_loss,
    #             allow_pickle=False)