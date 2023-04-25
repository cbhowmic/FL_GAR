" Step 1: Import the relevant packages and modules "
import syft as sy
from threading import Thread
from FLDataset import *
# from utils import *
from aggregation import *
# from plot import *
from networks import *
from operations import *
from config import *

" Step 2: specify the arguments needed for the experiment "
args = Arguments()

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


" Step 3: specify the device : CPU or GPU "
device = get_device()
kwargs = {'num_workers': 1, 'pin_memory': True}
torch.manual_seed(1234)

" Step 4: Get the testing dataset ready"
global_train, global_test, train_group, test_group = load_SpamDataset(args.clients, args.iid)
global_test_loader = DataLoader(global_test, batch_size=args.global_batches, shuffle=False)

print('Details: \n Num of clients:', args.clients, '\n Dataset:', args.dataset, '\n Network:', args.network,
      '\n Attack: ', args.attack, '\n Number of attacked:', args.num_attacked, '\n Attack type:', args.attack_type)


" Step 5: Find the loss and accuracy on global test dataset "
for rule in args.aggregate:
    print('============================\n')
    print('Aggregation rule : ', rule)

    # Set the seed, initialization
    torch.manual_seed(args.torch_seed)

    # Load the trained model
    try:
        trained_model = load_model(args, rule)
    except:
        ValueError('Model could not be loaded')

    # Calculate loss and accuracy on the testing dataset
    test_l, test_acc = loss_and_accuracy(trained_model, device, global_test_loader, 'Test')

    print(f'Aggregation method:', rule, 'Testing loss:', test_l, 'Testing accuracy:', test_acc)

