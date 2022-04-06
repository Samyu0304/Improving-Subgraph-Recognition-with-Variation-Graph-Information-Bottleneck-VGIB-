import torch
import random
from tqdm import trange
from layers import Subgraph
from utils import  GraphDatasetGenerator
import itertools
import itertools
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import json
from tqdm import tqdm
import numpy as np

import os

class Subgraph_Learning(object):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective Cautious Iteration model.
    """
    def __init__(self, args):
        super(Subgraph_Learning, self).__init__()
        """
        Creating dataset, doing dataset split, creating target and node index vectors.
        :param args: Arguments object.
        """
        self.args = args
        self.dataset_generator = GraphDatasetGenerator(self.args.data)
        self.batch_size = self.args.batch_size
        self.use_mi = self.args.use_mi
        self.train_percent = self.args.train_percent
        self.valiate_percent = self.args.validate_percent
        self.save_stat_img = self.args.save + 'stat.png'
        self.D_criterion = torch.nn.BCEWithLogitsLoss()
        self.inner_loop = self.args.inner_loop
        self.noise_scale = self.args.noise_scale

    def _dataset_spilt(self):
        Data_Length = len(self.dataset_generator.graphs)
        Training_Length = int(self.train_percent * Data_Length)
        Validate_Length = int(self.valiate_percent * Data_Length)
        Testing_Length = Data_Length - Training_Length - Validate_Length
        print(Training_Length)



        test_ind = [i for i in range(0, Testing_Length)]
        all_ind = [j for j in range(0, Data_Length)]
        train_val_ind = list(set(all_ind)-set(test_ind))

        train_ind = train_val_ind[0:Training_Length]
        validate_ind = train_val_ind[Training_Length:]

        self.training_data = [self.dataset_generator.graphs[i] for i in train_ind]
        self.valiate_data = [self.dataset_generator.graphs[i] for i in validate_ind]
        self.testing_data = [self.dataset_generator.graphs[i] for i in test_ind]



    def _setup_model(self):
        """
        Creating a SEAL model.
        """
        self.model = Subgraph(self.args, self.dataset_generator.number_of_features)

        if torch.cuda.is_available():
            self.model = Subgraph(self.args, self.dataset_generator.number_of_features).cuda()


    def set_requires_grad(self, net, requires_grad=False):

        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



    def fit_a_single_model(self):
        """
        Fitting a single SEAL model.

        """
        self._dataset_spilt()

        self._setup_model()

        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters()),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)


        Data_Length = len(self.training_data)
        Num_split = int(Data_Length / self.batch_size)

        best_mean = 1.0

        save_loss = []
        Iter = 0
        #mi_weight = 0

        for Epoch in tqdm(range(self.args.epochs)):

            for i in range(0, Num_split):

                Iter += 1

                data = self.training_data[int(i*self.batch_size): min(int((i+1)*self.batch_size),Data_Length)]

                embeddings, positive, noisy_embedding, mi_loss,  cls_loss, positive_penalty, preserve_rate = self.model(data)

                loss = cls_loss + positive_penalty

                optimizer.zero_grad()

                loss = loss + self.args.mi_weight * mi_loss

                loss.backward()

                optimizer.step()

                print("MI_pen:%.2f,CLS:%.2f,Pen:%.2f,Pre:%.2f"%(self.args.mi_weight * mi_loss,cls_loss,positive_penalty,preserve_rate))

                one_save_loss = str(self.args.mi_weight * mi_loss) + ' ' + str(cls_loss) + ' ' + str(positive_penalty / self.args.con_weight) + '\n'

                save_loss.append(one_save_loss)

        save_loss_path = self.args.save + 'loss.txt'

        with open(save_loss_path,'w') as F:
            F.writelines(save_loss)


    def return_index(self,data):
        self.model.eval()
        ind = self.model.assemble(data)

        return ind



    def test(self):
        #return a list of index of activated mol
        ind = self.return_index(self.testing_data)
        count = 0

        for data in ind:
            save_path = os.path.join(self.args.save, str(count) + '.json')
            dump_data = json.dumps(data)
            F = open(save_path, 'w')
            F.write(dump_data)
            F.close()
            count += 1

    def fit(self):
        """
        Training models sequentially.
        """
        print("\nTraining started.\n")
        self.fit_a_single_model()


if __name__ == '__main__':
    import torch
    a = torch.FloatTensor([[1],[3],[5]])
    b = torch.FloatTensor([[1],[3],[5]])
    print(a/(a + b))
