import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
from itertools import islice
from torch.utils.data import DataLoader, Dataset
import time
import csv
import torch
from torch.autograd import Variable
from Bio import SeqIO
import random
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# %%
CUDA_FLAG = torch.cuda.is_available()
print(CUDA_FLAG)
# %%

# %%
SEED = 1
torch.manual_seed(SEED)
if CUDA_FLAG:
    torch.cuda.manual_seed(SEED)

SEQ_LEN = 155
EMBEDDING_LEN = 120
NUM_TRAINING_PAIRS = 20 * 500
NUM_EPOCH = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 100


class Config():
    train_data_fp = './converted/pair_shuffle.fa'
    train_target_fp = './converted/dist_shuffle.txt'
    train_num_example = NUM_TRAINING_PAIRS
    train_batch_size = BATCH_SIZE
    num_epoch = NUM_EPOCH
    learning_rate = LEARNING_RATE


# %%
class MaxMinout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding1, embedding2):
        shape = list(embedding1.size())
        flat1 = embedding1.view(1, -1)
        flat2 = embedding2.view(1, -1)
        combined = torch.cat((flat1, flat2), 0)
        maxout = combined.max(0)[0].view(*shape)
        # minout = combined.min(0)[0].view(*shape)
        minout = ((combined * -1).max(0)[0].view(*shape) * -1)  # workaround for memory leak bug

        return maxout, minout


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxminout = MaxMinout()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU(),
        )
        max_pooling_len = SEQ_LEN
        max_pooling_len = np.floor((max_pooling_len - 2) / 2 + 1)
        max_pooling_len = np.floor((max_pooling_len - 2) / 2 + 1)
        max_pooling_len = np.floor((max_pooling_len - 2) / 2 + 1)
        self.fc = nn.Sequential(
            nn.Linear(int(48 * max_pooling_len), EMBEDDING_LEN),
        )

    def forward_one_side(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        embedding1 = self.forward_one_side(input1)
        embedding2 = self.forward_one_side(input2)
        maxout, minout = self.maxminout(embedding1, embedding2)
        return maxout, minout

    def align_sequences(self, seq1_embedding, seq2_embedding):
        maxout, minout = self.maxminout(seq1_embedding, seq2_embedding)
        return maxout, minout

    def get_embedding(self, sequence):
        seq_tensor = torch.zeros((1, 4, SEQ_LEN))

        for i, c in enumerate(sequence):
            seq_tensor[0, atcg_map.get(c, 0), i] = 1.0

        if CUDA_FLAG:
            seq_tensor = seq_tensor.cuda()

        seq_tensor = Variable(seq_tensor)
        embedding = self.forward_one_side(seq_tensor)

        return embedding


# %%
def weight_func(dist):
    return 1.0


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, maxout, minout, align_dist):
        weight = Variable(torch.FloatTensor([weight_func(x) for x in align_dist.data]), requires_grad=False)
        if CUDA_FLAG:
            weight = weight.cuda()
        loss_contrastive = torch.mean(torch.mul(weight, torch.pow(1 - minout.sum(1) / maxout.sum(1) - align_dist, 2)))

        return loss_contrastive


# %%
atcg_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


class SiameseNetworkDataset(Dataset):
    def __init__(self, data_fp, target_fp, N):
        self.data_fp = data_fp
        self.target_fp = target_fp
        self.N = N
        self.data_tensor = self.gen_data_tensor()
        self.target_tensor = self.gen_target_tensor()

    def __getitem__(self, index):
        return self.data_tensor[0][index], self.data_tensor[1][index], self.target_tensor[index]

    def __len__(self):
        return self.N

    def gen_data_tensor(self):
        seq1 = torch.zeros((self.N, 4, SEQ_LEN))
        seq2 = torch.zeros((self.N, 4, SEQ_LEN))
        cnt = 0
        with open(self.data_fp) as f:
            while True:
                next_n = list(islice(f, 4))
                if not next_n:
                    break
                if cnt >= self.N:
                    break
                read1 = next_n[1].strip()
                read2 = next_n[3].strip()
                for i, c in enumerate(read1):
                    seq1[cnt, atcg_map.get(c, 0), i] = 1.0
                for i, c in enumerate(read2):
                    seq2[cnt, atcg_map.get(c, 0), i] = 1.0
                cnt += 1
        return seq1, seq2

    def gen_target_tensor(self):
        target = torch.zeros(self.N)
        with open(self.target_fp) as f:
            for i, line in enumerate(f):
                if i >= self.N:
                    break
                pair_id, dist = line.strip().split()
                target[i] = float(dist)
        return target


# %%
net = SiameseNetwork()
if CUDA_FLAG:
    net.cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=Config.learning_rate, weight_decay=0)


training_dataset = SiameseNetworkDataset(Config.train_data_fp,
                                         Config.train_target_fp,
                                         Config.train_num_example)

training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=Config.train_batch_size,
    shuffle=True,
    num_workers=0,  # 4
)
# %%
train_num_batch = int(np.ceil(Config.train_num_example / Config.train_batch_size))
train_batch_interval = Config.train_num_example // Config.train_batch_size // 10
train_loss_hist = []

for epoch in range(Config.num_epoch):
    print('===========>')
    train_running_loss = 0
    for batch_index, (train_seq1, train_seq2, train_target) in enumerate(training_loader):
        if CUDA_FLAG:
            train_seq1 = train_seq1.cuda()
            train_seq2 = train_seq2.cuda()
            train_target = train_target.cuda()
        train_seq1 = Variable(train_seq1)
        train_seq2 = Variable(train_seq2)
        train_target = Variable(train_target).float()
        train_output1, train_output2 = net(train_seq1, train_seq2)
        train_loss_contrastive = criterion(train_output1, train_output2, train_target)
        # train_running_loss += train_loss_contrastive.data[0]
        train_running_loss += train_loss_contrastive.item()

        if batch_index % train_batch_interval == train_batch_interval - 1:
            print('Epoch: {:d}/{:d}, Batch: {:d}/{:d}\n'
                  'Accumulated loss: {:.4e}'.format(
                epoch + 1, Config.num_epoch,
                batch_index + 1, train_num_batch,
                train_running_loss / (batch_index + 1)))

        optimizer.zero_grad()
        train_loss_contrastive.backward()
        optimizer.step()
    train_loss = train_running_loss / train_num_batch
    train_loss_hist.append(train_loss)
    print('Train loss: {:.4e}'.format(train_loss))


def jaccard_dist(embedding1, embedding2):
    return 1 - np.sum(np.minimum(embedding1, embedding2)) / np.sum(np.maximum(embedding1, embedding2))

def pair_dist(fp, N, embedding_fp, dist_fp):
    seq = torch.zeros((N, 4, SEQ_LEN))
    cnt = 0
    seq_ids = []
    with open(fp) as f:
        while True:
            next_n = list(islice(f, 2))
            if not next_n:
                break
            seq_id = next_n[0].strip()[1:]
            read = next_n[1].strip()
            seq_ids.append(seq_id)
            for i, c in enumerate(read):
                seq[cnt, atcg_map.get(c, 0), i] = 1.0
            cnt += 1
    embeddings = net.forward_one_side(Variable(seq)).data.numpy()
    embeddings.tofile(embedding_fp, sep=',', format='%.4e')
    with open(dist_fp, 'w') as fo:
        for i in range(N):
            for j in range(N):
                if i < j:
                    fo.write('{}-{}\t{:.4f}\n'.format(
                        seq_ids[i], seq_ids[j],
                        jaccard_dist(embeddings[i],
                                     embeddings[j])))

def generate_individual_randomly(all_sequences):
    if len(all_sequences) < 2:
        raise ValueError("Not enough sequences to generate an individual")

    # Randomly select two distinct sequences
    chosen_indices = random.sample(range(len(all_sequences)), 2)
    seq1, seq2 = all_sequences[chosen_indices[0]], all_sequences[chosen_indices[1]]

    # Remove the chosen sequences by their unique identifiers
    all_sequences[:] = [seq for i, seq in enumerate(all_sequences) if i not in chosen_indices]

    return [seq1, seq2]


def generate_population(all_sequences, desired_population_size):
    if len(all_sequences) < 4:
        raise ValueError("Not enough sequences to generate a population with each having three unique pairings.")

    # Mescola gli indici delle sequenze per selezione casuale
    shuffled_indices = list(range(len(all_sequences)))
    random.shuffle(shuffled_indices)

    population = []
    # Inizializza un dizionario per tenere traccia delle coppie per ogni sequenza
    pairings_per_sequence = {index: [] for index in range(len(all_sequences))}

    # Itera su ogni sequenza per formare tre coppie uniche
    for seq_index in shuffled_indices:
        available_indices = [i for i in shuffled_indices if i != seq_index and i not in pairings_per_sequence[seq_index]]
        # Seleziona fino a tre indici unici per formare le coppie
        for _ in range(3):
            if available_indices:
                chosen_index = random.choice(available_indices)
                # Aggiungi la coppia alla popolazione se non supera il limite desiderato
                if len(population) < desired_population_size:
                    population.append([all_sequences[seq_index], all_sequences[chosen_index]])
                    # Aggiorna le coppie formate per entrambe le sequenze
                    pairings_per_sequence[seq_index].append(chosen_index)
                    pairings_per_sequence[chosen_index].append(seq_index)
                    # Rimuovi l'indice scelto dagli indici disponibili per evitare ripetizioni
                    available_indices.remove(chosen_index)

    return population

def count_occurrences(sequence):
    index_counts = {}

    for index in sequence:
        if index in index_counts:
            index_counts[index] += 1
        else:
            index_counts[index] = 1

    max_occurrences = {}
    for index, count in index_counts.items():
        if index not in max_occurrences or count > max_occurrences[index]:
            max_occurrences[index] = count

    return max_occurrences

def initialize_origin_arrays(individual):
    seq1, seq2 = individual
    origin_array1 = [seq1.id] * len(seq1.seq)
    origin_array2 = [seq2.id] * len(seq2.seq)
    return origin_array1, origin_array2

with open('log_file.txt', 'w') as log_file:
    def crossover_subsequence(population, origin_arrays):
        new_population = []  # Creare una nuova lista per gli individui modificati
        print(len(population))
        for i in range(2):
            indici_usati = list(range(len(population)))
            while len(indici_usati) >= 2:
                # Seleziona due individui casuali dalla popolazione
                ind1_idx = random.choice(indici_usati)
                print("indice 1 del crossover", file=log_file)
                print(ind1_idx, file=log_file)
                indici_usati.remove(ind1_idx)
                ind2_idx = random.choice(indici_usati)
                print("indice 2 del crossover", file=log_file)
                print(ind2_idx, file=log_file)
                indici_usati.remove(ind2_idx)
                print("popolazione è lunga", file=log_file)
                print(len(population), file=log_file)
                seq1_1, seq1_2 = population[ind1_idx]
                seq2_1, seq2_2 = population[ind2_idx]
                print("origin array è lung0", file=log_file)
                print(len(origin_arrays), file=log_file)
                origin_array1_1, origin_array1_2 = origin_arrays[ind1_idx]
                origin_array2_1, origin_array2_2 = origin_arrays[ind2_idx]
                print(origin_array1_1, file=log_file)
                print(origin_array1_2, file=log_file)
                print(origin_array2_1, file=log_file)
                print(origin_array2_2, file=log_file)

                # Applica la tua logica di crossover ai due individui selezionati
                start_position1 = random.randint(0, len(seq1_1.seq))
                #print("posizione di start1")
                #print(start_position1)
                # Imposta subset_size1 dalla posizione di start fino alla fine della sequenza
                subset_size1 = len(seq1_1.seq) - start_position1
                #print(subset_size1)


                subset_seq1_1 = seq1_1.seq[start_position1:start_position1 + subset_size1]
                subset_seq1_2 = seq1_2.seq[start_position1:start_position1 + subset_size1]

                subset_seq2_1 = seq2_1.seq[start_position1:start_position1 + subset_size1]
                subset_seq2_2 = seq2_2.seq[start_position1:start_position1 + subset_size1]
                # Trasforma le sequenze in oggetti SeqRecord con le sequenze modificate
                seq1_1_mutated = SeqRecord(Seq(str(seq1_1.seq[:start_position1]) + str(subset_seq2_1) + str(seq1_1.seq[start_position1 + subset_size1:])))
                seq1_2_mutated = SeqRecord(Seq(str(seq1_2.seq[:start_position1]) + str(subset_seq2_2) + str(seq1_2.seq[start_position1 + subset_size1:])))
                #print("sequenza 1 mutata")
                new_origin_array1_1 = origin_array1_1[:start_position1] + origin_array2_1[start_position1:start_position1 + subset_size1] + origin_array1_1[start_position1 + subset_size1:]
                new_origin_array1_2 = origin_array1_2[:start_position1] + origin_array2_2[start_position1:start_position1 + subset_size1] + origin_array1_2[start_position1 + subset_size1:]
                #print(seq1_mutated.seq)
                seq2_1_mutated = SeqRecord(Seq(str(seq2_1.seq[:start_position1]) + str(subset_seq1_1) + str(seq2_1.seq[start_position1 + subset_size1:])))
                seq2_2_mutated = SeqRecord(Seq(str(seq2_2.seq[:start_position1]) + str(subset_seq1_2) + str(seq2_2.seq[start_position1 + subset_size1:])))
                #print("sequenza 2 mutata")
                new_origin_array2_1 = origin_array2_1[:start_position1] + origin_array1_1[start_position1:start_position1 + subset_size1] + origin_array2_1[start_position1 + subset_size1:]
                new_origin_array2_2 = origin_array2_2[:start_position1] + origin_array1_2[start_position1:start_position1 + subset_size1] + origin_array2_2[start_position1 + subset_size1:]
                print("\n-------------------------------------")
                print("nuovo array 1_1", file=log_file)
                print(new_origin_array1_1, file=log_file)
                print("\n-------------------------------------")
                print("nuovo array 2_1", file=log_file)
                print(new_origin_array2_1, file=log_file)
                print("\n-------------------------------------")
                print("nuovo array 1_2", file=log_file)
                print(new_origin_array1_2, file=log_file)
                print("\n-------------------------------------")
                print("nuovo array 2_2", file=log_file)
                print(new_origin_array2_2, file=log_file)
                print("\n-------------------------------------")

                # Aggiungi gli individui mutati alla nuova popolazione
                nuovasequenza1 = (seq1_1_mutated, seq1_2_mutated)
                nuovasequenza2 = (seq2_1_mutated, seq2_2_mutated)

                new_population.append(nuovasequenza1)
                new_population.append(nuovasequenza2)

                origin_arrays[ind1_idx] = new_origin_array1_1, new_origin_array1_2
                origin_arrays[ind2_idx] = new_origin_array2_1, new_origin_array2_2

        print("\n FINE CROSSOVER -------------------------------------", file=log_file)
        print(len(new_population), file=log_file)
        return new_population, origin_arrays

    def mutation_subsequence(population, origin_arrays):
        new_population = population  # Creare una nuova lista per gli individui modificati
        new_origin_array = origin_arrays
        num_to_mutate = len(population)

        indici = list(range(len(population)))

        mutated_count = 0  # Contatore per gli individui mutati
        print(len(indici))
        while mutated_count < num_to_mutate:
            # Seleziona due individui casuali dalla popolazione
            ind1_idx = random.choice(indici)
            indici.remove(ind1_idx)
            ind2_idx = random.choice(indici)
            indici.remove(ind2_idx)
            print("indici usati rimanenti", file=log_file)
            print(len(indici), file=log_file)
            print("individui mutati", file=log_file)
            print(mutated_count, file=log_file)

            print("indice 1", file=log_file)
            print(ind1_idx, file=log_file)
            print("indice 2", file=log_file)
            print(ind2_idx, file=log_file)
            seq1_1, seq1_2 = population[ind1_idx]
            seq2_1, seq2_2 = population[ind2_idx]
            origin_array1_1, origin_array1_2 = origin_arrays[ind1_idx]
            print("array originale 1_1", file=log_file)
            print(origin_array1_1, file=log_file)
            print("array originale 1_2", file=log_file)
            print(origin_array1_2, file=log_file)
            origin_array2_1, origin_array2_2 = origin_arrays[ind2_idx]
            print("array originale 2_1", file=log_file)
            print(origin_array2_1, file=log_file)
            print("array originale 2_2", file=log_file)
            print(origin_array2_2, file=log_file)

            # Applica la tua logica di crossover ai due individui selezionati
            start_position1 = random.randint(0, len(seq1_1.seq)-1)
            #print("posizione di start1")
            #print(start_position1)
            min_size = 150
            if SEQ_LEN - start_position1 < 100:
                min_size = SEQ_LEN - start_position1

            # subset_size1 = random.randint(min_size, max_size)
            subset_size1 = min_size
            # Imposta subset_size1 dalla posizione di start fino alla fine della sequenza
            #subset_size1 = random.randint(1, (len(seq1_1.seq) - start_position1))
            #print(subset_size1)

            numero_casuale = random.randint(0, 1)
            print("è stato generato il numero:")
            print(numero_casuale)
            if (numero_casuale == 0):
                subset_seq1_1 = seq1_1.seq[start_position1:start_position1 + subset_size1]
                subset_seq2_1 = seq2_1.seq[start_position1:start_position1 + subset_size1]
                seq1_1_mutated = SeqRecord(Seq(str(seq1_1.seq[:start_position1]) + str(subset_seq2_1) + str(seq1_1.seq[start_position1 + subset_size1:])))
                new_origin_array1_1 = origin_array1_1[:start_position1] + origin_array2_1[start_position1:start_position1 + subset_size1] + origin_array1_1[start_position1 + subset_size1:]
                seq2_1_mutated = SeqRecord(Seq(str(seq2_1.seq[:start_position1]) + str(subset_seq1_1) + str(seq2_1.seq[start_position1 + subset_size1:])))
                new_origin_array2_1 = origin_array2_1[:start_position1] + origin_array1_1[start_position1:start_position1 + subset_size1] + origin_array2_1[start_position1 + subset_size1:]
                print("nuovo array 1_1", file=log_file)
                print(new_origin_array1_1, file=log_file)
                print("nuovo array 2_1", file=log_file)
                print(new_origin_array2_1, file=log_file)
                nuovasequenza1 = (seq1_1_mutated, seq1_2)
                nuovasequenza2 = (seq2_1_mutated, seq2_2)

                new_population[ind1_idx] = nuovasequenza1
                new_population[ind2_idx] = nuovasequenza2
                new_array1 = (new_origin_array1_1, origin_array1_2)
                new_array2 = (new_origin_array2_1, origin_array2_2)
                origin_arrays[ind1_idx] = (new_array1)
                origin_arrays[ind2_idx] = (new_array2)
                """
                new_population.append(nuovasequenza1)
                new_population.append(nuovasequenza2)
                new_array1 = (new_origin_array1_1, origin_array1_2)
                new_array2 = (new_origin_array2_1, origin_array2_2)
                new_origin_array.append(new_array1)
                new_origin_array.append(new_array2)
                """
                mutated_count += 2  # Aggiorna il contatore per gli individui mutati
            else:
                subset_seq1_2 = seq1_2.seq[start_position1:start_position1 + subset_size1]
                subset_seq2_2 = seq2_2.seq[start_position1:start_position1 + subset_size1]
                # Trasforma le sequenze in oggetti SeqRecord con le sequenze modificate
                seq1_2_mutated = SeqRecord(Seq(str(seq1_2.seq[:start_position1]) + str(subset_seq2_2) + str(seq1_2.seq[start_position1 + subset_size1:])))
                #print("sequenza 1 mutata")
                new_origin_array1_2 = origin_array1_2[:start_position1] + origin_array2_2[start_position1:start_position1 + subset_size1] + origin_array1_2[start_position1 + subset_size1:]
                #print(seq1_mutated.seq)
                seq2_2_mutated = SeqRecord(Seq(str(seq2_2.seq[:start_position1]) + str(subset_seq1_2) + str(seq2_2.seq[start_position1 + subset_size1:])))
                #print("sequenza 2 mutata")
                new_origin_array2_2 = origin_array2_2[:start_position1] + origin_array1_2[start_position1:start_position1 + subset_size1] + origin_array2_2[start_position1 + subset_size1:]
                print("nuovo array 1_2", file=log_file)
                print(new_origin_array1_2, file=log_file)
                print("nuovo array 2_2", file=log_file)
                print(new_origin_array2_2, file=log_file)
                nuovasequenza1 = (seq1_1, seq1_2_mutated)
                nuovasequenza2 = (seq1_2, seq2_2_mutated)

                new_population[ind1_idx] = nuovasequenza1
                new_population[ind2_idx] = nuovasequenza2
                new_array1 = (origin_array1_1, new_origin_array1_2)
                origin_arrays[ind1_idx] = (new_array1)
                new_array2 = (origin_array2_1, new_origin_array2_2)
                origin_arrays[ind2_idx] = (new_array2)
                """
                new_population.append(nuovasequenza1)
                new_population.append(nuovasequenza2)
                new_array1 = (origin_array1_1, new_origin_array1_2)
                new_origin_array.append(new_array1)
                new_array2 = (origin_array2_1, new_origin_array2_2)
                new_origin_array.append(new_array2)
                """
                mutated_count += 2  # Aggiorna il contatore per gli individui mutati

            print("\n FINE MUTAZIONE -------------------------------------", file=log_file)
        print(len(new_population), file=log_file)
        return new_population, new_origin_array

    csv_data = []

    def confronta_sequenze(seq1, seq2):
        lunghezza_minima = min(len(seq1), len(seq2))
        caratteri_comuni = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
        percentuale_somiglianza = (caratteri_comuni / lunghezza_minima) * 100
        return percentuale_somiglianza

    def evaluate(individual, net, origin_array):
        """
        Valuta la fitness media di un individuo basata sulle sottoporzioni identificate
        in origin_array.

        :param individual: Una tupla contenente le due sequenze dell'individuo.
        :param net: La rete neurale Siamese addestrata.
        :param origin_array: L'array che traccia l'origine delle sottoporzioni della sequenza.
        :return: La fitness media dell'individuo.
        """
        seq1, seq2 = individual
        weighted_fitness = 0
        total_length = 0

        current_origin = origin_array[0]
        start_index = 0
        for i, origin in enumerate(origin_array + ['']):  # Aggiunge un elemento per forzare l'ultima valutazione
            if origin != current_origin or i == len(origin_array):
                segment_length = i - start_index
                total_length += segment_length

                # Estrai la sottosequenza
                subseq1 = seq1.seq[start_index:i]
                subseq2 = seq2.seq[start_index:i]

                # Calcola la fitness della sottosequenza
                seq1_embedding = net.get_embedding(subseq1)
                seq2_embedding = net.get_embedding(subseq2)
                maxout, minout = net.align_sequences(seq1_embedding, seq2_embedding)
                distance = 1 - (minout.sum(1) / maxout.sum(
                    1)).item()  # Ad esempio, utilizza la distanza di Jaccard inversa come fitness

                # Applica il peso basato sulla lunghezza della sottosequenza
                weighted_fitness += distance * segment_length

                start_index = i
                current_origin = origin

        # Calcola la fitness media pesata
        average_weighted_fitness = weighted_fitness / total_length if total_length > 0 else 0

        return average_weighted_fitness

    def update_sequence_id(sequence, new_id):
        sequence.id = str(new_id)

    def generate_random_dna_sequence(length):
        bases = ['A', 'T', 'C', 'G']
        return ''.join(random.choice(bases) for _ in range(length))

    def save_fasta_file(filename, sequences):
        with open(filename, 'w') as file:
            for i, sequence in enumerate(sequences, start=1):
                file.write(f">Sequence{i}\n")
                file.write(sequence + '\n')

    num_sequences = 10
    sequence_length = 150
    input_file = [generate_random_dna_sequence(sequence_length) for _ in range(num_sequences)]
    save_fasta_file('random_sequences.fasta', input_file)
    print(f"{num_sequences} random DNA sequences of length {sequence_length} saved to 'random_sequences.fasta'.", file=log_file)
    # Leggi sequenze genomiche da un file FASTA
    #input_file = './demo/pair_shuffle.fa'
    input_file = 'random_sequences.fasta'
    all_sequences = list(SeqIO.parse(input_file, "fasta"))
    for i, seq in enumerate(all_sequences):
            update_sequence_id(seq, i)
            print(i)

    start_bruto = time.time()
    # Imposta il numero di generazioni e la dimensione della popolazione
    generations = 1
    population_size = 20
    population = generate_population(all_sequences, population_size)
    pop2 = population
    pop2 = [sequence for pair in pop2 for sequence in pair] #coppie in lista
    """
    new_id = 0
    for individual in population:
        for seq in individual:
            seq.id = str(new_id)
            new_id += 1
    """
    id_to_index_map = {seq.id: i for i, pair in enumerate(population) for seq in pair}
    origin_arrays = [initialize_origin_arrays(individual) for individual in population]
    csv_data2 = []
    popolazione_brutta = all_sequences.copy()

    print("Sequenze della popolazione:", file=log_file)
    for seq in popolazione_brutta:
        print(seq.seq, file=log_file)

    # Effettua l'allineamento per ogni sequenza rispetto a tutte le altre sequenze
    #for i in range(len(sequences)):
    for i in range(10):
        start_time = time.time()
        seq1_embedding = net.get_embedding(popolazione_brutta[i])

        # Inizializza le variabili per il punteggio migliore e la sequenza corrispondente
        best_score = float('inf')  # Inizializzato a infinito in modo che qualsiasi punteggio lo superi
        best_sequence = ""
        best_sequence_index = -1

        for j in range(10):
            if i != j:  # Evita di confrontare una sequenza con se stessa
                seq2_embedding = net.get_embedding(popolazione_brutta[j])

                maxout, minout = net.align_sequences(seq1_embedding, seq2_embedding)

                # Calcola la distanza di allineamento
                alignment_distance = (minout.sum(1) / maxout.sum(1)).detach().cpu().numpy()[0]

                # Aggiorna il punteggio migliore e la sequenza corrispondente se necessario
                if alignment_distance < best_score:
                    best_score = alignment_distance
                    best_sequence = popolazione_brutta[j].seq
                    best_sequence_index = j + 1  # Indice + 1 per ottenere il numero di sequenza

        # Calcola la percentuale del numero di sequenza rispetto al totale
        percentage = (best_sequence_index / len(popolazione_brutta)) * 100

        # Stampa il punteggio migliore, la sequenza corrispondente e l'allineamento per la sequenza corrente
        print(f"Per la sequenza {i + 1}:", file=log_file)
        print(f"Miglior punteggio di allineamento: {best_score}", file=log_file)
        print(f"Sequenza corrispondente: {best_sequence}", file=log_file)
        print(f"Numero di sequenza corrispondente: {best_sequence_index} su {len(popolazione_brutta)} (percentuale: {percentage:.2f}%)", file=log_file)
        csv_data2.append([i+1, best_sequence_index, best_sequence])
        end_time = time.time()
        tempo_effettivo = end_time - start_time
        #print(tempo_effettivo)

    print(f"Verifichiamo gli accoppiamenti creati:", file=log_file)
    # Stampa di ciascun accoppiamento di sequenze e dei corrispondenti origin_arrays
    for i, (pair, origin_array) in enumerate(zip(population, origin_arrays), start=1):
        seq1, seq2 = pair
        origin_array1, origin_array2 = origin_array
        print(f"Accoppiamento {i}:", file=log_file)
        print(f"Sequenza 1: {seq1.id}, Sequenza 2: {seq2.id}", file=log_file)
        print(f"Sequenza 1: {seq1.seq}, Sequenza 2: {seq2.seq}", file=log_file)
        print(f"Origin Array 1: {origin_array1}, Origin Array 2: {origin_array2}", file=log_file)
        print("-" * 20, file=log_file)

    start_time = time.time()
    current_sequence_index = 0
    # Esegui l'algoritmo genetico
    for run in range(1000):
        print(f"\nRun {run + 1}:", file=log_file)
        print(f"\nRun {run + 1}:")
        # Crea una popolazione iniziale
        print(f"lunghezza della popolazione : {len(population)}", file=log_file)
        """
        print("Sequenze della popolazione:", file=log_file)
        for seq in pop2:
            print(seq.seq, file=log_file)
        """
        # Riassegnazione degli ID
        for gen in range(generations):
            # Crea nuovi individui attraverso crossover e mutazione
            # numero1 = random.randint(0, 1)
            numero1 = 1
            # numero2 = random.randint(0, 1)
            numero2 = 1
            if numero1 == numero2:
                print("STA AVVENENDO LA MUTAZIONE", file=log_file)
                population, origin_arrays = mutation_subsequence(population, origin_arrays)
                fitness_values = [evaluate(individual, net, origin_arrays) for individual in population]
                for i, (fitness, (seq1, seq2)) in enumerate(zip(fitness_values, population)):
                    print(f"Per l'accoppiamento {i + 1} composto dalla sequenza {seq1.seq} e dalla sequenza {seq2.seq} la fitness è: {fitness}", file=log_file)
            else:
                print("nessuna mutazione", file=log_file)
            print(f"lunghezza popolazione dopo mutazione: {len(population)}", file=log_file)
            print("ESEGUO IL CROSSOVER", file=log_file)
            population, origin_arrays = crossover_subsequence(population, origin_arrays)
            print(f"lunghezza popolazione dopo crossover: {len(population)}", file=log_file)
            # Valuta la fitness degli individui
            fitness_values = [evaluate(individual, net, origin_arrays) for individual in population]
            # Stampare la fitness per ogni individuo
            print("Fitness degli individui dopo la mutazione e il crossover:", file=log_file)
            # Stampare la fitness per ogni individuo
            for i, (fitness, (seq1, seq2)) in enumerate(zip(fitness_values, population)):
                print(f"Per l'accoppiamento {i + 1} composto dalla sequenza {seq1.seq} e dalla sequenza {seq2.seq} la fitness è: {fitness}", file=log_file)

            best_individuals_data = sorted(zip(population, origin_arrays, fitness_values), key=lambda x: x[2], reverse=True)[:population_size]
            best_individuals = [ind for ind, _, _ in best_individuals_data]
            best_origin_arrays = [origin for _, origin, _ in best_individuals_data]
            population = best_individuals
            origin_arrays = best_origin_arrays
            pop2 = population
            pop2 = [sequence for pair in pop2 for sequence in pair]  # coppie in lista
            print("Sequenze della popolazione con la migliore fitness:", file=log_file)
            # Modifica qui: Itera su best_individuals_data invece che su pop2
            for i, (individual, _, fitness) in enumerate(best_individuals_data):
                seq1, seq2 = individual
                # Ora hai accesso sia alle sequenze sia alla loro fitness associata
                print(f"Sequenza {i * 2 + 1}: {seq1.seq}, Fitness: {fitness}", file=log_file)
                print(f"Sequenza {i * 2 + 2}: {seq2.seq}, Fitness: {fitness}", file=log_file)

        print("\n-------------------------------------", file=log_file)
        # Converti la popolazione bidimensionale in una lista unidimensionale
        j = 1
        for i in range(len(origin_arrays)):
            #print(f"Array numero {i + 1}:")
            array1, array2 = origin_arrays[i]

            max_occurrences1 = count_occurrences(array1)
            max_occurrences2 = count_occurrences(array2)

            max_index1 = max(max_occurrences1, key=max_occurrences1.get)
            max_index2 = max(max_occurrences2, key=max_occurrences2.get)

            total_occurrences1 = sum(max_occurrences1.values())
            total_occurrences2 = sum(max_occurrences2.values())

            max_value1 = max_occurrences1[max_index1]
            max_value2 = max_occurrences2[max_index2]

            percentage1 = (max_value1 / total_occurrences1) * 100
            percentage2 = (max_value2 / total_occurrences2) * 100

            max_index1_mapped = id_to_index_map[str(max_index1)]
            max_index2_mapped = id_to_index_map[str(max_index2)]
            j+=2
            #print("\n------------------")


        # Scrivi i risultati in un file csv
        with open('genetico.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['index', 'best_sequence_index', 'best_sequence'])
            writer.writerows(csv_data)

        # Calcola il tempo di esecuzione
        end_time = time.time()
        execution_time = end_time - start_time
        # Scrivi i risultati in un file csv
        with open('bruteforce.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['index', 'best_sequence_index', 'best_sequence'])
            writer.writerows(csv_data2)

        tempo_finale = time.time() - start_bruto
        matches_per_pair2 = {}  # Dizionario per tenere traccia delle corrispondenze per ogni coppia
        for result in csv_data2:
            sequence_index2, best_sequence_index2, _2 = result
            sequence_index_str2 = str(sequence_index2 - 1)  # -1 per correggere l'indice
            best_sequence_index_str2 = str(best_sequence_index2 - 1)
            pair_key2 = f"{sequence_index2}-{best_sequence_index2}"  # Chiave unica per la coppia
            matches_per_pair2[pair_key2] = {'count': 0, 'arrays': set()}  # Inizializza il contatore e l'insieme degli array

            for idx, origin_array in enumerate(origin_arrays):
                origin_array1, origin_array2 = origin_array
                array_key = f"Array {idx + 1}"

                for origin1, origin2 in zip(origin_array1, origin_array2):
                    if sequence_index_str2 == origin1 and best_sequence_index_str2 == origin2:
                        matches_per_pair2[pair_key2]['count'] += 1
                        matches_per_pair2[pair_key2]['arrays'].add(array_key)
        print(f"Sequenze individuate durante la generazione {run + 1}", file=log_file)
        # Stampa il numero di corrispondenze per ogni coppia e in quanti array diversi sono presenti
        for pair, info in matches_per_pair2.items():
            if info['count'] > 0:
                print(f"Per l'allineamento {pair} sono state trovate {info['count']} corrispondenze in {len(info['arrays'])} array diversi.", file=log_file)
                percentuale_corrispondenza = (info['count'] / SEQ_LEN) * 100
                percentuale_array = (len(info['arrays']) / population_size) * 100
                print(f"Percentuale di corrispondenza: {percentuale_corrispondenza:.2f}%. Percentuale di array: {percentuale_array:.2f}%.", file=log_file)

        print("\n------------------", file=log_file)
        print(f"Tempo di esecuzione totale dell'algoritmo brute force: {tempo_finale} secondi", file=log_file)
        print("\n------------------", file=log_file)

        print("\n------------------", file=log_file)
        print(f"Tempo di esecuzione totale dell'algoritmo genetico: {execution_time} secondi", file=log_file)
        print("\n------------------", file=log_file)


    # Stampa il numero di corrispondenze per ogni coppia e in quanti array diversi sono presenti
    for pair, info in matches_per_pair2.items():
        print(f"Per l'allineamento {pair} sono state trovate {info['count']} corrispondenze in {len(info['arrays'])} array diversi.", file=log_file)
        percentuale_corrispondenza = (info['count'] / SEQ_LEN) * 100
        percentuale_array = (len(info['arrays']) / population_size) * 100
        print(f"Percentuale di corrispondenza: {percentuale_corrispondenza:.2f}%. Percentuale di array: {percentuale_array:.2f}%.", file=log_file)