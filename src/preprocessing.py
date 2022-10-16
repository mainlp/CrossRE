import json
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from itertools import permutations

load_dotenv()

class DatasetMapper(Dataset):

    def __init__(self, sentences, entities_1, entities_2, relations):
        self.sentences = sentences
        self.entities_1 = entities_1
        self.entities_2 = entities_2
        self.relations = relations

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.entities_1[idx], self.entities_2[idx], self.relations[idx]


def prepare_data(data_path, labels2id, batch_size):

    sentences, entities_1, entities_2, relations = read_json_file(data_path, labels2id)
    data_loader = DataLoader(DatasetMapper(sentences, entities_1, entities_2, relations), batch_size=batch_size)
    return data_loader


# return sentences, idx within the sentence of entity-markers-start, relation labels
def read_json_file(json_file, labels2id, multi_label=False):

    sentences, entities_1, entities_2, relations = [], [], [], []

    with open(json_file) as data_file:
        for json_elem in data_file:
            document = json.loads(json_elem)

            # consider only the sentences with at least 2 entities
            if len(document["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(document["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1:{entity_pair[0][2]}>'
                    ent1_end = f'</E1:{entity_pair[0][2]}>'
                    ent2_start = f'<E2:{entity_pair[1][2]}>'
                    ent2_end = f'</E2:{entity_pair[1][2]}>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(document["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {document["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {document["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {document["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {document["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{document["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{document["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{document["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{document["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{document["sentence"][idx_token]} '

                    # retrieve relation label
                    dataset_relations = [(e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) for (e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) in document["relations"] if e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s == entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # prepare data
                    if len(dataset_relations) > 0:
                        if multi_label:
                            instance_labels = [0] * len(labels2id.keys())
                            for elem in dataset_relations:
                                instance_labels[labels2id[elem[4]]] = 1
                            relations.append(instance_labels)
                        else:
                            relations.append(labels2id[dataset_relations[0][4]])
                        sentences.append(sentence_marked.strip())
                        entities_1.append(sentence_marked.split(' ').index(f'{ent1_start}'))
                        entities_2.append(sentence_marked.split(' ').index(f'{ent2_start}'))

    return sentences, entities_1, entities_2, relations
