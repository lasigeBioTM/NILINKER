import networkx as nx
import utils as ut
import sys
import json


##############################################################################
# Calculate average number of ancestors per entity in given corpus
##############################################################################


dataset = sys.argv[1]
data_dir = "../../data/corpora/preprocessed/{}/".format(dataset)
root_id = ''

if dataset == "chr": 
    #data_dir += "chr/"
    #annotations = ut.parse_Pubtator(dataset, data_dir, entity_type=None)
    kb_graph, root_id= ut.load_obo("chebi", only_graph=True)

elif dataset == "ncbi_disease":
    #data_dir += "NCBI_disease_corpus/"
    #annotations = ut.parse_Pubtator(dataset, data_dir, entity_type=None)
    kb_graph, root_id = ut.load_obo("medic", only_graph=True)

elif dataset == "bc5cdr_medic_WORKS":
    #data_dir = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/"
    #annotations = ut.parse_Pubtator("bc5cdr", data_dir, entity_type="Disease")
    kb_graph, root_id = ut.load_obo("medic", only_graph=True)

elif dataset == "bc5cdr_chem":
    #data_dir = "BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/"
    #annotations = ut.parse_Pubtator("bc5cdr", data_dir, entity_type="Chemical")
    kb_graph, root_id = ut.load_ctd_chemicals(only_graph=True)

elif dataset == "gsc+":
    #annotations = ut.parse_GSC_corpus()
    kb_graph, root_id = ut.load_obo("hp", only_graph=True)

elif dataset == "phaedra":
    #annotations = ut.parse_phaedra_corpus()
    kb_graph, root_id = ut.load_ctd_chemicals(only_graph=True)


annotations = ut.import_input(data_dir + 'test.json')

ancestors_count = int()
ent_count = int()
missing_ids = int()
added = list()
annotation_count = 0
word_1 = 0
word_2 = 0

for doc in annotations.keys():
    doc_entities = annotations[doc]

    for entity in doc_entities:
        entity_text = entity[1]

        if entity_text not in added:
        
            # Get number of ancestors by calculating shortest path length
            # from the concept to the root
            if entity[0] in kb_graph.nodes():
                #ANALISE NUMBER OF ANCESTORS
                ent_count += 1
                ancestors = nx.shortest_path_length(kb_graph, 
                                                    source=entity[0], 
                                                        target=root_id)
            
                ancestors_count += ancestors
            
            else:
                missing_ids+= 1

ancestors_avg = ancestors_count / ent_count

print("The average number of ancestors per entity in the dataset {} is: \
    {}".format(dataset, str(ancestors_avg)))

print("missing ids: ", str(missing_ids))

