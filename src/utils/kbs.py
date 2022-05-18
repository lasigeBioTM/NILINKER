import csv
import logging
import obonet
#sys.path.append("./")


class KnowledgeBase:
    """Class representing a knowledge base that is loaded from local file.
    """

    global name_to_id, id_to_name, synonym_to_id, child_to_parent, umls_to_hp
    
    name_to_id = dict()
    id_to_name = dict()
    synonym_to_id = dict()
    child_to_parent = dict()
    umls_to_hp = dict()

    def __init__(self, kb):
        self.kb = kb
        self.root_dict = {"go_bp": ("GO:0008150", "biological_process"), 
                          "chebi": ("CHEBI:00", "root"), 
                          "hp": ("HP:0000001", "All"), 
                          "medic": ("MESH:C", "Diseases"), 
                          "ctd_anat": ("MESH:A", "Anatomy"), 
                          "ctd_chem": ("MESH:D", "Chemicals")}

    def load_obo(self, kb):
        """Loads KBs from local .obo files (ChEBI, HPO, MEDIC, GO) into 
        structured dicts containing the mappings name_to_id, id_to_name, 
        synonym_to_id, child_to_parent, umls_to_hp and the list of edges 
        between concepts. For 'chebi', only the concepts in the subset 3_STAR 
        are included, which correpond to manually validated entries. 
        
        :param kb: target ontology to load, has value 'medic', 'chebi', 
            'go_bp' or 'hp'
        :type kb: str
        """

        logging.info('-----> Loading {}'.format(kb))
        
        filepaths = {'medic': 'CTD_diseases', 'chebi': 'chebi', 
                     'go_bp': 'go-basic', 'hp': 'hp'}
        
        filepath = './data/kb_files/' + filepaths[kb] + '.obo'    
       
        graph = obonet.read_obo(filepath)
        edges = list()
        
        for node in  graph.nodes(data=True):
            add_node = False
            
            if "name" in node[1].keys():
                node_id, node_name = node[0], node[1]["name"]
                
                if kb == "go_bp": 
                    # For go_bp, ensure that only Biological Process 
                    # concepts are considered
                    
                    if node[1]['namespace'] == 'biological_process':
                        name_to_id[node_name] = node_id
                        id_to_name[node_id] = node_name
                        add_node = True

                elif kb == "medic":

                    if node_id[0:4] != "OMIM": 
                        # Exclude OMIM concepts 
                        name_to_id[node_name] = node_id
                        id_to_name[node_id] = node_name
                        add_node = True
                    
                else:
                    name_to_id[node_name] = node_id
                    id_to_name[node_id] = node_name
                    add_node = True

                if 'is_obsolete' in node[1].keys() and \
                        node[1]['is_obsolete'] == True:
                    add_node = False
                    del name_to_id[node_name]
                    del id_to_name[node_id] 

                if 'is_a' in node[1].keys() and add_node: 
                    # The root node of the ontology does not 
                    # have is_a relationships

                    if len(node[1]['is_a']) == 1: 
                        # Only consider concepts with 1 direct ancestor
                        child_to_parent[node_id] = node[1]['is_a'][0]

                    for parent in node[1]['is_a']: 
                        # To build the edges list, consider all 
                        # concepts with at least one ancestor
                        edges.append((node_id,parent))

                if "synonym" in node[1].keys() and add_node: 
                    # Check for synonyms for node (if they exist)
                        
                    for synonym in node[1]["synonym"]:
                        synonym_name = synonym.split("\"")[1]
                        synonym_to_id[synonym_name] = node_id

                if "xref" in node[1].keys() and add_node:

                    if kb == "hp": 
                        # Map UMLS concepts to HPO concepts
                        umls_xrefs = list()

                        for xref in node[1]['xref']:
                            if xref[:4] == "UMLS":
                                umls_id = xref.strip("UMLS:")
                                umls_to_hp[umls_id] =  node_id 
                     
        root_concept_name = self.root_dict[kb][1]
        root_id = str()
       
        if root_concept_name not in name_to_id.keys():
            root_id = self.root_dict[kb][0]
            name_to_id[root_concept_name] = root_id
            id_to_name[root_id] = root_concept_name

        if kb == 'chebi':
            # Add edges between the ontology root and sub-ontology roots
            chemical_entity = "CHEBI:24431"
            edges.append((chemical_entity, root_id))
            role = "CHEBI:50906"
            edges.append((role, root_id))
            subatomic_particle = "CHEBI:36342"
            edges.append((subatomic_particle, root_id))
            application = "CHEBI:33232"
            edges.append((application, root_id))


        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.synonym_to_id = synonym_to_id
        self.child_to_parent = child_to_parent
        self.umls_to_hp = umls_to_hp
        self.edges = edges
        
        logging.info('-----> {} loaded'.format(kb))

    def load_tsv(self, kb):
        """Loads KBs from local .tsv files (CTD-Chemicals, CTD-Anatomy) 
           into structured dicts containing the mappings name_to_id, 
           id_to_name, synonym_to_id, child_to_parent, umls_to_hp
           and the list of edges between concepts.
        
        :param kb: target ontology to load, has value 'ctd_chem' or 'ctd_anat'
        :type kb: str
        """
        
        logging.info('-----> Loading {}'.format(kb))
        
        kb_dict = {"ctd_chem": "CTD_chemicals", 
                   "ctd_anat": "CTD_anatomy"}
        filepath = "./data/kb_files/" + kb_dict[kb] + ".tsv"
        edges = list()

        with open(filepath) as kb_file:
            reader = csv.reader(kb_file, delimiter="\t")
            row_count = int()
        
            for row in reader:
                row_count += 1
                
                if row_count >= 30:
                    node_name = row[0] 
                    node_id = row[1]
                    node_parents = row[4].split('|')
                    synonyms = row[7].split('|')
                    name_to_id[node_name] = node_id
                    id_to_name[node_id] = node_name
                    
                    if len(node_parents) == 1: #
                        # Only consider concepts with 1 direct ancestor
                        child_to_parent[node_id] = node_parents[0]
                    
                    for synonym in synonyms:
                        synonym_to_id[synonym] = node_id

                    for parent in node_parents: 
                        # To build the edges list, consider 
                        # all concepts with at least one ancestor
                        edges.append((node_id,parent))
        
        root_concept_name = self.root_dict[kb][1]
        root_concept_id = self.root_dict[kb][0]
        name_to_id[root_concept_name] = root_concept_id
        id_to_name[root_concept_id] = root_concept_name

        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.synonym_to_id = synonym_to_id
        self.child_to_parent = child_to_parent
        self.edges = edges
        
        logging.info('-----> {} loaded'.format(kb))
    
    def load_chebi(self):
        "Load ChEBI ontology, but only the manually validated concepts"

        kb_dir = './data/kb_files/chebi/'

        name_to_id = dict()
        id_to_name = dict()
        synonym_to_id = dict()
        
        edges = list()
        ancestors_count = dict()
        terms_to_include = list()
        child_to_parent = dict()

        # Import relations
        print('relations')
        with open(kb_dir + 'relation_3star.tsv') as relations_file:
            reader = csv.reader(relations_file, delimiter='\t')
            
            for row in reader:
                
                if row[1] == 'is_a':
                    term1 = 'CHEBI:' + row[2]
                    term2 = 'CHEBI:' + row[3]
                    
                    edges.append((term1, term2))
    
                    if term1 not in terms_to_include:
                        terms_to_include.append(term1)

                    if term2 not in terms_to_include:
                        terms_to_include.append(term2)

                    if term1 in ancestors_count.keys():
                        added = ancestors_count[term1]
                        added += 1
                        ancestors_count[term1] = added
                    
                    else:
                        ancestors_count[term1] = 1

            relations_file.close()
        
        # Import term names
        print('compounds')
        with open(kb_dir + 'compounds_3star.tsv') as names_file:
            reader = csv.reader(names_file, delimiter='\t')

            for row in reader:
                
                chebi_id = row[2]
                name = row[5]

                if chebi_id in terms_to_include and name != 'null':
                    name_to_id[name] = chebi_id
                    id_to_name[chebi_id] = name

            names_file.close()

        # Import synonyms
        with open(kb_dir + 'names_3star.tsv') as syn_file:
            reader = csv.reader(syn_file, delimiter='\t')

            for row in reader:
                chebi_id = 'CHEBI:' + row[1]
                syn_name = row[4]

                if chebi_id in terms_to_include:
                    synonym_to_id[syn_name] = chebi_id
                    
            syn_file.close()
    
        # Add edges between the ontology root and sub-ontology roots
        root_concept_name = self.root_dict['chebi'][1]
        root_id = self.root_dict['chebi'][0]
        name_to_id[root_concept_name] = root_id
        id_to_name[root_id] = root_concept_name
        
        chemical_entity = "CHEBI:24431"
        edges.append((chemical_entity, root_id))
        role = "CHEBI:50906"
        edges.append((role, root_id))
        subatomic_particle = "CHEBI:36342"
        edges.append((subatomic_particle, root_id))
        application = "CHEBI:33232"
        edges.append((application, root_id))

        # Find child-parent links
        for edge in edges:
            child = edge[0]
            parent = edge[1]

            if child in ancestors_count.keys() \
                    and ancestors_count[child] == 1:
                # The term has only 1 direct ancestor
                child_to_parent[child] = parent

        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.synonym_to_id = synonym_to_id
        self.child_to_parent = child_to_parent
        self.umls_to_hp = umls_to_hp
        self.edges = edges
        
        logging.info('-----> {} loaded'.format('chebi'))