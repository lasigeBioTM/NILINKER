import os
import sys
sys.path.append("./")


def analyse_solution_in_candidates_list(dataset, link_mode, nil):
    """Analysis to study how many times NILINKER was able to retrieve the 
    correct candidate and include it in an entity's candidates list compared 
    with the none approach. Outputs file with statistics in 
    'analysis/solution_in_candidates_list"""
    
    candidates = dict()
    candidates_dir = "../../data/REEL/candidates/"
    candidates_dir += dataset + "/" + link_mode + "/" + nil + "/"

    docs = os.listdir(candidates_dir)
    solution_in_candidates = 0
    total_entities = 0
    link_count = 0
    
    for doc in docs:
       
        with open(candidates_dir + doc, "r") as input:
            data = input.readlines()
            input.close()
        
        candidates = list()
        entity_id = str()
        previous_line = str()
      
        for line in data:
          
            if line[:6] == "ENTITY":
                # This is a new entity. Add the info from previous one
                if entity_id in candidates:
                    solution_in_candidates += 1
         
                if entity_id == "-1" or entity_id == "NIL" \
                        or entity_id == "MESH_-1":

                    print(previous_line)
                print(candidates)
                
                # Reset since we are dealing with a new entity
                total_entities += 1
                entity_id = line.strip("\n").split("\t")[8].split("url:")[1]
                previous_line = line
                candidates = []

            else:
                # This is a candidate
                candidate_kb_id = line.strip("\n").split("\t")[5].split("url:")[1]
                candidates.append(candidate_kb_id)
                links = line.strip("\n").split("\t")[4].split("links:")[1].split(";")
                link_count += len(links)


    output = "Solution in candidates: " + str(solution_in_candidates) + "\n"
    output += "Total entities: " + str(total_entities) + "\n"
    output += "Solution is in list %: " + str(solution_in_candidates/total_entities) + "\n"
    output += "Number of total links between candidates: " + str(link_count) + "\n"
    print(output)
    
    #out_filename = "analysis/solution_in_candidates_list/{}_{}_{}".\
    #    format(dataset, link_mode, nil)

    #with open(out_filename, "w") as outfile:
    #    outfile.write(output)
    #    outfile.close()


if __name__ == "__main__":
    dataset = sys.argv[1]
    link_mode = sys.argv[2]
    nil = sys.argv[3]

    analyse_solution_in_candidates_list(dataset, link_mode, nil)
