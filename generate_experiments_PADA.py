data = ["review"]

few_shot="False"
unlabelledlst = ["False"]
freezelst = ["False"]
trgdic = {"rumor":["sydneysiege","ferguson","ottawashooting","germanwings-crash","charliehebdo"],
          "mnli":["travel","slate","telephone","fiction","government"],
          "review":["books","kitchen","dvd","electronics"]}

program_name = 'train_PADA.py'
with open('experiment_list.txt','w') as f:
    for data_chosen in data:
        for trg in trgdic[data_chosen]:
            lst = trgdic[data_chosen].copy()
            lst.remove(trg)
            srg = ",".join(lst)
            f.write(f'python3 {program_name} --param0={data_chosen} --param1={srg} --param2={trg}\n')