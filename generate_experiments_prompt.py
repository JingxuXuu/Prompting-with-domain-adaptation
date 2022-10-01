data = ["MNLI"]

few_shot="False"
unlabelledlst = ["False"]
freezelst = ["False"]
trgdic = {"Rumor":["sydneysiege","ferguson","ottawashooting","germanwings-crash","charliehebdo"],
          "MNLI":["travel","slate","telephone","fiction","government"],
          "Review":["books","kitchen","dvd","electronics"]}

program_name = 'training_for_three.py'
with open('experiment_list.txt','w') as f:
    for data_chosen in data:
        for freeze in freezelst:
            for unlabelled in unlabelledlst:
                for seed in range(16,21):
                    for trg in trgdic[data_chosen]:
                        lst = trgdic[data_chosen].copy()
                        lst.remove(trg)
                        srg = ",".join(lst)
                        f.write(f'python3 {program_name} --param0={unlabelled} --param1={srg} --param2={trg} --param3={data_chosen} --param4={freeze} --param5={seed} --param6={few_shot}\n')