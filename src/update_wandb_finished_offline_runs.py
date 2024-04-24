from typing import Union
import wandb, pprint
api = wandb.Api()

run_id = ""

run = api.run(f"marcelloceresini/Thesis/{run_id}")
summary = {}
i=0
for row in run.scan_history():
    # print(i)
    # i+=1
    summary.update(row)
    # for k,v in row.items():
    #     if isinstance(v, Union[None|str|float|int]):
    #         summary.update({k: v})

pprint.pprint(summary)
run.summary.update(summary)
print("Updated summary to: ", summary)