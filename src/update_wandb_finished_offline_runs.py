from typing import Union
import wandb, pprint
api = wandb.Api()

good_runs = [
    #"568h6oc2", "qmzxcn8q", "pe5g6emv", 
    "hm4nkv8d", 
    # "zcwgrsl9", 
    # "fn0w9rb0", 
    #"ehu8ya0l", "xlqu35is", "p1nbcy4p", "28ig2vzi", "iez5i4fg", "kylpiqqd"
]

UPDATE_TEST_VALUES = True
UPDATE_METRIC_MIN = True

def update_summary(run_id):
    run = api.run(f"marcelloceresini/Thesis/{run_id}")
    tmp_summary = {}
    i=0
    metric_list = []

    for row in run.scan_history():
        for k,v in row.items():
            if "test" in k and UPDATE_TEST_VALUES:
                if v is not None:
                    tmp_summary.update({k: v})
            if "metric" == k and UPDATE_METRIC_MIN:
                if v is not None:
                    metric_list.append(v)

    if UPDATE_METRIC_MIN:
        tmp_summary["metric"] = {"min":min(metric_list)}
    # print(list(run.summary._dict.keys()))
    run.summary.update(tmp_summary)


if __name__ == "__main__":
    for run_id in good_runs:
        print(run_id)
        update_summary(run_id)