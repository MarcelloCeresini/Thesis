from typing import Union
import wandb, pprint
api = wandb.Api()

good_runs = [
    # "2ygcpwue", # lemon-grass-38
    # "pnyash2s", # magic-oath-39
    # "ekffajd8", # desert-disco-41
    # "ywoed45q", # cool-pine-45
    # "0az598nq", # deep-capybara-49
    # "1fagc6rx", # brisk-galaxy-50
    # "htb6vv54", # silver-flower-51
    # "kylpiqqd", # apricot-deluge-52
    # "p68fr2sa", # worldy-music-55
    # "i6a0k4ca", # misty-feather-62
    # "0egkpy0z", # olive-star-63
    # "iugv186w", #pious-plant-64
    # "peh0csfv", # lucky-snow-65
    # "qqxymndp", # icy-firefly-66
    # "8tfcp71w", # misty-bird-73
    
]

UPDATE_TEST_VALUES = True
UPDATE_METRIC_MIN = False

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