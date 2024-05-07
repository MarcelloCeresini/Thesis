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
    "0egkpy0z", # olive-star-63
    "iugv186w", #pious-plant-64

]

def update_summary(run_id):
    run = api.run(f"marcelloceresini/Thesis/{run_id}")
    summary = {}
    i=0
    for row in run.scan_history():
        # print(i)
        # i+=1
        # summary.update(row)
        # summary.update({k:v for k, v in row.items() if v is not None})
        for k,v in row.items():
            if "test" in k:
                if v is not None:
                    summary.update({k: v})
        #     if isinstance(v, Union[str|float|int]):
        #         summary.update({k: v})
        #     if "metric" in k:
        #         pass
        #     if "min" in k:
        #         pass

    run.summary.update(summary)


if __name__ == "__main__":
    for run_id in good_runs:
        update_summary(run_id)