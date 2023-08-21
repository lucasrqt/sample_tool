#!/usr/bin/python3

import pandas as pd

from app_parser import App, Parser
from configs_parser import *


def save_database(data: list):
    full_database = "../results/all_failures_database.csv"
    pretty_names = {
        "vit32s_224": ("BS32-224", "Original ViT"),
        "vit16_224": ("B16-224", "Original ViT"),
        "vit16_384": ("B16-384", "Original ViT"),
        "vit14l_224": ("L14-224", "Original ViT"),
        "vit14h": ("H14-224", "Original ViT"),
        "eva_base": ("B14-448", "EVA"),
        "eva_large": ("L14-448", "EVA"),
        "swin_b12to16": ("B256", "Swin"),
        "swin_b12to14": ("B384", "Swin"),
        "swin_l12to16": ("L256", "Swin"),
        "maxvit_l384": ("L384", "MaxViT"),
        "maxvit_l512": ("L512", "MaxViT"),
    }
    pretty_names.update(**{f"{name}-hd": new_name for name, new_name in pretty_names.items()})
    pretty_names["vit32_sam_224"] = pretty_names["vit32s_224"]
    df = pd.DataFrame(data)
    df["config"] = df["model"].apply(lambda x: pretty_names[x][0])
    df["family"] = df["model"].apply(lambda x: pretty_names[x][1])
    df["hardening"] = df["model"].apply(lambda x: "Hardened" if "-hd" in x else "Unhardened")
    df.to_csv(full_database, index=False)
    print(f"Saving full database at:{full_database}")


def main():
    # parsser init
    parser = Parser()

    # sample tool app init
    inj_per_fm = 250
    injections = {
        G_GP: [FLIP_SINGLE_BIT, RANDOM_VALUE],
        G_LD: [FLIP_SINGLE_BIT, RANDOM_VALUE],
        G_FP32: [FLIP_SINGLE_BIT, RANDOM_VALUE, WARP_RANDOM_VALUE],
    }

    # swin_b1216_n = "swin_b12to16-hd"
    # swin_b1216_path = f"{BASE_DIR}/pascal/{swin_b1216_n}"
    # swin_b1216 = App(swin_b1216_n, swin_b1216_path, inj_per_fm, injections)

    # swin_b1224_n = "swin_b12to14-hd"
    # swin_b1224_path = f"{BASE_DIR}/pascal/{swin_b1224_n}"
    # swin_b1224 = App(swin_b1224_n, swin_b1224_path, inj_per_fm, injections)

    # swin_l1216_n = "swin_l12to16-hd"
    # swin_l1216_path = f"{BASE_DIR}/pascal/{swin_l1216_n}"
    # swin_l1216 = App(swin_l1216_n, swin_l1216_path, inj_per_fm, injections)

    # eva_b_n = "eva_base-hd"
    # eva_b_path = f"{BASE_DIR}/pascal/{eva_b_n}"
    # eva_b = App(eva_b_n, eva_b_path, inj_per_fm, injections)

    # eva_l_n = "eva_large-hd"
    # eva_l_path = f"{BASE_DIR}/pascal/{eva_l_n}"
    # eva_l = App(eva_l_n, eva_l_path, inj_per_fm, injections)

    maxvit_384_n = "maxvit_l384-hd"
    maxvit_384_path = f"{BASE_DIR}/pascal/{maxvit_384_n}"
    maxvit_384 = App(maxvit_384_n, maxvit_384_path, inj_per_fm, injections)

    maxvit_512_n = "maxvit_l512-hd"
    maxvit_512_path = f"{BASE_DIR}/pascal/{maxvit_512_n}"
    maxvit_512 = App(maxvit_512_n, maxvit_512_path, inj_per_fm, injections)

    vit14h_n = "vit14h-hd"
    vit14h_path = f"{BASE_DIR}/pascal/{vit14h_n}"
    vit14h = App(vit14h_n, vit14h_path, inj_per_fm, injections)

    apps = [
        # swin_b1216,
        # swin_b1224,
        # swin_l1216,
        # eva_b,
        # eva_l,
        maxvit_384,
        maxvit_512,
        vit14h,
    ]
    apps = list()
    for model_name in ["vit16_224", "vit16_384", "vit32s_224", "vit14l_224", "vit14h", "vit16_224-hd", "vit16_384-hd",
                       "vit32s_224-hd", "vit14l_224-hd", "vit14h-hd", "eva_base", "eva_large", "eva_base-hd",
                       "eva_large-hd", "swin_b12to16", "swin_b12to14", "swin_l12to16", "swin_b12to16-hd",
                       "swin_b12to14-hd", "swin_l12to16-hd", "maxvit_l384", "maxvit_l512", "maxvit_l384-hd",
                       "maxvit_l512-hd", ]:
        for board in ["pascal", "ampere"]:
            if board == "ampere" and model_name == "vit32s_224":
                model_name = model_name.replace("vit32s_224", "vit32_sam_224")
            apps.append(App(model_name, f"{BASE_DIR}/{board}/{model_name}", inj_per_fm, injections))

    errs_by_model = {}
    full_database = list()
    for app in apps:
        board = "Ampere" if "ampere" in app.app_folder else "Pascal"
        print(f" [+] parsing results per category")
        res = parser.parse_per_cat(app=app, calculate_percentage=False)
        print(res)
        errs_by_model[app.app_name] = {"SDC": 0.0, "Critical SDC": 0.0, "DUE": 0.0}
        bfms_cnt = 0
        for group in res:
            for bfm in res[group]:
                bfms_cnt += 1
                errs_by_model[app.app_name]["SDC"] += res[group][bfm]["SDC"]
                errs_by_model[app.app_name]["Critical SDC"] += res[group][bfm][
                    "Critical SDC"
                ]
                errs_by_model[app.app_name]["DUE"] += res[group][bfm]["DUE"]
                full_database.append({
                    "board": board,
                    "model": app.app_name, "bfm": bfm, "group": group, "SDC": res[group][bfm]["SDC"],
                    "critical_SDC": res[group][bfm]["Critical SDC"], "DUE": res[group][bfm]["DUE"]
                })
        errs_by_model[app.app_name]["SDC"] /= bfms_cnt
        errs_by_model[app.app_name]["Critical SDC"] /= bfms_cnt
        errs_by_model[app.app_name]["DUE"] /= bfms_cnt

        print(f" [+] parsing results per bfm")
        res_stdout, res_stderr = parser.parse_per_bfm(app)

        print(f" [+] parsing results for kernel")
        res_kernels = parser.parse_per_kernel(app)

        # print(f" [+] deep parsing results for kernel")
        # res_ker_bfm = parser.parse_per_kernel_bfm(sample_tool)

        df_res = parser.dict_to_dataframe(res)
        # df_stdout = parser.dict_to_dataframe(res_stdout)
        # df_stderr = parser.dict_to_dataframe(res_stderr)

        df_res.to_csv(f"{app.app_folder}/results_cat.csv")
        # df_stdout.to_csv(f"{app.app_folder}/results_stdout.csv")
        # df_stderr.to_csv(f"{app.app_folder}/results_stderr.csv")

        # df_kernels = pd.DataFrame.from_dict(res_kernels, orient="index")
        # df_kernels.to_csv(f"{app.app_folder}/results_kernel.csv")

        # df_kbfm = dict_to_dataframe(res_ker_bfm)
        # df_kbfm.to_csv(f"./{app.app_folder}/results_kernel_bfm.csv"
        output_pvf_path = f"{BASE_DIR}/pascal/pvf_by_models.csv"
        df_pvf_models = pd.DataFrame.from_dict(errs_by_model, orient="index")

        # issue on appending data to csv file
        # when check if file exists or not, it appends 
        # the same row multiple times
        # if not os.path.isfile(output_pvf_path):
        df_pvf_models.to_csv(output_pvf_path)
        # else:
        #    df_pvf_models.to_csv(output_pvf_path, mode='a', index=True, header=False)

        # for k in errs_by_model:
        #    sdc = errs_by_model[k]["SDC"]
        #    crit = errs_by_model[k]["Critical SDC"]
        #    due = errs_by_model[k]["DUE"]
        #    print(f"{k},{sdc},{crit},{due}")

    save_database(data=full_database)


if __name__ == "__main__":
    main()
