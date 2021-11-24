import json

def listing(folder, filt, print_layers, verbose):

    for case_path in sorted(folder.iterdir()):

        if not (case_path / "saved_model.pb").is_file():
            # Training didn't finish in this case
            continue

        if not (case_path / "case.json").is_file():
            # Training didn't finish in this case
            continue

        with open(case_path / "case.json", "r") as f:
            data = json.load(f)

            if filt:
                if not str(data[filt[0]]) == filt[1]:
                    continue

            if verbose:
                print("------------------------")
                for key, value in data.items():

                    ignored = ["history", "model_summary"]
                    if key in ignored:

                        continue

                    elif key == "layers_config":

                        if print_layers:
                            for layer in value:
                                print(layer)
                        else:
                            print(key, "/ number of layers :", len(value))

                    elif key == "eval":

                        print(key, ":")
                        for metric, score in value.items():
                            print("   ", metric, ":", score)

                    else:

                        print(key, ":", value)

            else:
                print(data["id"])
