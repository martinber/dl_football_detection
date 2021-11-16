import json

def listing(folder, filt, verbose):

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

                    if key == "layers_config":
                        for layer in value:
                            print(layer)
                    else:
                        print(key, ":", value)

            else:
                print(data["id"])
