import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Split:
    def __init__(self, args) -> None:
        self.args = args

    def exec(self):
        print("Starting the data splitting process")
        print("Loading file: ", self.args.data_dir)
        print("Saving to: ", self.args.outputs_dir)

        if not os.path.exists(self.args.outputs_dir):
            os.makedirs(self.args.outputs_dir)

        # load the data and separate labels
        data = pd.read_excel(self.args.data_dir, index_col=0)

        label = data["Label"]
        data = data.drop("Label", axis=1)

        # perform stratified data split
        data_train, data_test, label_train, label_test = train_test_split(
            data,
            label,
            test_size=self.args.datasplit_ratio,
            random_state=self.args.random_seed,
            stratify=label,
        )

        print(
            "Datasplit: total: {}, train {}, test {}".format(
                data.shape[0], data_train.shape[0], data_test.shape[0]
            )
        )

        data_formatted_train = pd.concat([data_train, label_train], axis=1)
        data_formatted_train.to_excel(
            os.path.join(
                self.args.outputs_dir, self.args.filename_prefix + "data_train.xlsx"
            )
        )

        data_formatted_test = pd.concat([data_test, label_test], axis=1)
        data_formatted_test.to_excel(
            os.path.join(
                self.args.outputs_dir, self.args.filename_prefix + "data_test.xlsx"
            )
        )
