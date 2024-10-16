def scale_data(self, scaling, train_data, train_labels):
    if scaling == 'standard':
        self.train_data[[col for col in self.train_data]] = StandardScaler().fit_transform(self.train_data[[col for col in self.train_data]], self.test_data)
    elif scaling == 'min_max':
        min_max_scaler = MinMaxScaler()
        self.train_data[[col for col in self.train_data]] = min_max_scaler.fit_transform(self.train_data[[col for col in self.train_data]], self.test_data)
    else:
        print("Entered scaling arg not supported")
    return

def preprocess_ids_data():
    ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Test_data.csv'

    ids_train_data = pd.read_csv(ids_train_file)

    label_encoder = LabelEncoder()

    ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

    ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

    ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

    ids_train_data = ids_train_data.sample(frac=1)

    train_set, test_set = train_test_split(ids_train_data, test_size = .2)

    train_set = train_set.reset_index(drop=True)

    test_set = test_set.reset_index(drop=True)

    train_data = train_set.loc[:, train_set.columns != 'class']

    train_labels = train_set['class']

    test_data = test_set.loc[:, test_set.columns != 'class']

    test_labels = test_set['class']

    return train_data, train_labels, test_data, test_labels

