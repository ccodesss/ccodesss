class GAN():
    def __init__(self):
        self.urlfeature_dims = 60
        self.z_dims = 20   # could try 20,malware has to add url calls if you want to defraud the classifier
                           # Z the larger the dry so
        self.hide_layers = 120
        self.generator_layers = [self.urlfeature_dims + self.z_dims, self.hide_layers, self.urlfeature_dims]
        self.substitute_detector_layers = [self.urlfeature_dims, self.hide_layers, 1]
        self.blackbox = 'MLP'
        optimizer = Adam(lr=0.001)

        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.urlfeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator
        self.substitute_detector.trainable = False

        # The discriminator takes generated URLs as input and determines validity
        validity = self.substitute_detector(malware_examples)

        # The combined model  (stacked generator and substitute_detector)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_blackbox_detector(self):

        if self.blackbox == 'MLP':
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                                              solver='sgd', verbose=0, tol=1e-4, random_state=1,
                                              learning_rate_init=.1)
        return blackbox_detector

    def build_generator(self):

        example = Input(shape=(self.urlfeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        i =0
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='relu')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator

    def build_substitute_detector(self):

        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='relu')(x)
        substitute_detector = Model(input, x, name='substitute_detector')
        substitute_detector.summary()
        return substitute_detector

    def load_data(self, filename):
        
        df_xmal = df_mal.iloc[:,:-1]
        df_ymal = df_mal.iloc[:,60:61]
        df_xben = df_leg.iloc[:,:-1]
        df_yben = df_leg.iloc[:,60:61]
        
        xmal = df_xmal.to_numpy()
        ymal = df_ymal.to_numpy()
        xben = df_xben.to_numpy()
        yben = df_yben.to_numpy()
            
        return (xmal, ymal), (xben, yben)
    
    def train(self, epochs, batch_size):

        # Load the dataset
        (xmal, ymal), (xben, yben) = self.load_data('')
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.30)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.30)

        # Train blackbox_detctor
        self.blackbox_detector.fit(np.concatenate([xmal, xben]),
                                   np.concatenate([ymal, yben]))

        ytrain_ben_blackbox = self.blackbox_detector.predict(xtrain_ben)
        Original_Train_TRR = self.blackbox_detector.score(xtrain_mal, ytrain_mal)
        Original_Test_TRR = self.blackbox_detector.score(xtest_mal, ytest_mal)
        Train_TRR, Test_TRR = [], []