import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm


def trainModel(model, train_ds, test_ds, epochs=5):
    

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # check to ensure correct loss function is used
    optimizer=tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



    # train the model
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = loss_object(y_train, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(y_train, predictions)

        for x_test, y_test in test_ds:
            predictions = model(x_test, training=False)
            t_loss = loss_object(y_test, predictions)

            test_loss(t_loss)
            test_accuracy(y_test, predictions)


        template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                    train_loss.result(),
                    train_accuracy.result() * 100,
                    test_loss.result(),
                    test_accuracy.result() * 100))
        
    return model

class Distiller:
    def __init__(self, teacher, student, x_train, y_train, x_test, y_test, x_transfer=None, y_transfer=None, temp=4, epochs=5, task_balance=0.8, early_stopping=False, batch_size=32):
        self.teacher = teacher
        self.student = student
        self.student_base = tf.keras.models.clone_model(student)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        if x_transfer is None or y_transfer is None:
            x_transfer = x_train
            y_transfer = y_train
        self.x_transfer = x_transfer
        self.y_transfer = y_transfer
        self.train_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_train.shape[0]).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        self.transfer_ds = tf.data.Dataset.from_tensor_slices((x_transfer, y_transfer)).shuffle(x_transfer.shape[0]).batch(batch_size)
        self.temp = temp
        self.epochs = epochs
        self.task_balance = task_balance
        self.early_stopping = early_stopping  # to do
        self.batch_size = batch_size
        self.history = {}
        self.history['teacher'] = {}
        self.history['student'] = {}
        self.history['models'] = {}
        
    '''
    Train a teacher network with the specified temperature.
    Store the soft targets for the training set
    '''
    def trainTeacher(self):
        print('training teacher network...')
        # self.teacher = trainModel(self.teacher, self.train_ds, self.test_ds, epochs=epochs)
        self.history['teacher']['train_accuracy'] = []
        self.history['teacher']['train_loss'] = []
        self.history['teacher']['test_accuracy'] = []
        self.history['teacher']['test_loss'] = []

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # check to ensure correct loss function is used
        optimizer=tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



        # train the model
        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()


            for x_train, y_train in self.train_ds:
                with tf.GradientTape() as tape:
                    predictions = self.teacher(x_train, training=True)
                    loss = loss_object(y_train, predictions)
                
                gradients = tape.gradient(loss, self.teacher.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.teacher.trainable_variables))

                train_loss(loss)
                train_accuracy(y_train, predictions)

            for x_test, y_test in self.test_ds:
                predictions = self.teacher(x_test, training=False)
                t_loss = loss_object(y_test, predictions)

                test_loss(t_loss)
                test_accuracy(y_test, predictions)
            
            self.history['teacher']['train_loss'].append(train_loss.result())
            self.history['teacher']['train_accuracy'].append(train_accuracy.result())
            self.history['teacher']['test_loss'].append(test_loss.result())
            self.history['teacher']['test_accuracy'].append(test_accuracy.result())

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss.result()}, Train Accuracy: {train_accuracy.result()*100}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()*100}')

    
    
    '''
    Get the soft targets from the teacher network's logits
    on a transfer set
    '''
    def getTeacherTargets(self):
        print('\nfetching soft targets set by teacher network...')
        # for the training set, get the soft targets set by the teacher network
        teacher_targets = []
        
        for sample in tqdm(self.x_transfer):
            teacher_targets.append(tf.nn.softmax(self.teacher.predict(np.expand_dims(sample, axis=0)/self.temp)))
            
        self.transfer_ds_soft = tf.data.Dataset.from_tensor_slices((self.x_transfer, teacher_targets)).batch(32)      
        
    
    
    '''
    Train the student network to match the soft targets
    set by the teacher network and the hard targets
    from the dataset
    '''
    def teachStudent(self):
        print('\ntraining student model to match soft targets set by teacher and hard targets from dataset...')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer=tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.history['student']['train_accuracy'] = []
        self.history['student']['train_loss'] = []
        self.history['student']['test_accuracy'] = []
        self.history['student']['test_loss'] = []
        
        
        print('training the student network...')
        for epoch in range(self.epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
        
            for (x_train, y_train_hard), (_, y_train_soft) in zip(self.transfer_ds, self.transfer_ds_soft):
                
                with tf.GradientTape() as tape:
                    predictions = self.student(x_train, training=True)
                    hard_target_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_train_hard, tf.int32), logits=predictions)
                    soft_target_xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_train_soft, logits=predictions)
        
                    soft_target_xent *= self.temp**2
                    total_loss = self.task_balance*hard_target_xent
                    total_loss += (1-self.task_balance)*soft_target_xent
                
                gradients = tape.gradient(total_loss, self.student.trainable_variables)

                optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
                
                train_loss(total_loss)
                train_accuracy(y_train_hard, predictions)
                
            for x_test, y_test in self.test_ds:
                predictions = self.student(x_test, training=False)
                loss = loss_object(y_test, predictions)

                test_loss(loss)
                test_accuracy(y_test, predictions)


            self.history['student']['train_loss'].append(train_loss.result())
            self.history['student']['train_accuracy'].append(train_accuracy.result())
            self.history['student']['test_loss'].append(test_loss.result())
            self.history['student']['test_accuracy'].append(test_accuracy.result())

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss.result()}, Train Accuracy: {train_accuracy.result()*100}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()*100}')
            

    def run(self):
        self.trainTeacher() 
        self.getTeacherTargets()
        self.teachStudent() 
        self.history['models']['teacher'] = self.teacher
        self.history['models']['student'] = self.student

        return self.history
        


def teacherModel():
    teacher = models.Sequential()
    teacher = models.Sequential()
    teacher.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    teacher.add(layers.MaxPooling2D((2, 2)))
    teacher.add(layers.Conv2D(32, (3, 3), activation='relu'))
    teacher.add(layers.MaxPooling2D((2, 2)))
    teacher.add(layers.Flatten())
    teacher.add(layers.Dense(56, activation='relu'))
    # teacher.add(layers.Dropout(rate=0.2))
    teacher.add(layers.Dense(56, activation='relu'))
    # teacher.add(layers.Dropout(rate=0.2))
    teacher.add(layers.Dense(56, activation='relu'))
    # teacher.add(layers.Dropout(rate=0.2))
    teacher.add(layers.Dense(56, activation='relu'))
    teacher.add(layers.Dense(10, name='teacher_logits'))
    
    # teacher.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    return teacher

def studentModel():
    student = models.Sequential()
    student.add(layers.Flatten(input_shape=(28, 28, 1)))
    student.add(layers.Dense(10, activation='relu'))
    student.add(layers.Dense(10, name='student_logits'))
    
    # student.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    return student

if __name__ == '__main__':
    
    # a small demo

    tf.keras.backend.clear_session()
    
    
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[:10000, :, :]
    y_train = y_train[:10000]
    x_test = x_test[:3000, :, :]
    y_test = y_test[:3000]

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    EPOCHS = 10
    teacher = teacherModel()

    dis = Distiller(teacher, tf.keras.models.clone_model(teacher), x_train, y_train, x_test, y_test, temp=4, epochs=EPOCHS, task_balance=0.8)

    history = dis.run()

    fig, ax = plt.subplots()
    ax.plot(np.arange(EPOCHS), history['student']['train_accuracy'], linewidth=2, label='student training accuracy', color='darkorange', linestyle='--')
    ax.plot(np.arange(EPOCHS), history['student']['test_accuracy'], linewidth=2, label='student test accuracy', color='darkorange')
    ax.plot(np.arange(EPOCHS), history['teacher']['train_accuracy'], linewidth=2, label='teacher training accuracy', color='navy', linestyle='--')
    ax.plot(np.arange(EPOCHS), history['teacher']['test_accuracy'], linewidth=2, label='teacher test accuracy', color='navy')

    ax.legend(loc='best')

    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_title('Learning Curves')
    plt.show()
    


