import tensorflow as tf
from data_utils import *
import sys,time
from os import path
# import edit_distance as ed

class CNN_BGRU(tf.keras.Model):
    def __init__(self,config):
        super(CNN_BGRU,self).__init__()
        self.config = config
        self.img_h = int(self.config['scale_h'])
        nb_filters = [int(n) for n in self.config['filters']]
        kernel_ws = [int(n) for n in self.config['kernel_width']]
        # first create observation CNN with filter height = max height
        self.glimpse_cnn_layer = tf.keras.layers.Conv2D(nb_filters[0],[self.img_h,kernel_ws[0]],[self.img_h,1],padding='same',data_format='channels_last')
        self.all_cnn = []
        for f in range(1,len(nb_filters)): # first one used in glimpse
            kernel_shape = [1,kernel_ws[f]]
            cnn_layer = tf.keras.layers.Conv2D(nb_filters[f],kernel_shape,strides=1,padding='same',data_format='channels_last')
            pool_layer = tf.keras.layers.MaxPool2D(kernel_shape,strides=1,padding='same',data_format='channels_last')
            norm_layer = tf.keras.layers.LayerNormalization()
            drop_layer = tf.keras.layers.Dropout(float(self.config['dropout']))
            self.all_cnn.extend([cnn_layer,pool_layer,norm_layer,drop_layer])
        print('CNN layers created')
        # create rnn layers
        self.all_rnn = []
        for l in range(int(self.config['rnn_layers'])):
            a_layer = tf.keras.layers.GRU(int(self.config['rnn_dim']),return_state=False,return_sequences=True)
            a_layer = tf.keras.layers.Bidirectional(a_layer)
            self.all_rnn.append(a_layer)
        # create output layer
        self.output_layer = tf.keras.layers.Dense(int(self.config['N_out']))

    def call(self, inputs,input_lens, training=None, mask=None):
        glimpse_out = self.glimpse_cnn_layer(inputs)
        layer_out = glimpse_out
        for a_layer in self.all_cnn:
            layer_out = a_layer(layer_out)
        # add residual connection
        cnn_out = tf.concat([layer_out,glimpse_out],axis=-1)
        # squeeze for RNN
        layer_out = tf.squeeze(cnn_out,axis=1)
        for a_layer in self.all_rnn:
            layer_out = a_layer(layer_out)
        rnn_mask = tf.expand_dims(tf.cast(tf.sequence_mask(input_lens),tf.float32),-1)
        layer_out = layer_out * rnn_mask
        output = self.output_layer(layer_out)
        output_t = tf.transpose(output,[1,0,2]) # time axis comes first
        return output_t

class HandWritingRecognizer:
    def __init__(self,config_file,savepath="Weights",logpath='Logs'):
        self.config = get_congig(config_file)
        self.model = CNN_BGRU(self.config)
        self.optimizer = tf.keras.optimizers.Adam(float(self.config['learning_rate']))
        self.savepath = savepath
        self.logpath = logpath
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.weight_manager = tf.train.CheckpointManager(self.checkpoint,self.savepath,max_to_keep=2)

    def train_step(self,bx,bx_lens,by,by_lens):
        with tf.GradientTape() as tape:
            bx = tf.expand_dims(bx, -1)
            logit = self.model(bx,bx_lens)
            loss = tf.reduce_mean(tf.nn.ctc_loss(by,logit,by_lens,bx_lens,logits_time_major=True))
            weights = self.model.trainable_variables
            grads = tape.gradient(loss,weights)
        self.optimizer.apply_gradients(zip(grads,weights))
        prediction,_ = tf.nn.ctc_greedy_decoder(logit,bx_lens)
        prediction = tf.sparse.to_dense(prediction[0])
        return loss,prediction

    def test_step(self,bx,bx_lens,by,by_lens):
        bx = tf.expand_dims(bx, -1)
        logit = self.model(bx, bx_lens)
        loss = tf.reduce_mean(tf.nn.ctc_loss(by, logit, by_lens, bx_lens, logits_time_major=True))
        return loss

    def train(self,img_pkl,train_file,test_file,tok_file):
        restore_from = tf.train.latest_checkpoint(self.savepath)
        self.checkpoint.restore(restore_from)
        print('Model restored from ',restore_from)
        batch_size = int(self.config['batch_size'])
        epochs = int(self.config['epochs'])
        pkl_img, total = initiate_batch_generation(img_pkl, train_file)

        batches = batch_generator(train_file, batch_size, pkl_img, tok_file)
        nb_batches = int(np.ceil(total / batch_size))
        # load validation batches
        val_pkl_img, val_total = initiate_batch_generation(img_pkl, test_file)
        val_batches = batch_generator(test_file, batch_size, val_pkl_img, tok_file)
        val_nb_batches = int(np.ceil(val_total / batch_size))
        if(path.exists(self.logpath+"/training_log.txt")==False):
          f = open(self.logpath+"/training_log.txt",'w')
          f.write('Training HW Recognizer\n')
          f.write("Training with %d data Testing with %d data\n"%(total,val_total))
          f.close()
        v_loss = 0
        for e in range(epochs):
            e_loss = 0
            start_time = time.time()
            for b in range(nb_batches):
                bx, bx_lens, by, by_lens, start, end = next(batches)
                b_loss,b_pred = self.train_step(bx,bx_lens,by,by_lens)
                sys.stdout.write('\rBatch %d/%d from %d to %d loss %0.3f'%(b,nb_batches,start,end,b_loss))
                sys.stdout.flush()
                e_loss += b_loss
            e_loss /= nb_batches
            self.weight_manager.save()
            end_time = time.time()
            duration = end_time - start_time
            f = open(self.logpath+"/training_log.txt",'a')
            
            
            v_loss = 0
            for b in range(val_nb_batches):
                bx, bx_lens, by, by_lens, start, end = next(val_batches)
                v_b_loss = self.test_step(bx, bx_lens, by, by_lens)
                sys.stdout.write('\rValidation Batch %d/%d from %d to %d loss %0.3f' % (b, nb_batches, start, end, v_b_loss))
                sys.stdout.flush()
                v_loss += v_b_loss
            v_loss /= val_nb_batches
            f.write("%d\t%0.3f\t%0.3f\t%f\n"%(e,e_loss,v_loss,duration))
            print("\tEpoch %d/%d Loss %0.3f validation loss %.3f Time %0.2f" % (e, epochs, e_loss,v_loss,duration))
        f.close()

    def load_model(self,savepath=None):
        if savepath is not None:
            restore_from = tf.train.latest_checkpoint(self.savepath)
        else:
            restore_from = tf.train.latest_checkpoint(self.savepath)
        self.checkpoint.restore(restore_from)
        print('Model restored from ', restore_from)

    def predict(self,bx,bx_len,ind2tok):
        bx = bx/255.0 # normalize
        bx = tf.expand_dims(bx, 0) # give a batch dim
        bx = tf.expand_dims(bx,-1) # give a channel dim
        logit = self.model(bx,bx_len) # time major
        output = tf.nn.ctc_greedy_decoder(logit,[bx_len])
        dense_output = tf.sparse.to_dense(output[0][0])
        dense_output = dense_output.numpy()
        tok_output = []
        for ind in dense_output[0]:
            tok_output.append(ind2tok[ind])
        return dense_output[0],tok_output


