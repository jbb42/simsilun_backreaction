import configparser as cp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time
import psutil,os
import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from subprocess import check_output
from cnn import model
from data_machinary import *

import Pk_library as PKL
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from subprocess import check_output



config = cp.ConfigParser()
config.read("config.ini")

main_box = int(config.get("myvars", "main_box"))
sub_box = int(config.get("myvars", "sub_box"))
Number_of_sub_boxes = int(config.get("myvars", "Number_of_sub_boxes"))
ml_algorithm = config.get("myvars", "ml_algorithm")
boxsize = int(config.get("myvars", "boxsize"))

frac = main_box/sub_box   

if ml_algorithm == 'cnn' or ml_algorithm == 'rf' or ml_algorithm == 'both':



    path_LCDM_1 = config.get("myvars", "path_LCDM_1")
    path_LCDM_2 = config.get("myvars", "path_LCDM_2")
    path_LCDM_3 = config.get("myvars", "path_LCDM_3")
    path_LCDM_4 = config.get("myvars", "path_LCDM_4")
    path_LCDM_5 = config.get("myvars", "path_LCDM_5")
    path_LCDM_6 = config.get("myvars", "path_LCDM_6")

    path_k_1 = config.get("myvars", "path_k_1")
    path_k_2 = config.get("myvars", "path_k_2")
    path_k_3 = config.get("myvars", "path_k_3")
    path_k_4 = config.get("myvars", "path_k_4")
    path_k_5 = config.get("myvars", "path_k_5")
    path_k_6 = config.get("myvars", "path_k_6")



    # w = -1 and c^2 = 1

    delta1 = delta(path_LCDM_1,main_box) # seed = 42
    delta2 = delta(path_LCDM_2,main_box) # seed = 43
    delta3 = delta(path_LCDM_3,main_box) # seed = 44
    delta4 = delta(path_LCDM_4,main_box) # seed = 45
    delta5 = delta(path_LCDM_5,main_box) # seed = 46
    delta6 = delta(path_LCDM_6,main_box) # seed = 47


    # w = - 0.9 and c^2 = 1

    delta7 = delta(path_k_1,main_box) # seed = 42
    delta8 = delta(path_k_2,main_box) # seed = 43
    delta9 = delta(path_k_3,main_box) # seed = 44
    delta10 = delta(path_k_4,main_box) # seed = 45
    delta11 = delta(path_k_5,main_box) # seed = 46
    delta12 = delta(path_k_6,main_box) # seed = 47


    data_train = data_provider_train(Number_of_sub_boxes/5)
    X_train = np.concatenate([data_train[0],data_train[1]],axis=0)
    y_train = np.concatenate([generate_labels(Number_of_sub_boxes,0),generate_labels(Number_of_sub_boxes,1)],axis=0)
    print('X_train : ',X_train.shape)
    print('y_train : ',y_train.shape)

    data_val = data_provider_test(Number_of_sub_boxes/10)
    X_val = np.concatenate([data_val[0],data_val[1]],axis=0)
    y_val = np.concatenate([generate_labels(Number_of_sub_boxes/10,0),generate_labels(Number_of_sub_boxes/10,1)],axis=0)
    print('X_val : ',X_val.shape)
    print('y_val : ',y_val.shape)

    data_test = data_provider_test(Number_of_sub_boxes/10)
    X_test = np.concatenate([data_test[0],data_test[1]],axis=0)
    y_test = np.concatenate([generate_labels(Number_of_sub_boxes/10,0),generate_labels(Number_of_sub_boxes/10,1)],axis=0)
    print('X_test : ',X_test.shape)
    print('y_test : ',y_test.shape)


    if ml_algorithm == 'cnn' or ml_algorithm == 'both':

        print('CNNs training started')

        start_cnn = time.time()


        model = model()

        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6)
        opt = Adam(lr=0.0002, beta_1=0.9)
        model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])

        filepath="weights_best.h5"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,cooldown=0,
                                    patience=3,mode='max', min_lr=0.000001, verbose=1)

        history = model.fit(X_train,y_train, callbacks=[early_stopping_cb, model_checkpoint_callback, reduce_lr],shuffle=True, epochs=50,batch_size=32,verbose = 1,validation_data = (X_val,y_val)) #w = -0.9

        model.load_weights("weights_best.h5")
        test_acc = model.evaluate(X_test,y_test)
        print('accuarcy on test data: ', test_acc[1])


        # save results

        hist_df = pd.DataFrame(history.history) 
        hist_csv_file = 'history_cnn.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)


        hist_df_eval = pd.DataFrame(test_acc) 
        hist_csv_eval = 'evaluate_cnn.csv'
        with open(hist_csv_eval, mode='w') as f:
            hist_df_eval.to_csv(f)    


        # save model & summary

        model.summary()

        # prediction and confusion matrix
        pred = model.predict(X_test)

        print("confusion matrix for CNN: ",confusion_matrix(y_test,np.round(abs(pred))))
        confusion = pd.DataFrame(confusion_matrix(y_test,np.round(abs(pred)))) 
        confusion_csv_file = 'confusion_matrix_cnn.csv'
        with open(confusion_csv_file, mode='w') as f:
            confusion.to_csv(f)

        print("classification report for CNN: ",classification_report(y_test,np.round(abs(pred))))
        classification = pd.DataFrame(classification_report(y_test,np.round(abs(pred)),output_dict = True)).transpose() 
        classification_csv_file = 'classification_report_cnn.csv'
        with open(classification_csv_file, mode='w') as f:
            classification.to_csv(f)

        print("accuracy score for CNN: ",accuracy_score(y_test,np.round(abs(pred))))
    

        end_cnn = time.time()
        print('end of cnn training')
        print("---------------------------------------------------------------------------")
        print(f"The time of execution of CNN is : {end_cnn-start_cnn} seconds")
        print(f"CPU-hour for cnn is : {cpu_num*((end_cnn-start_cnn)/3600)}")



    if ml_algorithm == 'rf' or ml_algorithm == 'both':

        print('Random forests classification')

        # data providing
        shape_train = X_train.shape[0]
        shape_test = X_test.shape[0]

        Lambda_cdm_train = X_train[0:shape_train/2]
        w_98_train = X_train[shape_train/2:shape_train]

        Lambda_cdm_test = X_test[0:shape_test/2]
        w_98_test = X_test[shape_test/2:shape_test]

        # convert to power spectrum
        # varaibles

        q1t = []
        q2t = []
        q3t = []
        q4t = []
        q1 = []
        q2 = []
        q3 = []
        q4 = []

        # convert to power

        for i in range(0,shape_train/2,1):
            Pk = PKL.Pk(np.squeeze(Lambda_cdm_train[i], axis=3),sub_box/frac, axis = 0, MAS = 'CIC',  verbose = True)
            k = Pk.k3D
            Pk = Pk.Pk[:,0]
        # k = k[:11]
        # Pk = Pk[:11]
            q1t.append(k)
            q2t.append(Pk)

        q1t = np.array(q1t)
        q2t = np.array(q2t)


        for i in range(0,shape_train/2,1):
            Pk = PKL.Pk(np.squeeze(w_98_train[i], axis=3),sub_box/frac, axis = 0, MAS = 'CIC',  verbose = True)
            k = Pk.k3D
            Pk = Pk.Pk[:,0]
            q3t.append(k)
            q4t.append(Pk)

        q3t = np.array(q3t)
        q4t = np.array(q4t)


        for i in range(0,shape_test/2,1):
            Pk = PKL.Pk(np.squeeze(Lambda_cdm_test[i], axis=3),sub_box/frac, axis = 0, MAS = 'CIC',  verbose = True)
            k = Pk.k3D
            Pk = Pk.Pk[:,0]
            q1.append(k)
            q2.append(Pk)

        q1 = np.array(q1)
        q2 = np.array(q2)


        for i in range(0,shape_test/2,1):
            Pk = PKL.Pk(np.squeeze(w_98_test[i], axis=3),sub_box/frac, axis = 0, MAS = 'CIC',  verbose = True)
            k = Pk.k3D
            Pk = Pk.Pk[:,0]
        # k = k[:11]
        # Pk = Pk[:11]
            q3.append(k)
            q4.append(Pk)

        q3 = np.array(q3)
        q4 = np.array(q4)

        # training data


        x1 = pd.DataFrame(q2t)
        y1 = pd.DataFrame(q4t)

        e1 = pd.DataFrame(y_train[0:shape_train/2])
        e2 = pd.DataFrame(y_train[shape_train/2:shape_train])

        x1e1 = pd.concat((x1,e1),axis = 1)
        y1e2 = pd.concat((y1,e2),axis = 1)
        zere = pd.concat((x1e1,y1e2),axis = 0)

        # test data


        x1_test = pd.DataFrame(q2)
        y1_test = pd.DataFrame(q4)

        e1_test = pd.DataFrame(y_test[0:shape_test/2])
        e2_test = pd.DataFrame(y_test[shape_test/2:shape_test])

        x1e1_test = pd.concat((x1_test,e1_test),axis = 1)
        y1e2_test = pd.concat((y1_test,e2_test),axis = 1)
        zere_test = pd.concat((x1e1_test,y1e2_test),axis = 0)

        if sub_box == 16:

            zere = zere.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','class'],axis = 1)

            zere_test = zere_test.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','class'],axis = 1)

            rfc_train_X = zere[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13']]
            rfc_train_y = zere['class']

            rfc_test_X = zere_test[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13']]
            rfc_test_y = zere_test['class']

        elif sub_box == 32:
            zere = zere.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','class'],axis = 1)

            zere_test = zere_test.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','class'],axis = 1)

            rfc_train_X = zere[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27']]
            rfc_train_y = zere['class']

            rfc_test_X = zere_test[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27']]
            rfc_test_y = zere_test['class']

        elif sub_box == 128:
            zere = zere.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55','pk56','pk57','pk58','pk59','pk60','pk61','pk62','pk63','pk64','pk65','pk66','pk67','pk68','pk69','pk70','pk71','pk72','pk73','pk74','pk75','pk76','pk77','pk78','pk79','pk80','pk81','pk82','pk83','pk84','pk85','pk86','pk87','pk88','pk89','pk90','pk91','pk92','pk93','pk94','pk95','pk96','pk97','pk98','pk99','pk100','pk101','pk102','pk103','pk104','pk105','pk106','pk107','pk108','pk109','pk110','class'],axis = 1)

            zere_test = zere_test.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55','pk56','pk57','pk58','pk59','pk60','pk61','pk62','pk63','pk64','pk65','pk66','pk67','pk68','pk69','pk70','pk71','pk72','pk73','pk74','pk75','pk76','pk77','pk78','pk79','pk80','pk81','pk82','pk83','pk84','pk85','pk86','pk87','pk88','pk89','pk90','pk91','pk92','pk93','pk94','pk95','pk96','pk97','pk98','pk99','pk100','pk101','pk102','pk103','pk104','pk105','pk106','pk107','pk108','pk109','pk110','class'],axis = 1)

            rfc_train_X = zere[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55','pk56','pk57','pk58','pk59','pk60','pk61','pk62','pk63','pk64','pk65','pk66','pk67','pk68','pk69','pk70','pk71','pk72','pk73','pk74','pk75','pk76','pk77','pk78','pk79','pk80','pk81','pk82','pk83','pk84','pk85','pk86','pk87','pk88','pk89','pk90','pk91','pk92','pk93','pk94','pk95','pk96','pk97','pk98','pk99','pk100','pk101','pk102','pk103','pk104','pk105','pk106','pk107','pk108','pk109','pk110','class']]
            rfc_train_y = zere['class']

            rfc_test_X = zere_test[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55','pk56','pk57','pk58','pk59','pk60','pk61','pk62','pk63','pk64','pk65','pk66','pk67','pk68','pk69','pk70','pk71','pk72','pk73','pk74','pk75','pk76','pk77','pk78','pk79','pk80','pk81','pk82','pk83','pk84','pk85','pk86','pk87','pk88','pk89','pk90','pk91','pk92','pk93','pk94','pk95','pk96','pk97','pk98','pk99','pk100','pk101','pk102','pk103','pk104','pk105','pk106','pk107','pk108','pk109','pk110','class']]
            rfc_test_y = zere_test['class']  

        else:
            zere = zere.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55','class'],axis = 1)

            zere_test = zere_test.set_axis(['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55','class'],axis = 1)

            rfc_train_X = zere[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55']]
            rfc_train_y = zere['class']

            rfc_test_X = zere_test[['pk1','pk2','pk3','pk4','pk5','pk6','pk7','pk8','pk9','pk10','pk11','pk12','pk13','pk14','pk15','pk16','pk17','pk18','pk19','pk20','pk21','pk22','pk23','pk24','pk25','pk26','pk27','pk28','pk29','pk30','pk31','pk32','pk33','pk34','pk35','pk36','pk37','pk38','pk39','pk40','pk41','pk42','pk43','pk44','pk45','pk46','pk47','pk48','pk49','pk50','pk51','pk52','pk53','pk54','pk55']]
            rfc_test_y = zere_test['class']



        # standardscater

        sc = StandardScaler()
        X_train_st = sc.fit_transform(rfc_train_X)
        X_test_st = sc.transform(rfc_test_X)

        # Random forests tarining
        print('RFs training started')

        start_rfc = time.time()

        regressor = RandomForestRegressor(n_estimators = 1000,max_depth=10,random_state=0)
        regressor.fit(X_train_st, rfc_train_y)
        y_pred = regressor.predict(X_test_st)
        y_pred = np.array(y_pred)
        y_pred = pd.DataFrame(y_pred)

        # results

        print("confusion matrix for RFC: ",confusion_matrix(rfc_test_y,np.round(abs(y_pred))))
        confusion_rfc = pd.DataFrame(confusion_matrix(rfc_test_y,np.round(abs(y_pred)))) 
        confusion_rfc_csv_file = 'confusion_rfc_0.95_N128_1.csv'
        with open(confusion_rfc_csv_file, mode='w') as f:
            confusion_rfc.to_csv(f)


        print("classifiation report for RFC: ",classification_report(rfc_test_y,np.round(abs(y_pred))))
        classification_rfc = pd.DataFrame(classification_report(rfc_test_y,np.round(abs(y_pred)),output_dict = True)).transpose() 
        classification_rfc_csv_file = 'classification_rfc_0.95_N128_1.csv'
        with open(classification_rfc_csv_file, mode='w') as f:
            classification_rfc.to_csv(f)

        print("accuracy score for RFC: ",accuracy_score(rfc_test_y,np.round(abs(y_pred))))
        #accuracy_score_rfc = pd.DataFrame(accuracy_score(y_test,np.round(abs(pred)))) 
        #accuracy_score_rfc_csv_file = 'accuracy_score_rfc_0.9.csv'
        #with open(accuracy_score_rfc_csv_file, mode='w') as f:
        #    accuracy_score_rfc.to_csv(f)

        end_rfc = time.time()    
    
        print("---------------------------------------------------------------------------")
        print(f"The time of execution of rfc is : {end_rfc-start_rfc} seconds")
        print(f"CPU-hour-rfc: {cpu_num*((end_rfc-start_rfc)/3600)}")

    else:
        print("please choose one of cnn, rf or both")
