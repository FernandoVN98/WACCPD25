# Classification
Inside this folder one can find the experiments included in the paper: "Scalable Neural Network Training: Distributed Data-Parallel Approaches". 
Inside the **Figures** folder one can find the training curves obtained using different training strategies. The folder **Generate_Dataset** contains the script used to load the CINIC-10 dataset and transform it 
into PyTorch tensors that were used in the experiments. 

There are two more folders: **Training_Losses** and **Times**

## Training Losses
The experiments contained in this folder try to replicate the figures containing the training losses contained inside the folder **Figures** and in the article. The scripts inside ./Training_Losses/Asynchronous_Training/ are the ones used to obtain the data that is shown in Figure 2 in the article. 
In order to reproduce any of these experiments one should go inside the corresponding folder, f. example:
$ cd ./Training_Losses/Asynchronous_Training/4_workers
Then, specify the path to the corresponding dislib, in this case the dislib contained in ./Training_Losses/Asynchronous_Training:
$ export PATH_TO_DISLIB=....
Finally, launch the execution using the bash script:
./launch_nn_double.sh $PATH_TO_X_TRAIN $PATH_TO_Y_TRAIN $PATH_TO_X_TEST $PATH_TO_Y_TEST

## Times
