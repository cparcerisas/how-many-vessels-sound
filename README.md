# How many vessels can we hear and classify in a window of time? 

authors: Irene T. Roca and Clea Parcerisas

To run this code: 
1. create a virtualenvironment using poetry:

```bash
poetry install
```

To train a new model: 
1. Change the paths in the images_config.json file to match where your data should be. You can also change the spectrograms parameters. 
2. Create the training dataset. For this, you need to run the preprocess_data.py script
3. Change the paths on the yolo_config.yml file to match the paths to the output of the training dataset. You might want to split the spectrograms and labels you just generated in train and validation set.
4. Train the model by running the train_yolo.py script


To use the model: 
1. Run the detections_analysis.py script. The prompt will ask you to point to the path of the model you want to use for predictions.



