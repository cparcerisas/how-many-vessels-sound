from ultralytics import YOLO
import os
import json
import pandas as pd
import datetime

import preprocess_data


def compute_unique_time(df):
    df = df.sort_values('x')
    total_presence_time = 0
    previous_row_end = 0
    for i, row in df.iterrows():
        current_row_start = row.x - row.width/2
        current_row_end = row.x + row.width/2

        if i == 0:
            total_presence_time += row.width
        elif current_row_start < previous_row_end:
            if current_row_end > previous_row_end:
                total_presence_time += current_row_end - previous_row_end
        else:
            total_presence_time += row.width

        previous_row_end = current_row_end

    return total_presence_time


if __name__ == '__main__':
    config_path = './images_config_test.json'
    f = open(config_path)
    config = json.load(f)

    ds = preprocess_data.YOLODataset(config)
    predictions_folder = ds.dataset_folder.joinpath('predictions')
    labels_path = predictions_folder.joinpath('labels')

    if not predictions_folder.joinpath('labels').exists():
        model_path = input('Where is the model to predict?')
        model = YOLO(model_path)
        os.mkdir(predictions_folder)
        os.mkdir(labels_path)
        print('creating spectrograms...')
        ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                               labels_path=labels_path)

    results_df = pd.DataFrame(columns=['datetime', 'total_percentage_ship', 'percentage_shipA', 'percentage_shipB', 'n_shipsA', 'n_shipsB'])
    for chunk_predictions in labels_path.glob('*.txt'):
        detections = pd.read_table(chunk_predictions, header=None, sep=' ', names=['class', 'x', 'y', 'width', 'height', 'confidence'])
        detections_a = detections.loc[detections['class'] == 0]
        detections_b = detections.loc[detections['class'] == 1]
        timestamp_str, offset = chunk_predictions.stem.split('_')
        _, timestamp_str = timestamp_str.split('.')

        timestamp = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        timestamp += datetime.timedelta(seconds=float(offset) * ds.duration)
        results_df.loc[len(results_df)] = ({'datetime': timestamp,
                                        'total_percentage_ship': compute_unique_time(detections),
                                        'percentage_shipA': compute_unique_time(detections_a),
                                        'percentage_shipB': compute_unique_time(detections_b),
                                        'n_shipsA': len(detections_a),
                                        'n_shipsB': len(detections_b)})

    results_df.to_csv(ds.dataset_folder.joinpath('results_detections.csv'))




