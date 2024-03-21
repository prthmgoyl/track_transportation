# utils.py
import csv

def write_to_csv(csv_file_path, frame_number, object_type, track_id):
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([frame_number, object_type, track_id])
