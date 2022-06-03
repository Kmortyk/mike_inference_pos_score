from src.sampler.tf_annotation import TFAnnotation
import tensorflow as tf


class TFRecordSampler:
    def __init__(self, tf_record_path, batch):
        self.count = sum(1 for _ in tf.data.TFRecordDataset(tf_record_path))
        self.batch = batch
        self.dataset = tf.data.TFRecordDataset(tf_record_path)
        self.sample = self.dataset.shuffle(self.count).take(batch).__iter__()
        self.cur_record_idx = 0

    def take(self):
        record = next(self.sample)
        annot = TFAnnotation()
        annot.parse_record(record)

        self.cur_record_idx += 1
        if self.cur_record_idx + 1 == self.batch:
            raise StopIteration

        return annot.cv_image(alpha=False)
