import datetime
import tensorflow as tf

from logs import log_dir

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_dir = log_dir + '/tf_summaries/' + current_time + '/train'


class SummaryWriter:
    def __init__(self, summary_dir=train_summary_dir):
        self.summary_dir = summary_dir
        self.summary_writer = self.get_summary_writer()

    def get_summary_writer(self):
        summary_writer = tf.summary.create_file_writer(self.summary_dir)

        return summary_writer

    def write_summary(self, name, data, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name=name, data=data, step=step)
