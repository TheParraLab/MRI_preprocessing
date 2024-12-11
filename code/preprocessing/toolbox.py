import time
import logging
import os

def get_logger(name: str, save_dir: str = ''):
    # Check if save_dir exists
    if save_dir and save_dir[-1] != '/':
        save_dir += '/'
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(save_dir + name + '.log')
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class ProgressBar:
    # Class to create a progress bar
    # Will display a progress bar with the current progress, the current step, the status, and the estimated time remaining
    def __init__(self, total, splits=20, update_interval=1):
        self.total = total
        self.splits = splits
        self.current = 0
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update(index=0)

    def update(self, index = None, status=''):
       # with self.lock:
        if index is None:
            index = self.current + 1

        if index % self.update_interval != 0 and index != self.total:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if index > 0:
            avg_time_per_step = elapsed_time / index
            remaining_steps = self.total - index
            eta = avg_time_per_step * remaining_steps
        else:
            eta = 0

        current = int((index / self.total) * self.splits)
        current_progress = ''
        for i in range(self.splits):
            if i < current:
                current_progress += '■'
            else:
                current_progress += '□'

        eta_formatted = self.format_time(eta)
        print(f'\r {current_progress} | {index}/{self.total} | {status} | ETA: {eta_formatted} |', end='', flush=True)
        self.current = index

    @staticmethod
    def format_time(seconds):
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f'{hours:02}:{mins:02}:{secs:02}'

# Example usage
if __name__ == '__main__':
    import random
    total_steps = 100
    progress_bar = ProgressBar(total_steps)

    for i in range(total_steps):
        time.sleep(random.random()/2)  # Simulate work
        progress_bar.update(i + 1, status='Processing')