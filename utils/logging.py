import logging
import logging.handlers
import multiprocessing

__all__ = ["setup_main_logging", "setup_worker_logging"]

def setup_main_logging(log_file_path):
    """
    Set up logging in the main process with a QueueListener.
    Returns the log queue and the listener (which must be started and stopped).
    """
    log_queue = multiprocessing.Queue(-1)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    listener = logging.handlers.QueueListener(log_queue, file_handler, console_handler)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    # Only add a handler for errors in the main process itself (optional)
    return log_queue, listener

def setup_worker_logging(log_queue):
    """
    Set up logging in a worker process to send all logs to the main process via the log_queue.
    """
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(queue_handler) 