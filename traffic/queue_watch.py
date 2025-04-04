from multiprocessing import Process, Queue, Event
import time
from queue import Empty
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueueWatch(Process):
    def __init__(self, queues, queue_names, processes):
        super().__init__(name="QueueWatch")
        self.queues = queues
        self.queue_names = queue_names
        self.stop_event = Event()
        self.processes = processes
        
    def set_config(self, buffer_size=6):
        self.buffer_size = buffer_size
        self.qsize_buffer = [0] * (self.buffer_size - 1) 
        self.qsize_buffer.append(1)
        
    def run(self):
        while not self.stop_event.is_set():
            string = ""
            qsize_sum = 0
            for queue, name in zip(self.queues, self.queue_names):
                
                sub_string = name + ": " + str(queue.qsize())
                string += sub_string + " | "
                
                qsize_sum += queue.qsize()
                
            self.qsize_buffer.pop(0)
            self.qsize_buffer.append(qsize_sum)

            logger.info(string)
            time.sleep(5)
                
            if sum(self.qsize_buffer) == 0:
                logger.info(f"All queues empty, job done")
                self.stop_event.set()
            

            