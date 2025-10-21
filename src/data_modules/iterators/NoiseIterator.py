import torch
import numpy as np
import pickle
import threading
import queue
from random import choice
import logging



class NoiseIteratorDisk:
    def __init__(self, noise_dir: str, queue_timeout: float = 0.1, 
                 num_workers: int = 2, buffer_size: int = 1000):
        
        self.noise_dir = noise_dir
        self.queue_timeout = queue_timeout
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self.noises = self._preload_from_pickle()

        # Start worker threads
        self.workers = []
        for worker_id in range(num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(worker_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def _preload_from_pickle(self):
        """Load precomputed pickle data"""
        pkl_dir = self.noise_dir + ".pkl"
        try:
            with open(pkl_dir, "rb") as p:
                arr = pickle.load(p)
            return arr
        except Exception as e:
            logging.error(f"Failed to load pickle data: {e}")

    def _worker(self, worker_id):
        """Optimized worker thread with better batching and error handling"""
        while not self._stop_event.is_set():
            try:
                data = choice(self.noises)
                try:
                    self.queue.put(data, timeout=self.queue_timeout)
                except queue.Full:
                    continue
                                    
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        print(self.queue.qsize(), flush = True)
        try:
            return self.queue.get(timeout=0.1)
        except queue.Empty:
            if self._stop_event.is_set():
                raise StopIteration
            self.__next__()

    def stop(self):
        self._stop_event.set()
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
                    
        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.stop()


class NoiseIteratorPickle:
    def __init__(self, noise_dir: str, queue_timeout: float = 0.1, 
                 num_workers: int = 2, buffer_size: int = 1000):
        
        self.noise_dir = noise_dir
        self.queue_timeout = queue_timeout
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self.noises = self._preload_from_pickle()

        # Start worker threads
        self.workers = []
        for worker_id in range(num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(worker_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def _preload_from_pickle(self):
        """Load precomputed pickle data"""
        pkl_dir = self.noise_dir + ".pkl"
        try:
            with open(pkl_dir, "rb") as p:
                arr = pickle.load(p)
            return arr
        except Exception as e:
            logging.error(f"Failed to load pickle data: {e}")

    def _worker(self, worker_id):
        """Optimized worker thread with better batching and error handling"""
        while not self._stop_event.is_set():
            try:
                data = choice(self.noises)
                try:
                    self.queue.put(data, timeout=self.queue_timeout)
                except queue.Full:
                    continue
                                    
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        print(self.queue.qsize(), flush = True)
        try:
            return self.queue.get(timeout=0.1)
        except queue.Empty:
            if self._stop_event.is_set():
                raise StopIteration
            self.__next__()

    def stop(self):
        self._stop_event.set()
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
                    
        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.stop()
