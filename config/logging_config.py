# central logging setup for entire project
# without centralized logging, every developer will set logging differently
# print() or logging.basicConfig()

# every log will be uniform - centralized logging setup
# timestamp | log level info | which file it came from | message

import logging # built-in python library - log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
import sys # to write all logs to stdout so that cloud run captures them automatically
# cloud run will read stdout and sends it to google cloud logging
# no local log files needed - cloud run handles storage

def setup_logging(name: str) -> logging.Logger:
    """
    this function is to be called at the top of every .py file
    each file will get its own logger, namespaced to that file path
    this means log lines will show which file generated a particular log
    """
    logger = logging.getLogger(name)
    # getLogger(name) either creates a new logger or returns an existing one
    # if setup_logging called two times for a particular name, getLogger(name) will NOT create a new logger again
    # prevents duplicacy of logger for a particular name
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    # set minimum log level to INFO

    # handler
    handler = logging.StreamHandler(sys.stdout)
    # StreamHandler sends log lines to the stdout stream
    
    # formatter
    # fmt defines how each log line is formatted
    # asctime -> timestamp formatted by datefmt below
    # 2026-03-21 22:12:56
    # levelname -> log level - 8s means add spaces to fill up the 8 characters
    # name -> logger name - which file has logged in [module path (example: ingestion.clinical_trials_client)]
    # message -> actual message which passed to logger.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # attach the formatter to the handler
    handler.setFormatter(formatter)
    # handler now knows how to format each line before printing
    # attaching the handler to the logger
    logger.addHandler(handler)

    # when called logger.info("something"), it flows: 
    # logger -> handler -> formatter -> stdout -> terminal

    logger.propagate = False
    # propagate controls whether the log messages travel upto the root logger after being handled here
    # logger.propagate = False stops this propagation
    return logger
    





