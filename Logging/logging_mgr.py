import logging
import datetime
import traceback as tb

class LoggingMgr:
    INFO_LOG_LEVEL  = logging.INFO
    ERROR_LOG_LEVEL = logging.ERROR

    def __init__(self, log_file_path, log_level='ERROR'):
        try:
            self.logging_mgr = logging.getLogger()

            actual_logging_level = self._lookup_log_level(log_level)
            self.logging_mgr.setLevel(actual_logging_level)

            handler = logging.FileHandler(log_file_path)
            handler.setLevel(actual_logging_level)
            handler.flush()
            self.logging_mgr.addHandler(handler)

        except Exception as e:
            # Use the Python logger directly to log the error
            logger = logging.getLogger()
            logger.exception('*** Exception caught in logger.LoggingMgr.__init__(): ' + str(e))

    def log_message(self, message, log_level):
        '''
        DESCRIPTION:
            This method logs the message passed in the first parameter to the log file passed
            to this class's Constructor. The message is only logged if the level passed in the
            second paramter is the same or higher than the level also specified in the Construtor.
            Note: the lowest log level is 'INFO' and the highest one is 'ERROR'.

        INPUTS:
            message:   Contains the string that will be written to the log file
            log_level: Indicates the level (e.g., 'INFO', or 'ERROR') of the message being logged

        OUTPUT:
            None
        '''
       
        try:
            if log_level == self.INFO_LOG_LEVEL:
                formatted_msg = str(datetime.datetime.now()) + ' - INFO - ' + message
                self._log_info_msg(formatted_msg)
            else:
                formatted_msg = str(datetime.datetime.now()) + ' - ERROR - ' + message
                self._log_error_msg(formatted_msg)

        except Exception as e:
            # Use the Python logger directly to log the error
            logger = logging.getLogger()
            logger.exception('*** Exception caught in logger.LoggingMgr.logMessage(): ' + str(e))

    def dispose(self):
        '''
        DESCRIPTION:
            This method must be invoked by the client who instantiated it. It cleans up any resources
            that it is hanging onto. specifically, it releases the open file handle to the log file.

        INPUTS:
            None

        OUTPUT:
            None
        '''

        try:
            for handler in self.logging_mgr.handlers:
                self.logging_mgr.removeHandler(handler)
                handler.flush()
                handler.close()

        except Exception as e:
            # Use the Python logger directly to log the error
            logger = logging.getLogger()
            logger.exception('*** Exception caught in logger.LoggingMgr.dispose(): ' + str(e))

    ###########################################################################
    #  Private Methods
    ###########################################################################

    def _lookup_log_level(self, log_level):
        if log_level == 'INFO':
            return logging.INFO
        else:
            return logging.ERROR

    def  _log_info_msg(self, message):
        self.logging_mgr.info(message)

    def  _log_error_msg(self, message):
        self.logging_mgr.error(message)
