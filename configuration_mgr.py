import json
import os

class ConfigurationMgr:
    def __init__(self):
        CONFIGURATION_FILE = 'configuration.json'

        # Read app configuration file entries
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIGURATION_FILE)
        with open(config_file_path, 'r') as config_file_handle:
            app_config_settings = json.load(config_file_handle)

        # Load runtime configuration variables
        self.runtimeEnv      = app_config_settings['Environment']['Runtime']
        self.outputDirectory = app_config_settings['Environment']['OutputDirectory']

        # Load logging configuration variables
        self.log_level          = app_config_settings['Logging']['LogLevel']
        self.log_file_directory = app_config_settings['Logging']['LogFileDirectory']
        self.log_file_name      = app_config_settings['Logging']['LogFileName']
        self.log_file_path      = os.path.join(self.log_file_directory, self.log_file_name)

    def get_runtime_environment(self):
        return self.runtimeEnv

    def get_output_directory(self):
        return self.outputDirectory

    def get_logging_level(self):
        return self.log_level

    def get_log_file_path(self):
        return self.log_file_path

    ###########################################################################
    #  Private Methods
    ###########################################################################

    def _dump_config_entries(self, loggingMgr):
        """This is intended for debugging purposes. It is used to log all the properties to the log file"""

        loggingMgr.logMessage('\n\nDumping Config Entries', loggingMgr.INFO_LOG_LEVEL)

        loggingMgr.logMessage("Runtime Env: {0}".format(self.get_runtime_environment()), loggingMgr.INFO_LOG_LEVEL)
        loggingMgr.logMessage("Output Directory: {0}".format(self.get_output_directory()), loggingMgr.INFO_LOG_LEVEL)
        loggingMgr.logMessage("Log Level: {0}".format(self.get_logging_level()), loggingMgr.INFO_LOG_LEVEL)
        loggingMgr.logMessage("Log File Path: {0}".format(self.get_log_file_path()), loggingMgr.INFO_LOG_LEVEL)