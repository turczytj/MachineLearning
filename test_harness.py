import logging_mgr as logger
import child1_class as child1
import child2_class as child2

import os
import unittest

class ConfigurationTestCase(unittest.TestCase):
    def setUp(self):
        from importlib import import_module
        config_module = import_module('Configuration.configuration_mgr')

        self.config_mgr  = config_module.ConfigurationMgr()
        self.logging_mgr = logger.LoggingMgr(self.config_mgr.get_log_file_path())

    def tearDown(self):
        # Invoke dispose() here so it releases the file lock on the log file
        self.logging_mgr.dispose()

        # Clean up unit test
        log_file_path = self.config_mgr.get_log_file_path()
        os.remove(log_file_path)
        self.assertFalse(os.path.isfile(log_file_path))

    def test_required_config_entries_are_present(self):
        # Verify all required configuration entries are present
        self.assertNotEqual(self.config_mgr.get_runtime_environment(), None)
        self.assertNotEqual(self.config_mgr.get_output_directory(), None)
        self.assertNotEqual(self.config_mgr.get_logging_level(), None)

class LoggingTestCase(unittest.TestCase):
    def setUp(self):
        self.config_mgr  = cm.ConfigurationMgr()
        self.logging_mgr = logger.LoggingMgr(self.config_mgr.get_log_file_path())

    def tearDown(self):
        # Invoke dispose() here so it releases the file lock on the log file
        self.logging_mgr.dispose()

        # Clean up unit test
        log_file_path = self.config_mgr.get_log_file_path()
        os.remove(log_file_path)
        self.assertFalse(os.path.isfile(log_file_path))

    def test_log_file_exists(self):
        log_file_path = self.config_mgr.get_log_file_path()
        self.assertTrue(os.path.isfile(log_file_path))

    def test_valid_write(self):
        # Log an ERROR entry
        self.logging_mgr.log_message('Inside LoggingTestCase.test_valid_write()', self.logging_mgr.ERROR_LOG_LEVEL)

        file_size = os.path.getsize(self.config_mgr.get_log_file_path())
        self.assertTrue(file_size > 0)

    def test_invalid_write(self):
        # Log an INFO entry. Since the default logging level is 'ERROR', if I try to write an 'INFO' msg, it will be ignored.
        self.logging_mgr.log_message('Inside LoggingTestCase.test_invalid_write()', self.logging_mgr.INFO_LOG_LEVEL)

        file_size = os.path.getsize(self.config_mgr.get_log_file_path())
        self.assertTrue(file_size == 0)

class InheritanceTestCase(unittest.TestCase):
    def setUp(self):
        self.feature_name = ''

    def tearDown(self):
        self.feature_name = ''

    def test_inheritance(self):
        self.feature_name = 'child1'
        feature = child1.child1_class(self.feature_name)
        salutation = feature.get_salutation()
        self.assertTrue(salutation.find('Greetings') >= 0)

        self.feature_name = 'child2'
        feature = child2.child2_class(self.feature_name)
        salutation = feature.get_salutation()
        self.assertTrue(salutation.find('Hello') >= 0)

    def test_child1(self):
        self.feature_name = 'child1'
        feature = child1.child1_class(self.feature_name)
        self.assertTrue(self.feature_name, feature.get_info())

    def test_child2(self):
        self.feature_name = 'child2'
        feature = child2.child2_class(self.feature_name)
        self.assertTrue(self.feature_name, feature.get_info())

# If running from the Python command line then execute unittest.main()
if __name__ == '__main__': #unittest.main()
    testsToRun = unittest.TestSuite()

    # The following are used to run specific tests at a time instead of the whole suite of unit tests

    # ConfigurationTestCase unit tests
    testsToRun.addTest(ConfigurationTestCase('test_required_config_entries_are_present'))

    # LoggingTestCase unit tests
    #testsToRun.addTest(LoggingTestCase('test_log_file_exists'))
    #testsToRun.addTest(LoggingTestCase('test_valid_write'))
    #testsToRun.addTest(LoggingTestCase('test_invalid_write'))

    # FeatureTestCase unit tests
    #testsToRun.addTest(InheritanceTestCase('test_inheritance'))
    #testsToRun.addTest(InheritanceTestCase('test_child1'))
    #testsToRun.addTest(InheritanceTestCase('test_child2'))

    unittest.TextTestRunner().run(testsToRun)
