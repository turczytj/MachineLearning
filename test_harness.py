################################################################################
#                                                                              #
# Author:      Todd Turczynski                                                 #
# Create Date: Oct 24, 2019                                                    #
#                                                                              #
# Description:                                                                 #
#   This app is used to test various Python capabilities along with a way to   #
#   learn new Machine Learning topics using a simple test harness.             #
#                                                                              #
################################################################################

import os
import sys
import unittest

class ConfigurationTestCase(unittest.TestCase):
    def setUp(self):
        from importlib import import_module

        self.config_module_name = 'Configuration.configuration_mgr'
        config_module = import_module(self.config_module_name)
        self.config_mgr  = config_module.ConfigurationMgr()

        self.logging_module_name = 'Logging.logging_mgr'
        logging_module = import_module(self.logging_module_name)
        self.logging_mgr = logging_module.LoggingMgr(self.config_mgr.get_log_file_path())

    def tearDown(self):
        # Delete the Configuration and Logging modules loaded in SetUp()
        del sys.modules[self.config_module_name]
        del sys.modules[self.logging_module_name]

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
        from importlib import import_module

        self.config_module_name = 'Configuration.configuration_mgr'
        config_module = import_module(self.config_module_name)
        self.config_mgr  = config_module.ConfigurationMgr()

        self.logging_module_name = 'Logging.logging_mgr'
        logging_module = import_module(self.logging_module_name)
        self.logging_mgr = logging_module.LoggingMgr(self.config_mgr.get_log_file_path())

    def tearDown(self):
        # Delete the Configuration and Logging modules loaded in SetUp()
        del sys.modules[self.config_module_name]
        del sys.modules[self.logging_module_name]

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
    CHILD1_NAME = 'Nick'
    CHILD2_NAME = 'Alex'

    def test_inheritance(self):
        from Inheritance.child1 import Child1
        from Inheritance.child2 import Child2

        child_name = self.CHILD1_NAME
        my_child = Child1(child_name)
        salutation = my_child.get_salutation()
        self.assertTrue(salutation.find('Greetings') >= 0)

        child_name = self.CHILD2_NAME
        my_child = Child2(child_name)
        salutation = my_child.get_salutation()
        self.assertTrue(salutation.find('Hello') >= 0)

    def test_child1(self):
        from Inheritance.child1 import Child1

        child_name = self.CHILD1_NAME
        my_child = Child1(child_name)
        self.assertTrue(child_name, my_child.get_info())

    def test_child2(self):
        from Inheritance.child2 import Child2

        child_name = self.CHILD2_NAME
        my_child = Child2(child_name)
        self.assertTrue(child_name, my_child.get_info())

class KerasToTFTestCase(unittest.TestCase):
    def test_check_version(self):
        from KerasToTensorFlow.keras_to_tf import get_tf_version

        tf_version = get_tf_version()
        self.assertNotEqual(tf_version, None)

    def test_create_mlp_for_binary_classification(self):
        from KerasToTensorFlow.keras_to_tf import create_mlp_for_binary_classification

        model_accuracy = create_mlp_for_binary_classification()
        self.assertTrue(model_accuracy > 0.90)

    def test_create_mlp_for_multiclass_classification(self):
        from KerasToTensorFlow.keras_to_tf import create_mlp_for_multiclass_classification

        model_accuracy = create_mlp_for_multiclass_classification()
        self.assertTrue(model_accuracy > 0.90)

    def test_create_mlp_for_regression_predictions(self):
        from KerasToTensorFlow.keras_to_tf import create_mlp_for_regression_predictions
        from KerasToTensorFlow.keras_to_tf import make_mlp_regression_prediction

        model = create_mlp_for_regression_predictions()
        self.assertNotEqual(model, None)

        # The prediction should be about $150,000. Let's assume within 10% is good
        data = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
        prediction = make_mlp_regression_prediction(model, data)

        # unit is in $1000s
        self.assertTrue(prediction > 135.000)
        self.assertTrue(prediction < 165.000)


# If running from the Python command line then execute unittest.main()
if __name__ == '__main__': #unittest.main()
    testsToRun = unittest.TestSuite()

    # The following are used to run specific tests at a time instead of the whole suite of unit tests

    # ConfigurationTestCase unit tests
    #testsToRun.addTest(ConfigurationTestCase('test_required_config_entries_are_present'))

    # LoggingTestCase unit tests
    #testsToRun.addTest(LoggingTestCase('test_log_file_exists'))
    #testsToRun.addTest(LoggingTestCase('test_valid_write'))
    #testsToRun.addTest(LoggingTestCase('test_invalid_write'))

    # FeatureTestCase unit tests
    #testsToRun.addTest(InheritanceTestCase('test_inheritance'))
    #testsToRun.addTest(InheritanceTestCase('test_child1'))
    #testsToRun.addTest(InheritanceTestCase('test_child2'))

    # KerasToTFTestCase unit tests
    #testsToRun.addTest(KerasToTFTestCase('test_check_version'))
    #testsToRun.addTest(KerasToTFTestCase('test_create_mlp_for_binary_classification'))
    #testsToRun.addTest(KerasToTFTestCase('test_create_mlp_for_multiclass_classification'))
    testsToRun.addTest(KerasToTFTestCase('test_create_mlp_for_regression_predictions'))
    
    unittest.TextTestRunner().run(testsToRun)
