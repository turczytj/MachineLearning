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
        self.config_mgr = config_module.ConfigurationMgr()

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
        self.config_mgr = config_module.ConfigurationMgr()

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
        data = [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]
        prediction = make_mlp_regression_prediction(model, data)

        # unit is in $1000s
        self.assertTrue(prediction > 135.000)
        self.assertTrue(prediction < 165.000)

    def test_create_cnn_for_image_classification(self):
        from KerasToTensorFlow.keras_to_tf import create_cnn_for_image_classification

        model_accuracy = create_cnn_for_image_classification()
        self.assertTrue(model_accuracy > 0.90)


class KerasRegressionTestCase(unittest.TestCase):
    def test_regression(self):
        from KerasToTensorFlow.keras_regression import calculate_mpg

        calculate_mpg()


class NLPTestCase(unittest.TestCase):
    def test_download_samples(self):
        from NLP.nlp import download_sample_data

        download_sample_data()

    def test_tokenize(self):
        from NLP.nlp import tokenize

        sentence = 'My name is George and I love NLP'
        words = tokenize(sentence)

        self.assertNotEqual(words, None)

        # Should be: ['My', 'name', 'is', 'George', 'and', 'I', 'love', 'NLP']
        self.assertEqual(len(words), 8)

    def test_remove_stop_words(self):
        from NLP.nlp import remove_stop_words

        sentence = "This is a sentence for removing stop words"
        words = remove_stop_words(sentence)

        self.assertNotEqual(words, None)

        # Should be: ['This', 'sentence', 'removing', 'stop', 'words']
        self.assertEqual(len(words), 5)

    def test_run_stemming_process(self):
        from NLP.nlp import run_stemming_process

        sentence = "cook cooks cooking cooked"
        words = run_stemming_process(sentence)

        self.assertNotEqual(words, None)

        # Should be: ['cook', 'cook', 'cook', 'cook']
        self.assertTrue(words[0] == 'cook')
        self.assertTrue(words[1] == 'cook')
        self.assertTrue(words[2] == 'cook')
        self.assertTrue(words[3] == 'cook')

    def test_run_word_embedding_process(self):
        from NLP.nlp import run_word_embedding_process

        status = run_word_embedding_process()
        self.assertTrue(status)

    def test_calculate_term_frequency(self):
        from NLP.nlp import calculate_term_frequency

        line_1 = "TF-IDF uses statistics to measure how important a word is to a particular document"
        line_2 = "The TF-IDF is perfectly balanced, considering both local and global levels of statistics for the target word."
        line_3 = "Words that occur more frequently in a document are weighted higher, but only if they're more rare within the whole document."
        text = [line_1, line_2, line_3]

        results = calculate_term_frequency(text)

        self.assertTrue(len(results) > 0)


class DataFrameTestCase(unittest.TestCase):
    def test_run(self):
        from DataFrame.DataFramePlayground import DataFramePlayground

        # Update Path env variable to access DataFramePlayground files
        sys.path.append('.\\DataFrame')

        playground = DataFramePlayground()
        playground.run()


class SMOTETestCase(unittest.TestCase):
    def test_run(self):
        from SMOTE.smote import run_main

        # Update Path env variable to access SMOTE files
        sys.path.append('.\\SMOTE')

        run_main()


class PyCaretTestCase(unittest.TestCase):
    def test_run(self):
        from PyCaret.playground import run_demo

        # Update Path env variable to access PyCaret files
        sys.path.append('.\\PyCaret')

        run_demo()


class ScikitLearnTipsTestCase(unittest.TestCase):
    def test_run(self):
        from Scikit_Learn_Tips.playground import run_demo

        # Update Path env variable to access PyCaret files
        sys.path.append('.\\Scikit_Learn_Tips')

        run_demo()


# If running from the Python command line then execute unittest.main()
if __name__ == '__main__':  # unittest.main()
    testsToRun = unittest.TestSuite()

    # The following are used to run specific tests at a time instead of the whole suite of unit tests

    # ConfigurationTestCase unit tests
    testsToRun.addTest(ConfigurationTestCase('test_required_config_entries_are_present'))

    # LoggingTestCase unit tests
    testsToRun.addTest(LoggingTestCase('test_log_file_exists'))
    testsToRun.addTest(LoggingTestCase('test_valid_write'))
    testsToRun.addTest(LoggingTestCase('test_invalid_write'))

    # FeatureTestCase unit tests
    testsToRun.addTest(InheritanceTestCase('test_inheritance'))
    testsToRun.addTest(InheritanceTestCase('test_child1'))
    testsToRun.addTest(InheritanceTestCase('test_child2'))

    # KerasToTFTestCase unit tests
    testsToRun.addTest(KerasToTFTestCase('test_check_version'))
    testsToRun.addTest(KerasToTFTestCase('test_create_mlp_for_binary_classification'))
    testsToRun.addTest(KerasToTFTestCase('test_create_mlp_for_multiclass_classification'))
    testsToRun.addTest(KerasToTFTestCase('test_create_mlp_for_regression_predictions'))
    testsToRun.addTest(KerasToTFTestCase('test_create_cnn_for_image_classification'))

    # KerasRegressionTestCase unit tests
    testsToRun.addTest(KerasRegressionTestCase('test_regression'))

    # NLPTestCase unit tests
    testsToRun.addTest(NLPTestCase('test_download_samples'))
    testsToRun.addTest(NLPTestCase('test_tokenize'))
    testsToRun.addTest(NLPTestCase('test_remove_stop_words'))
    testsToRun.addTest(NLPTestCase('test_run_stemming_process'))
    testsToRun.addTest(NLPTestCase('test_run_word_embedding_process'))
    testsToRun.addTest(NLPTestCase('test_calculate_term_frequency'))

    # DataFrameTestCase unit tests
    testsToRun.addTest(DataFrameTestCase('test_run'))

    # SMOTETestCase unit tests
    testsToRun.addTest(SMOTETestCase('test_run'))

    # PyCaretTestCase unit tests
    testsToRun.addTest(PyCaretTestCase('test_run'))

    # ScikitLearnTipsTestCase unit tests
    testsToRun.addTest(ScikitLearnTipsTestCase('test_run'))

    unittest.TextTestRunner().run(testsToRun)
