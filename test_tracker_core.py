import unittest
import tempfile
import os
import json
from tracker_core import Model, ModelManager

class TestModels(unittest.TestCase):

    def test_valid_run(self):
        run = Model("ResNet18",98.7,0.012,97.6,0.015,10,32,0.01)
        self.assertIsInstance(run,Model)
        self.assertEqual(run.model_name,"ResNet18")
        self.assertEqual(run.training_accuracy,98.7)
        self.assertEqual(run.training_loss,0.012)
        self.assertEqual(run.validation_accuracy,97.6)
        self.assertEqual(run.validation_loss,0.015)
        self.assertEqual(run.epochs,10)
        self.assertEqual(run.batch_size,32)
        self.assertEqual(run.learning_rate,0.01)

    def test_training_accuracy_value_above_100(self):
        with self.assertRaises(ValueError):
            Model("ResNet18",150,0.012,97.6,0.015,10,32,0.01)

    def test_training_accuracy_as_string(self):
        with self.assertRaises(ValueError):
            Model("ResNet18","abc",0.012,97.6,0.015,10,32,0.01)

    def test_training_loss_as_zero(self):
        run = Model("ResNet18",98.2,0,97.6,0.012,10,32,0.01)
        self.assertIsInstance(run,Model)

    def test_validation_loss_as_zero(self):
        run = Model("ResNet18",98.2,0.015,97.6,0,10,32,0.01)
        self.assertIsInstance(run,Model)

    def test_training_loss_as_negative(self):
        with self.assertRaises(ValueError):
            Model("ResNet18",98.2,-0.1,97.6,0.1,10,32,0.01)

    def test_validation_loss_as_negative(self):
        with self.assertRaises(ValueError):
            Model("ResNet18",98.2,0.1,97.6,-0.1,10,32,0.01)

    def test_epochs_as_zero(self):
        with self.assertRaises(ValueError):
            Model("ResNet18",98.2,0.01,97.6,0.015,0,32,0.01)

    def test_learning_rate_as_zero(self):
        with self.assertRaises(ValueError):
            Model("ResNet18",98.2,0.012,97.6,0.015,10,32,0.0)

    def test_model_name_as_empty_string(self):
        with self.assertRaises(ValueError):
            Model("",98.2,0.012,97.6,0.015,10,32,0.01)


class TestModelManager(unittest.TestCase):
    def test_model_manager(self):
        r1 = Model("ResNet18",98.7,0.012,97.6,0.015,10,32,0.01)
        r2 = Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015)
        manager = ModelManager()
        manager.add_model(r1)
        manager.add_model(r2)
        runs = manager.get_models()
        self.assertIn(r1,runs)
        self.assertIn(r2,runs)
        self.assertEqual(len(runs),2)

    def test_invalid_run_in_manager(self):
        with self.assertRaises(TypeError):
            ModelManager().add_model("")

    def test_copy_of_runs(self):
        manager = ModelManager()
        manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        runs = manager.get_models()
        runs.append(Model("ResNet16",99.7,0.017,98.6,0.018,10,32,0.015))
        runs = manager.get_models()
        self.assertEqual(len(runs),1)

class TestLoadSave(unittest.TestCase):
    def test_round_trip(self):
        manager = ModelManager()
        manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        run = manager.get_models()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"test.json")
            manager.save_json(path)
            new_manager = ModelManager()
            new_manager.load_json(path)
            loaded_run = new_manager.get_models()
            self.assertEqual(run,loaded_run)

    def test_invalid_file(self):
        manager = ModelManager()
        original_run = Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015)
        manager.add_model(original_run)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"bad_json.json")
            with open(path,"w",encoding="utf-8") as f:
                json.dump(
                            [
                                {
                                    "model_name": "ResNet18",
                                    "training_accuracy": 150,
                                    "training_loss": 0.12,
                                    "validation_accuracy": 87.5,
                                    "validation_loss": 0.15,
                                    "epochs": 10,
                                    "batch_size": 32,
                                    "learning_rate": 0.01
                                }
                            ], f)
            with self.assertRaises(ValueError):
                manager.load_json(path)

            self.assertEqual([original_run],manager.get_models())

    def test_file_not_found(self):
        manager = ModelManager()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"missing_json.json")
            with self.assertRaises(FileNotFoundError):
                manager.load_json(path)