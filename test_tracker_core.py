import unittest
import tempfile
import os
import json
from tracker_core import Model, ModelManager, merge_sort, merge

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
        self.assertEqual({1:r1,2:r2},runs)
        self.assertEqual(len(runs),2)

    def test_invalid_run_in_manager(self):
        with self.assertRaises(TypeError):
            ModelManager().add_model("")

    def test_copy_of_runs(self):
        manager = ModelManager()
        manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        runs = manager.get_models()
        runs[2] = Model("ResNet16",99.7,0.017,98.6,0.018,10,32,0.015)
        runs = manager.get_models()
        self.assertEqual(len(runs),1)

class TestLoadSave(unittest.TestCase):
    def test_json_round_trip(self):
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

    def test_json_invalid_file(self):
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

            self.assertEqual({1:original_run},manager.get_models())

    def test_json_file_not_found(self):
        manager = ModelManager()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"missing_json.json")
            with self.assertRaises(FileNotFoundError):
                manager.load_json(path)

    def test_csv_round_trip(self):
        manager = ModelManager()
        manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        runs = manager.get_models()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"runs.csv")
            manager.save_csv(path)
            new_manager = ModelManager()
            new_manager.load_csv(path)
            loaded_runs = new_manager.get_models()
            self.assertEqual(runs,loaded_runs)

    def test_csv_invalid_file(self):
        manager = ModelManager()
        original_run = Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015)
        manager.add_model(original_run)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"bad_run.csv")
            with open(path,"w",newline="",encoding="utf-8") as f:
                f.write("model_name,training_accuracy,training_loss,validation_accuracy,validation_loss,epochs,batch_size,learning_rate\n")
                f.write("ResNet18,abc,0.017,98.6,0.018,10,32,0.015\n")
            with self.assertRaises(ValueError):
                manager.load_csv(path)
            self.assertEqual({1:original_run},manager.get_models())

    def test_csv_file_not_found(self):
        manager = ModelManager()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"missing.csv")
            with self.assertRaises(FileNotFoundError):
                manager.load_csv(path)

    def test_csv_accuracy_out_of_range(self):
        manager = ModelManager()
        original_run = Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015)
        manager.add_model(original_run)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"bad_run.csv")
            with open(path,"w",newline="",encoding="utf-8") as f:
                f.write("model_name,training_accuracy,training_loss,validation_accuracy,validation_loss,epochs,batch_size,learning_rate\n")
                f.write("ResNet18,150,0.017,98.6,0.018,10,32,0.015\n")
            with self.assertRaises(ValueError):
                manager.load_csv(path)
            self.assertEqual({1:original_run},manager.get_models())

    def test_runs_ids_increment(self):
        manager = ModelManager()
        id1 = manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        id2 = manager.add_model(Model("VGG16",99.7,0.017,98.6,0.018,10,32,0.015))
        self.assertEqual(id1,1)
        self.assertEqual(id2,2)

    def test_next_id_after_load(self):
        manager = ModelManager()
        id1 = manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        id2 = manager.add_model(Model("VGG16",99.7,0.017,98.6,0.018,10,32,0.015))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"temp.json")
            manager.save_json(path)
            new_manager = ModelManager()
            new_manager.load_json(path)
            id3 = new_manager.add_model(Model("VGG19",99.7,0.017,98.6,0.018,10,32,0.015))
            self.assertEqual(id3,3)

    def test_failed_load_keeps_next_id(self):
        manager = ModelManager()
        manager.add_model(Model("ResNet32",99.7,0.017,98.6,0.018,10,32,0.015))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"bad.json")
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
            id2 = manager.add_model(Model("VGG19",99.7,0.017,98.6,0.018,10,32,0.015))
            self.assertEqual(id2,2)

class TestEditDelete(unittest.TestCase):
    def setUp(self):
        self.run1 = Model("ResNet18",98.5,0.012,97.6,0.015,10,32,0.01)
        self.run2 = Model("VGG19",98.5,0.012,97.6,0.015,10,32,0.01)
        self.manager = ModelManager()
        self.id1 = self.manager.add_model(self.run1)
        self.id2 = self.manager.add_model(self.run2)

    def test_delete_an_id(self):
        self.manager.delete_model(self.id1)
        self.assertEqual(self.manager.get_models(),{self.id2:self.run2})

    def test_delete_missing_id(self):
        with self.assertRaises(KeyError):
            self.manager.delete_model(99)

    def test_id_not_reused(self):
        self.manager.delete_model(self.id2)
        run3 = Model("AlexNet",98.5,0.012,97.6,0.015,10,32,0.01)
        id3 = self.manager.add_model(run3)
        self.assertEqual(id3,3)

    def test_update_existing_id(self):
        new_run = Model("AlexNet14",98.5,0.012,97.6,0.015,10,32,0.01)
        self.manager.update_model(self.id1,new_run)
        self.assertEqual(self.manager.get_models(),{self.id1:new_run,self.id2:self.run2})

    def test_edit_invalid_run_id(self):
        invalid_id = 99
        with self.assertRaises(KeyError):
            self.manager.update_model(invalid_id,self.run1)

    def test_edit_non_Model_object(self):
        invalid_run = "not a run"
        with self.assertRaises(TypeError):
            self.manager.update_model(self.id1,invalid_run)
        self.assertEqual(self.manager.get_models(),{self.id1: self.run1, self.id2: self.run2})

class TestMergeSort(unittest.TestCase):

    TEST_LISTS = [[5, 2, 8, 1], [], [3], [2, 2, 1], [1, 2, 3], [3, 2, 1]]
    DATA = [("a", 3), ("b", 1), ("c", 2)]
    REPEATED_DATA = [[("a", 2), ("b", 1), ("c", 2), ("d", 1)]]

    def test_matches_sorted(self): 
        for data in self.TEST_LISTS:
            with self.subTest(data=data):
                self.assertEqual(merge_sort(data),sorted(data))

    def test_matches_reverse_sorted(self):
        for data in self.TEST_LISTS:
            with self.subTest(data=data):
                self.assertEqual(merge_sort(data,reverse=True),sorted(data,reverse=True))

    def test_sort_with_key(self):
        self.assertEqual(merge_sort(self.DATA,key=lambda x: x[1]),sorted(self.DATA,key=lambda x: x[1]))

    def test_sort_is_stable(self):
        self.assertEqual(merge_sort(self.REPEATED_DATA,key=lambda x: x[1]),sorted(self.REPEATED_DATA,key=lambda x:x[1]))
        self.assertEqual(merge_sort(self.REPEATED_DATA,key=lambda x: x[1],reverse=True),sorted(self.REPEATED_DATA,key=lambda x:x[1],reverse=True))

    def test_merge_sort_doesnt_modify_original(self):
        data = [5,2,8,1]
        copy_of_data = data.copy()
        merge_sort(data)
        self.assertEqual(data,copy_of_data)

class TestSortRuns(unittest.TestCase):

    def setUp(self):
        self.manager = ModelManager()
        self.id1 = self.manager.add_model(Model("ResNet18",98.7,0.015,99.9,0.018,10,32,0.01))
        self.id2 = self.manager.add_model(Model("ResNet32",98.7,0.015,89.4,0.028,10,32,0.01))
        self.id3 = self.manager.add_model(Model("ResNet19",98.7,0.015,98.7,0.048,10,32,0.01))

    def test_sort_by_accuracy_high_to_low(self):
        results = self.manager.sort_runs("validation_accuracy")
        ids = [pair[0] for pair in results]
        self.assertEqual(ids,[self.id1,self.id3,self.id2])

    def test_sort_by_loss_low_to_high(self):
        results = self.manager.sort_runs("validation_loss")
        ids = [pair[0] for pair in results]
        self.assertEqual(ids,[self.id1,self.id2,self.id3])

    def test_invalid_metric_raises_an_error(self):
        with self.assertRaises(ValueError):
            self.manager.sort_runs("training_accuracy")

    def test_manager_unchanged_after_sorting(self):
        runs = self.manager.get_models()
        self.manager.sort_runs("validation_accuracy")
        self.assertEqual(list(runs.items()),list(self.manager.get_models().items()))





