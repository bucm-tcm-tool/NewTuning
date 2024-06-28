Base LLM:
	ChatGLM2-6b: https://huggingface.co/THUDM/chatglm2-6b
Dataset:
	Chinese Biomedical Language Understanding Evaluation: https://github.com/CBLUEbenchmark/CBLUE
How to use:
1. Define adapter
The adapter_type variable in the chatglm2/configuration_chatglm.py file defines the type of adapter (series, parallel), such as adapter_types="series".
2. Train through train.py:
(1) Define the adapter name that needs to be saved in the torch. save() function of train. py;
(2) Define the training file in the train_json_dir variable:
single inference method: train_json_dir = "./ data/single_inference_train_ids.json"
stepwise inference method: train_json_dir = "./ data/stepwise_inference_train_ids.json"
3. Testing of adapter
Run single_inference.py or stepwise inference.py to perform single or stepwise inferences
(1) Define the path to save test results in "jw=open()."
(2) Define the path of the adapter file in the adapter_path variable for loading.
(3) Define the path to save the test results for the variable json_dir in the prepressing/generatetest_file.py file and execute the generate() function.
(4) Compress the generated CMeEE-V2utest.json into a zip file and submit it to: https://tianchi.aliyun.com/dataset/95414/submission .
