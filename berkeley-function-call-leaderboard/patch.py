with open('bfcl_eval/model_handler/local_inference/base_oss_handler.py', 'r') as f:
    data = f.read()
with open('bfcl_eval/model_handler/local_inference/base_oss_handler.py', 'w') as f:
    f.write(data.replace('self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)', 'self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)\n        self.model_name = self.registry_name'))
print('Done patching')
